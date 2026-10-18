r"""Segment unlabelled BraTS cases with a trained run and write one BraTS label map per case."""
import argparse
import io
import os
import sys


def _parse_regions(text, default, cast):
    r"""Parse 'WT=a,TC=b,ET=c' into a per-region dictionary."""
    out = {"WT": default, "TC": default, "ET": default}
    for item in filter(None, (text or "").split(",")):
        region, _, value = item.partition("=")
        if region.strip() not in out:
            raise ValueError(f"unknown region: {region}")
        out[region.strip()] = cast(value)
    return out


def main(argv=None) -> int:
    r"""Run the model (or reuse saved probabilities) and save <case>.nii.gz label maps."""
    ap = argparse.ArgumentParser(description="Write BraTS label maps for unlabelled cases.")
    ap.add_argument("--run", help="run folder with config.yaml and best.pth (not needed with --from-probs)")
    ap.add_argument("--input", help="folder of case folders (four modalities each)")
    ap.add_argument("--output", required=True, help="folder for <case>.nii.gz label maps")
    ap.add_argument("--modalities", default="t1n,t1c,t2w,t2f",
                    help="modality suffixes in the training channel order (default: BraTS 2023 names)")
    ap.add_argument("--thresholds", default="", metavar="WT=0.65,TC=0.65,ET=0.6",
                    help="per-region probability thresholds (default 0.5)")
    ap.add_argument("--min-component", default="", metavar="WT=400,TC=25,ET=400",
                    help="drop predicted components below this many voxels (default 0)")
    ap.add_argument("--et-to-ncr", default="", metavar="RATIO,VOXELS",
                    help="relabel ET as necrotic core when it is below VOXELS and below RATIO x TC")
    ap.add_argument("--ensemble-n", type=int, default=None,
                    help="stochastic passes per case (default: the run's evaluation.ensemble_n)")
    ap.add_argument("--no-tta", action="store_true", help="disable flip test-time augmentation")
    ap.add_argument("--et-label", type=int, default=3, help="label for enhancing tumour (3: 2023+, 4: 2021)")
    ap.add_argument("--save-probs", default=None, metavar="DIR",
                    help="also save each case's probabilities, so post-processing can be redone without the model")
    ap.add_argument("--from-probs", default=None, metavar="DIR",
                    help="skip the model and post-process probabilities saved with --save-probs")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N cases")
    args = ap.parse_args(argv)

    import nibabel as nib
    import numpy as np

    from glo_nca.config import REGIONS
    from glo_nca.evaluation import decode_regions, masks_to_label, paste_to_scan

    thresholds = _parse_regions(args.thresholds, 0.5, float)
    min_component = _parse_regions(args.min_component, 0, int)
    et_rule = tuple(float(x) for x in args.et_to_ncr.split(",")) if args.et_to_ncr else None
    os.makedirs(args.output, exist_ok=True)

    def write_label(prob, bbox, shape, affine, header, out_path):
        r"""Threshold, clean and relabel one probability map, then save it as a BraTS label map."""
        # The same decoder as the internal evaluation with nested regions.
        masks = decode_regions(paste_to_scan(prob, bbox, shape), thresholds, min_component, et_rule)
        out = nib.Nifti1Image(masks_to_label(masks, args.et_label), affine, header)
        out.set_data_dtype(np.uint8)
        tmp = out_path + ".tmp.nii.gz"
        nib.save(out, tmp)
        os.replace(tmp, out_path)

    if args.from_probs:
        cases = sorted(f[:-4] for f in os.listdir(args.from_probs) if f.endswith(".npz"))[:args.limit]
        print(f"{len(cases)} cases from saved probabilities | thresholds {thresholds} | "
              f"min component {min_component} | et-to-ncr {et_rule}", flush=True)
        for n, case in enumerate(cases, 1):
            with np.load(os.path.join(args.from_probs, case + ".npz")) as z:
                bbox = tuple(int(v) for v in z["bbox"]) if z["bbox"].size else None
                header = nib.Nifti1Header.from_fileobj(io.BytesIO(z["header"].tobytes()))
                write_label(z["prob"].astype(np.float32), bbox, tuple(int(v) for v in z["shape"]),
                            z["affine"], header, os.path.join(args.output, f"{case}.nii.gz"))
            if n % 20 == 0 or n == len(cases):
                print(f"  {n}/{len(cases)} cases written", flush=True)
        print("Saved to", args.output)
        return 0

    import torch

    from glo_nca.config import load_config
    from glo_nca.evaluation import predict_case
    from glo_nca.trainer import build, set_seed

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    C = load_config(os.path.join(args.run, "config.yaml"))
    if len(modalities) != len(C.MODALITIES):
        print(f"FAILED: {len(modalities)} modalities given, the model expects {len(C.MODALITIES)}.")
        return 2
    ensemble = args.ensemble_n if args.ensemble_n is not None else C.ENSEMBLE_N
    tta = C.USE_TTA and not args.no_tta
    set_seed(C.SEED)
    ds, ca, agent, dev = build(C, args.input, args.run, ([], [], []))
    ds.MODALITIES = modalities
    ck = torch.load(os.path.join(args.run, "best.pth"), map_location=dev, weights_only=False)
    for m, sd in zip(ca, ck["m"]):
        m.load_state_dict(sd)
    agent.exp.set_model_state("test")
    if args.save_probs:
        os.makedirs(args.save_probs, exist_ok=True)

    cases = sorted(d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))
                   and any(f.endswith((".nii", ".nii.gz")) for f in os.listdir(os.path.join(args.input, d))))
    if args.limit:
        cases = cases[:args.limit]
    print(f"{len(cases)} cases | best epoch {ck['ep']} | ensemble {ensemble} | tta {tta} | "
          f"thresholds {thresholds} | min component {min_component} | et-to-ncr {et_rule}", flush=True)

    for n, case in enumerate(cases, 1):
        out_path = os.path.join(args.output, f"{case}.nii.gz")
        prob_path = os.path.join(args.save_probs, f"{case}.npz") if args.save_probs else None
        if os.path.exists(out_path) and (prob_path is None or os.path.exists(prob_path)):
            continue   # resumable: finished cases are kept
        folder = os.path.join(args.input, case)
        files = [ds._find_modality_file(folder, case, m) for m in modalities]
        ref = nib.load(files[0])
        raw = np.stack([nib.load(f).get_fdata() for f in files], axis=-1)
        bbox = ds._foreground_bbox(raw) if ds.use_foreground_crop else None
        x0, x1, y0, y1, z0, z1 = bbox if bbox is not None else (0, raw.shape[0], 0, raw.shape[1], 0, raw.shape[2])
        crop = raw[x0:x1, y0:y1, z0:z1, :]
        img = ds.normalize(ds.resize_image(crop))
        data = (["_" + case + "_0"], torch.from_numpy(img[None].astype(np.float32)),
                torch.zeros((1, *ds.size, len(REGIONS))))
        with torch.no_grad():
            _, prob, _ = predict_case(agent, data, ensemble, tta)
        if prob_path:
            np.savez_compressed(prob_path + ".tmp.npz", prob=prob.astype(np.float32),
                                bbox=np.array(bbox if bbox is not None else [], np.int32),
                                shape=np.array(raw.shape[:3], np.int32), affine=ref.affine,
                                header=np.frombuffer(ref.header.binaryblock, np.uint8))
            os.replace(prob_path + ".tmp.npz", prob_path)
        write_label(prob, bbox, raw.shape[:3], ref.affine, ref.header, out_path)
        if n % 10 == 0 or n == len(cases):
            print(f"  {n}/{len(cases)} cases written", flush=True)
    agent.exp.set_model_state("train")
    print("Saved to", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
