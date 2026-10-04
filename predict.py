#!/usr/bin/env python
r"""Segment unlabelled BraTS cases with a trained run and write one BraTS label map per case."""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _parse_regions(text, default, cast):
    r"""Parse 'WT=a,TC=b,ET=c' into a per-region dictionary."""
    out = {"WT": default, "TC": default, "ET": default}
    for item in filter(None, (text or "").split(",")):
        region, _, value = item.partition("=")
        if region.strip() not in out:
            raise ValueError(f"unknown region: {region}")
        out[region.strip()] = cast(value)
    return out


def main() -> int:
    r"""Run the model on every case folder under --input and save <case>.nii.gz label maps."""
    ap = argparse.ArgumentParser(description="Write BraTS label maps for unlabelled cases.")
    ap.add_argument("--run", required=True, help="run folder with config.yaml and best.pth")
    ap.add_argument("--input", required=True, help="folder of case folders (four modalities each)")
    ap.add_argument("--output", required=True, help="folder for <case>.nii.gz label maps")
    ap.add_argument("--modalities", default="t1n,t1c,t2w,t2f",
                    help="modality suffixes in the training channel order (default: BraTS 2023 names)")
    ap.add_argument("--thresholds", default="", metavar="WT=0.65,TC=0.65,ET=0.6",
                    help="per-region probability thresholds (default 0.5)")
    ap.add_argument("--min-component", default="", metavar="WT=400,TC=25,ET=400",
                    help="drop predicted components below this many voxels (default 0)")
    ap.add_argument("--ensemble-n", type=int, default=None,
                    help="stochastic passes per case (default: the run's evaluation.ensemble_n)")
    ap.add_argument("--no-tta", action="store_true", help="disable flip test-time augmentation")
    ap.add_argument("--et-label", type=int, default=3, help="label for enhancing tumour (3: 2023+, 4: 2021)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N cases")
    args = ap.parse_args()

    import nibabel as nib
    import numpy as np
    import torch

    from glo_nca.config import REGIONS, load_config
    from glo_nca.evaluation import paste_to_scan, predict_case, remove_small_components
    from glo_nca.trainer import build, set_seed

    thresholds = _parse_regions(args.thresholds, 0.5, float)
    min_component = _parse_regions(args.min_component, 0, int)
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

    cases = sorted(d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))
                   and any(f.endswith((".nii", ".nii.gz")) for f in os.listdir(os.path.join(args.input, d))))
    if args.limit:
        cases = cases[:args.limit]
    os.makedirs(args.output, exist_ok=True)
    print(f"{len(cases)} cases | best epoch {ck['ep']} | ensemble {ensemble} | tta {tta} | "
          f"thresholds {thresholds} | min component {min_component}", flush=True)

    for n, case in enumerate(cases, 1):
        out_path = os.path.join(args.output, f"{case}.nii.gz")
        if os.path.exists(out_path):
            continue   # resumable: finished cases are kept
        folder = os.path.join(args.input, case)
        files = [ds._find_modality_file(folder, case, m) for m in modalities]
        ref = nib.load(files[0])
        raw = np.stack([nib.load(f).get_fdata() for f in files], axis=-1)
        bbox = ds._foreground_bbox(raw) if ds.use_foreground_crop else None
        x0, x1, y0, y1, z0, z1 = bbox if bbox is not None else (0, raw.shape[0], 0, raw.shape[1], 0, raw.shape[2])
        crop = raw[x0:x1, y0:y1, z0:z1, :]
        img = np.stack([ds._resize_to(crop[..., c], ds.size) for c in range(crop.shape[-1])], axis=-1)
        if ds.use_nonzero_norm:
            img = ds.nonzero_norm(img)
        data = (["_" + case + "_0"], torch.from_numpy(img[None].astype(np.float32)),
                torch.zeros((1, *ds.size, len(REGIONS))))
        with torch.no_grad():
            _, prob, _ = predict_case(agent, data, ensemble, tta)
        full = paste_to_scan(prob, bbox, raw.shape[:3])
        masks = {r: remove_small_components(full[..., i] >= thresholds[r], min_component[r])
                 for i, r in enumerate(REGIONS)}
        # Nested BraTS labels: edema inside WT, necrotic core inside TC, enhancing tumour on top.
        label = np.zeros(raw.shape[:3], np.uint8)
        label[masks["WT"]] = 2
        label[masks["TC"]] = 1
        label[masks["ET"]] = args.et_label
        out = nib.Nifti1Image(label, ref.affine, ref.header)
        out.set_data_dtype(np.uint8)
        tmp = out_path + ".tmp.nii.gz"
        nib.save(out, tmp)
        os.replace(tmp, out_path)
        if n % 10 == 0 or n == len(cases):
            print(f"  {n}/{len(cases)} cases written", flush=True)
    agent.exp.set_model_state("train")
    print("Saved to", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
