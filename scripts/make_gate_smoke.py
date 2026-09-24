#!/usr/bin/env python
r"""Build the pre-training gate's engineering smoke run (never the production run).

Selects a small deterministic subset of MASTER-TRAIN cases only -- one case per
subject, subjects in sorted order, skipping cases named in the data-quality
policy -- copies them into <stage>/data and writes <stage>/smoke_split.json and
<stage>/smoke_config.yaml:

    train 8 / validation 2 / test 2 cases, every one drawn from master TRAIN.

The 200 validation and 198 test cases of the production split are never staged,
so a smoke container that mounts only <stage>/data cannot read them. The smoke's
own "test" partition is two training cases that exercise the end-of-run
evaluation code; it is not the production test set.

The config is the production config with smoke-only overrides (experiment name,
epochs, split file, policy). The production config and split/master_split.json
are only read, never written.

Usage (paths as seen by the smoke container are given with --container-*):
    python scripts/make_gate_smoke.py --data-root /data --stage /stage \
        --container-data /data --container-stage /smoke
"""
import argparse
import copy
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import yaml  # noqa: E402

from src.experiment import datasource  # noqa: E402

N_TRAIN, N_VAL, N_TEST = 8, 2, 2
SMOKE_NAME = "GLO-NCA-GATE-SMOKE"
SMOKE_EPOCHS = 3


def select_cases(master, excluded):
    """Deterministic: first case (sorted) of each subject, subjects sorted, master-train only."""
    by_subject = {}
    for case in sorted(master["train"]):
        if case in excluded:
            continue
        by_subject.setdefault(datasource.subject_of(case), case)
    picked = [by_subject[s] for s in sorted(by_subject)][:N_TRAIN + N_VAL + N_TEST]
    if len(picked) < N_TRAIN + N_VAL + N_TEST:
        raise SystemExit(f"FAIL: only {len(picked)} eligible master-train subjects")
    return picked[:N_TRAIN], picked[N_TRAIN:N_TRAIN + N_VAL], picked[N_TRAIN + N_VAL:]


def verify(exp_dir, stage, split_path):
    """Check a finished smoke run: epochs once each, completed, no production val/test ids."""
    import csv
    master = datasource.load_master_split(split_path)
    with open(os.path.join(stage, "smoke_split.json"), encoding="utf-8") as fh:
        smoke = json.load(fh)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    with open(os.path.join(exp_dir, "metrics", "train.csv"), encoding="utf-8") as fh:
        epochs = [int(r["epoch"]) for r in csv.DictReader(fh)]
    check("every epoch recorded exactly once", epochs == list(range(1, SMOKE_EPOCHS + 1)), epochs)
    with open(os.path.join(exp_dir, "status.json"), encoding="utf-8") as fh:
        state = json.load(fh).get("state")
    check("run completed", state == "completed", state)
    with open(os.path.join(exp_dir, "split", "split.json"), encoding="utf-8") as fh:
        run_split = json.load(fh)
    used = set(run_split["train"]) | set(run_split["validation"]) | set(run_split["test"])
    check("run used the smoke split",
          used == set(smoke["train"]) | set(smoke["validation"]) | set(smoke["test"]))
    check("run cases all from master train", used <= set(master["train"]), f"{len(used)} cases")
    forbidden = set(master["validation"]) | set(master["test"])
    hits = set()
    for root, _, files in os.walk(exp_dir):
        for name in files:
            if name.endswith((".pth", ".pt", ".png")) or "tfevents" in name:
                continue
            with open(os.path.join(root, name), encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
            hits |= {c for c in forbidden if c in text}
    check("production val/test case ids in run outputs", not hits, f"{len(hits)} found")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Build (or --verify) the GLO-NCA gate smoke subset.")
    ap.add_argument("--data-root", help="full BraTS dataset root (build mode)")
    ap.add_argument("--stage", required=True, help="stage directory (emptied in build mode)")
    ap.add_argument("--verify", metavar="EXPERIMENT_DIR",
                    help="verify a finished smoke run instead of building")
    ap.add_argument("--config", default=os.path.join(_REPO, "configs", "glo_nca_production.yaml"))
    ap.add_argument("--split", default=os.path.join(_REPO, "split", "master_split.json"))
    ap.add_argument("--policy", default=os.path.join(_REPO, "split", "data_quality_policy.json"))
    ap.add_argument("--container-data", default="/data",
                    help="where the smoke container sees <stage>/data")
    ap.add_argument("--container-stage", default="/smoke",
                    help="where the smoke container sees <stage>")
    args = ap.parse_args()
    if args.verify:
        return verify(args.verify, args.stage, args.split)
    if not args.data_root:
        ap.error("--data-root is required to build the smoke stage")

    master = datasource.load_master_split(args.split)
    excluded = set()
    if os.path.isfile(args.policy):
        with open(args.policy, encoding="utf-8") as fh:
            excluded = {e["case_id"] for e in json.load(fh).get("tolerated_seg_labels", [])}
    tr, va, te = select_cases(master, excluded)
    chosen = tr + va + te

    # Hard guarantee: nothing outside master TRAIN is staged.
    forbidden = set(master["validation"]) | set(master["test"])
    leaked = sorted(set(chosen) & forbidden)
    if leaked:
        raise SystemExit(f"FAIL: production val/test case selected: {leaked}")

    paths = datasource.case_path_map(args.data_root)
    missing = [c for c in chosen if c not in paths]
    if missing:
        raise SystemExit(f"FAIL: selected cases not found under {args.data_root}: {missing}")

    # Empty the stage in place (it may be a bind-mount point, which cannot be removed).
    os.makedirs(args.stage, exist_ok=True)
    for entry in os.listdir(args.stage):
        p = os.path.join(args.stage, entry)
        shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
    data_dir = os.path.join(args.stage, "data")
    os.makedirs(data_dir)
    for c in chosen:
        shutil.copytree(os.path.join(args.data_root, paths[c]), os.path.join(data_dir, c))

    split = {"split_version": "gate-smoke",
             "source_split_sha256": master["split_sha256"],
             "train": tr, "validation": va, "test": te,
             "train_count": len(tr), "val_count": len(va), "test_count": len(te),
             "split_sha256": datasource.split_fingerprint(tr, va, te)}
    with open(os.path.join(args.stage, "smoke_split.json"), "w", encoding="utf-8") as fh:
        json.dump(split, fh, indent=2)

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    smoke = copy.deepcopy(cfg)
    smoke["experiment"]["name"] = SMOKE_NAME
    smoke["training"]["epochs"] = SMOKE_EPOCHS
    smoke["dataset"]["number_of_patients"] = 0
    smoke["dataset"]["root"] = args.container_data
    smoke["data"]["split_file"] = f"{args.container_stage}/smoke_split.json"
    smoke["data"]["quality_policy_file"] = None  # policy cases are excluded above
    with open(os.path.join(args.stage, "smoke_config.yaml"), "w", encoding="utf-8") as fh:
        fh.write("# GLO-NCA gate smoke config (generated by scripts/make_gate_smoke.py).\n"
                 "# Engineering smoke only -- never the production run.\n")
        yaml.safe_dump(smoke, fh, sort_keys=False)

    staged = sorted(os.listdir(data_dir))
    report = {"train": tr, "validation": va, "test": te,
              "staged_cases": len(staged),
              "production_val_cases_staged": len(set(staged) & set(master["validation"])),
              "production_test_cases_staged": len(set(staged) & set(master["test"]))}
    with open(os.path.join(args.stage, "smoke_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"smoke subset: train {len(tr)} / val {len(va)} / test {len(te)} "
          f"(all master-train); staged {len(staged)} cases; "
          f"production test cases staged: {report['production_test_cases_staged']}")
    return 0 if report["production_test_cases_staged"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
