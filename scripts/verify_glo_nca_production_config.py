r"""GLO-NCA — PRODUCTION CONFIGURATION IDENTITY GATE (Phase 4).

Purpose
-------
Fail CLOSED unless the supplied configuration IS the intended production
candidate. This exists because the repository deliberately keeps the FROZEN
thesis reference (``extra/configs/historical/v3_multilevel_ckpt.yaml``, 32/96/128) alongside the
current production candidate (``configs/glo_nca_production.yaml``, 48/64), and a
300-epoch cloud run started with the wrong one would burn days of GPU time on
the wrong architecture before anyone noticed.

It verifies the ARCHITECTURE THAT IS ACTUALLY BUILT, not just the YAML text:
the model is constructed and its parameter count and level geometry are read
back from the live object.

Usage
-----
    python scripts/verify_glo_nca_production_config.py <config.yaml>

Exit codes
----------
    0  config IS the production candidate
    2  config is valid YAML but is NOT the production candidate (fail closed)
    3  no config supplied / file missing (fail closed -- never defaults)

Scientific note
---------------
Passing this gate means the config matches the intended 48/64 CANDIDATE. It does
NOT mean that geometry is scientifically approved: 48^3/64^3 remains a Category C
decision relative to the frozen 32/96/128 reference and needs author/supervisor
sign-off. This script checks identity, not scientific merit.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# The intended production candidate. Changing anything here is a deliberate
# scientific act, not a maintenance edit.
EXPECTED = {
    "level1_resolution": 48,
    "level1_channels": 24,
    "level1_kernel_size": 5,
    "level1_nca_steps": 15,
    "level2_resolution": 64,
    "level2_channels": 24,
    "level2_kernel_size": 5,
    "level2_nca_steps": 15,
    "level3_enabled": False,
    "total_nca_steps": 30,
    "spatial_kernel_size": 7,   # the thesis contribution's receptive field (restored 5 -> 7)
    "inference_parameters": 30209,
    "auxiliary_parameters": 75,
    "training_parameters": 30284,
    "working_volume": 128,
    "patchify_enabled": False,
    "roi_fraction": 1.0,
    "use_attention": True,      # SE channel global context
    "use_spatial": True,        # spatial global context
    "fusion_in_out": (48, 24),  # learned fusion
    "precision": "bf16",
    "seed": 42,
    "split_file": "split/master_split.json",
    "split_counts": (898, 200, 198),
    "split_sha256": "d30d71956ee9267017010e5ad71fc033158da818f4af53569e8a65289209559d",
}

PROD_CONFIG = "configs/glo_nca_production.yaml"


def _fail(msg: str, code: int = 2) -> "int":
    print(f"  FAIL  {msg}")
    return code


def main(argv) -> int:
    print("=" * 74)
    print("GLO-NCA PRODUCTION CONFIGURATION IDENTITY GATE")
    print("=" * 74)

    # ---- §8/§22 fail-closed: no config => FAIL. Never fall back to a default.
    if len(argv) < 2 or not str(argv[1]).strip():
        print("  FAIL  no configuration supplied.")
        print()
        print("  A production configuration is REQUIRED. This gate deliberately")
        print("  has NO default: silently falling back to the frozen reference")
        print(f"  would train the WRONG architecture. Use:")
        print(f"      python {os.path.basename(__file__)} {PROD_CONFIG}")
        return 3

    path = argv[1]
    if not os.path.isabs(path):
        cand = os.path.join(HERE, path)
        path = cand if os.path.exists(cand) else path
    if not os.path.isfile(path):
        print(f"  FAIL  configuration not found: {argv[1]}")
        return 3

    import torch
    from src.experiment.config import load_config
    # Build through the SAME entry point the runner uses; a gate that picks
    # its own builder can certify a model that never trains.
    from src.experiment.runner import _build_production_model

    cfg = load_config(path)
    raw = cfg.raw
    fails = []

    def chk(name, actual, expected):
        ok = actual == expected
        print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {actual!r:>28}"
              f"{'' if ok else '   expected ' + repr(expected)}")
        if not ok:
            fails.append(name)

    print(f"\n  config: {os.path.relpath(path, HERE)}\n")

    m_ = raw.get("model", {}) or {}
    if str(m_.get("version", "")).lower() != "v3":
        return _fail("model.version is not 'v3' -- not a GLO-NCA V3 config.")
    if (m_.get("global_context") is None):
        return _fail("model.global_context absent -- this is NOT the global-context "
                     "architecture (likely the frozen reference or a V2 config).")

    l1, l2, l3 = (m_.get("level1") or {}), (m_.get("level2") or {}), (m_.get("level3") or {})
    chk("level1.resolution", l1.get("resolution"), EXPECTED["level1_resolution"])
    chk("level1.channels", l1.get("channels"), EXPECTED["level1_channels"])
    chk("level1.nca_steps", l1.get("nca_steps"), EXPECTED["level1_nca_steps"])
    chk("level1.kernel_size", l1.get("kernel_size"),
        EXPECTED["level1_kernel_size"])
    chk("level2.resolution", l2.get("resolution"), EXPECTED["level2_resolution"])
    chk("level2.channels", l2.get("channels"), EXPECTED["level2_channels"])
    chk("level2.nca_steps", l2.get("nca_steps"), EXPECTED["level2_nca_steps"])
    chk("level2.kernel_size", l2.get("kernel_size"),
        EXPECTED["level2_kernel_size"])
    chk("level3.enabled", bool(l3.get("enabled", False)), EXPECTED["level3_enabled"])
    chk("use_attention (SE)", bool(m_.get("use_attention")), EXPECTED["use_attention"])
    chk("use_spatial (spatial GC)", bool(m_.get("use_spatial")), EXPECTED["use_spatial"])
    # The spatial global-context receptive field. Must be stated by the config:
    # a missing key would silently fall back to the old hardcoded 7 and build a
    # different architecture than the one declared here.
    chk("model.spatial_kernel_size", m_.get("spatial_kernel_size"),
        EXPECTED["spatial_kernel_size"])
    chk("global_context.roi_fraction",
        float((m_.get("global_context") or {}).get("roi_fraction", -1)),
        EXPECTED["roi_fraction"])

    data = raw.get("data", {}) or {}
    tp = data.get("training_patch") or {}
    chk("data.training_patch.enabled", bool(tp.get("enabled", False)),
        EXPECTED["patchify_enabled"])
    # The working volume is training.patch_size: the dataset resamples to it
    # and the model receives it whole. data.training_patch.working_volume
    # applies only when patchify is enabled, which production forbids.
    chk("working volume",
        int((raw.get("training") or {}).get("patch_size", 0)),
        EXPECTED["working_volume"])
    chk("training_patch.working_volume consistent",
        int(tp.get("working_volume", 0)), EXPECTED["working_volume"])
    chk("data.split_file", data.get("split_file"), EXPECTED["split_file"])
    chk("performance.precision",
        str((raw.get("performance") or {}).get("precision", "")).lower(),
        EXPECTED["precision"])
    chk("experiment.seed", int((raw.get("experiment") or {}).get("seed", -1)),
        EXPECTED["seed"])

    # ---- §9/§32: verify the ARCHITECTURE THAT IS BUILT, not just the YAML ----
    model = _build_production_model(cfg, torch.device("cpu"))
    n_params = sum(p.numel() for p in model.parameters())
    n_aux = (sum(p.numel() for p in model.aux_heads.parameters())
             if getattr(model, "aux_heads", None) else 0)
    chk("BUILT inference parameters", n_params - n_aux,
        EXPECTED["inference_parameters"])
    chk("BUILT auxiliary parameters", n_aux, EXPECTED["auxiliary_parameters"])
    chk("BUILT training parameters", n_params, EXPECTED["training_parameters"])
    chk("BUILT level count (no level3)", len(model.levels), 2)
    chk("BUILT total NCA steps", sum(l.nca_steps for l in model.levels),
        EXPECTED["total_nca_steps"])
    fusion = [(x.in_channels, x.out_channels) for x in model.modules()
              if isinstance(x, torch.nn.Conv3d) and x.in_channels == 48]
    chk("BUILT learned fusion", fusion[0] if fusion else None,
        EXPECTED["fusion_in_out"])

    # ---- §29 split identity -------------------------------------------------
    split_path = os.path.join(HERE, EXPECTED["split_file"])
    if os.path.isfile(split_path):
        import json
        sp = json.load(open(split_path, encoding="utf-8"))
        chk("split counts",
            (len(sp["train"]), len(sp["validation"]), len(sp["test"])),
            EXPECTED["split_counts"])
        chk("split sha256", sp.get("split_sha256"), EXPECTED["split_sha256"])
    else:
        fails.append("split file missing")
        print(f"  FAIL  split file missing: {EXPECTED['split_file']}")

    print()
    print("=" * 74)
    if fails:
        print(f"  RESULT: FAIL CLOSED -- {len(fails)} mismatch(es): {', '.join(fails[:6])}"
              + (" ..." if len(fails) > 6 else ""))
        print()
        print("  The supplied configuration is NOT the production candidate.")
        print("  NO substitute configuration will be selected automatically.")
        print(f"  Production candidate: {PROD_CONFIG}")
        print("=" * 74)
        return 2

    print("  RESULT: PASS -- configuration IS the GLO-NCA production candidate.")
    print(f"          GLO-NCA | {EXPECTED['working_volume']}^3 working volume | "
          f"L1 48^3 k{EXPECTED['level1_kernel_size']} | "
          f"L2 64^3 k{EXPECTED['level2_kernel_size']} | "
          f"{EXPECTED['level1_nca_steps']}+{EXPECTED['level2_nca_steps']} steps "
          f"| spatial GC k{EXPECTED['spatial_kernel_size']}")
    print(f"          {n_params - n_aux:,} inference | {n_aux} auxiliary | "
          f"{n_params:,} training parameters")
    print("          global context ON | patchify OFF | deep supervision ON | bf16")
    print()
    print("  NOTE: identity verified, NOT scientific approval. No Dice/IoU/HD95")
    print("        evidence exists for this architecture until the thesis run.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
