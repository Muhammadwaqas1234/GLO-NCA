r"""
================================================================================
GLO-NCA V3 -- developer environment check (Python 3.12.9 / torch 2.5.1+cu121)
================================================================================
Reports which interpreter is running, whether it is the SUPPORTED one, and
whether the GPU stack and the project's real dependencies are present.

WHY THIS EXISTS
---------------
Phase 5 found that `python` on this machine resolves to Python 3.14, which has
no torch installed. Under that interpreter the repository's guard tests appear
to FAIL for reasons that have nothing to do with the code: `train.py` dies at
`import torch` (rc=1) BEFORE its argument guards can return rc=2. The failure is
environmental, but it reads like a correctness regression.

This script makes the interpreter explicit and the diagnosis immediate.

IMPORTANT: this module deliberately does NOT import torch at module level. It
must be able to RUN on the wrong interpreter in order to report that the
interpreter is wrong. `src/experiment/environment.py` (the per-run record) does
import torch at module level and therefore cannot serve this purpose -- the two
are complementary, not duplicates.

USAGE
-----
    <python-3.12.9> scripts/check_environment.py

Exit codes: 0 = supported environment, 1 = problem found (see the report).
This is a developer tool; it is NOT imported by training and adds no runtime
overhead to a production run.
================================================================================
"""
from __future__ import annotations

import importlib
import os
import platform
import sys

# The supported production/development environment. These are the values the
# GLO-NCA V3 work has actually been verified against -- not aspirational.
REQUIRED_PYTHON = "3.12.9"
REQUIRED_TORCH = "2.5.1+cu121"
REQUIRED_CUDA = "12.1"

# Third-party packages actually imported by src/, scripts/ and train.py.
# Kept deliberately in sync with the real imports; nothing speculative.
REQUIRED_PACKAGES = [
    "torch", "numpy", "yaml", "nibabel", "torchio", "scipy",
    "cv2", "psutil", "matplotlib", "seaborn", "pandas", "tensorboard",
]

_OK, _BAD, _WARN = "  OK  ", " FAIL ", " WARN "


def _line(status: str, label: str, value: str) -> None:
    print(f"[{status}] {label:<22} {value}")


def check_interpreter() -> bool:
    print("-" * 78)
    print("INTERPRETER")
    print("-" * 78)
    ver = platform.python_version()
    _line(_OK, "executable", sys.executable)
    ok = ver == REQUIRED_PYTHON
    _line(_OK if ok else _BAD, "python version", f"{ver} (required {REQUIRED_PYTHON})")
    if not ok:
        print()
        print("  This is NOT the supported interpreter. Run the project with:")
        print(r"    C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe")
        print("  or `py -3.12`. Do NOT install torch into another interpreter to")
        print("  work around this -- the supported stack is pinned deliberately.")
    return ok


def check_torch() -> bool:
    print("-" * 78)
    print("PYTORCH / CUDA / GPU")
    print("-" * 78)
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # the exact failure seen under Python 3.14
        _line(_BAD, "torch import", f"{type(exc).__name__}: {exc}")
        print("\n  torch is not installed for THIS interpreter. That is the whole")
        print("  problem: train.py imports torch, so it cannot even reach its own")
        print("  argument guards. Switch interpreter rather than installing torch.")
        return False

    ok = True
    tv = torch.__version__
    good_tv = tv == REQUIRED_TORCH
    _line(_OK if good_tv else _WARN, "torch version", f"{tv} (expected {REQUIRED_TORCH})")
    ok &= good_tv

    cv = getattr(torch.version, "cuda", None)
    good_cv = cv == REQUIRED_CUDA
    _line(_OK if good_cv else _WARN, "torch cuda build", f"{cv} (expected {REQUIRED_CUDA})")
    ok &= good_cv

    avail = torch.cuda.is_available()
    _line(_OK if avail else _WARN, "cuda available", str(avail))
    if not avail:
        print("  No CUDA device visible. CPU is fine for correctness checks, but")
        print("  profiling and memory work require the GPU.")
        return ok

    _line(_OK, "gpu", torch.cuda.get_device_name(0))
    _line(_OK, "capability", str(torch.cuda.get_device_capability(0)))
    props = torch.cuda.get_device_properties(0)
    _line(_OK, "gpu memory", f"{props.total_memory / 1024 ** 3:.1f} GB")
    print("  NOTE: the local GPU is a development/profiling device. Phase 3")
    print("  measured 128^3 forward+backward as SPILL here -- production")
    print("  training is NOT local.")
    return ok


def check_packages() -> bool:
    print("-" * 78)
    print("DEPENDENCIES")
    print("-" * 78)
    missing = []
    for name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(name)
            _line(_OK, name, str(getattr(mod, "__version__", "n/a")))
        except Exception as exc:
            missing.append(name)
            _line(_BAD, name, f"{type(exc).__name__}: {exc}")
    if missing:
        print(f"\n  Missing for this interpreter: {', '.join(missing)}")
        print("  Report these rather than auto-installing: an unplanned upgrade")
        print("  (especially of torch) would change the verified environment.")
    return not missing


def check_project() -> bool:
    """Import the project and re-verify the production invariants."""
    print("-" * 78)
    print("GLO-NCA V3 PROJECT")
    print("-" * 78)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import json
        import torch
        from src.experiment.config import load_config
        from src.experiment.datasource import split_fingerprint
        from src.models.Model_GLO_NCA_V3 import build_v3_from_config
    except Exception as exc:
        _line(_BAD, "project import", f"{type(exc).__name__}: {exc}")
        return False

    ok = True
    cfg_path = os.path.join(root, "extra", "configs", "historical", "v3_multilevel_ckpt.yaml")
    cfg = load_config(cfg_path)
    model = build_v3_from_config(cfg, 4, 3, torch.device("cpu"))

    n = sum(p.numel() for p in model.parameters())
    good = n == 40656
    _line(_OK if good else _BAD, "parameters", f"{n} (expected 40,656)")
    ok &= good

    levels = [(l.resolution, l.channels, l.nca_steps) for l in model.levels]
    good = levels == [(32, 24, 20), (96, 24, 20), (128, 16, 10)]
    _line(_OK if good else _BAD, "levels", str(levels))
    ok &= good

    steps = sum(l.nca_steps for l in model.levels)
    good = steps == 50
    _line(_OK if good else _BAD, "total NCA steps", f"{steps} (expected 50)")
    ok &= good

    good = bool(model.gradient_checkpointing)
    _line(_OK if good else _BAD, "grad checkpointing", str(good))
    ok &= good

    split_path = os.path.join(root, "split", "master_split.json")
    with open(split_path, encoding="utf-8") as fh:
        d = json.load(fh)
    sizes = (len(d["train"]), len(d["validation"]), len(d["test"]))
    good = sizes == (898, 200, 198)
    _line(_OK if good else _BAD, "canonical split", f"{sizes[0]}/{sizes[1]}/{sizes[2]}")
    ok &= good

    expected_sha = ("d30d71956ee9267017010e5ad71fc033158da818f4af53569e8"
                    "a65289209559d")
    fp = split_fingerprint(d["train"], d["validation"], d["test"])
    good = fp == expected_sha
    _line(_OK if good else _BAD, "split SHA256", f"{fp[:16]}... match={good}")
    ok &= good

    cache_on = bool(((cfg.section("data") or {}).get("cache") or {}).get("enabled", False))
    _line(_OK if not cache_on else _WARN, "production cache",
          "OFF (expected)" if not cache_on else "ON -- expected OFF")
    ok &= not cache_on
    return ok


def main() -> int:
    print("=" * 78)
    print("GLO-NCA V3 -- ENVIRONMENT CHECK")
    print("=" * 78)
    results = {
        "interpreter": check_interpreter(),
        "torch/cuda/gpu": check_torch(),
        "dependencies": check_packages(),
        "project invariants": check_project(),
    }
    print("=" * 78)
    failed = [k for k, v in results.items() if not v]
    for key, val in results.items():
        print(f"  {'PASS' if val else 'FAIL'}  {key}")
    print("=" * 78)
    if failed:
        print(f"ENVIRONMENT CHECK: FAIL ({', '.join(failed)})")
        return 1
    print("ENVIRONMENT CHECK: PASS")
    print(f"  Python {REQUIRED_PYTHON} | torch {REQUIRED_TORCH} | CUDA {REQUIRED_CUDA}")
    print("  Production training is NOT local. GCP is NOT used by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
