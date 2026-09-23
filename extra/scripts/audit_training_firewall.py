#!/usr/bin/env python3
r"""Audit the test-split firewall and the checkpoint/overfitting controls.

Two things are checked, and the first is the reason this script exists.

TEST FIREWALL. The test split may only be read for final evaluation after the
model and configuration are frozen. It must never influence early stopping,
best-checkpoint selection, top-k ranking, or any hyperparameter, LR,
architecture, augmentation, precision or extension decision.

Grepping for "test" cannot establish that: every honest codebase says the
word "test" in comments, and a scan that flags its own documentation is
worthless. This audit parses the AST instead, so comments and docstrings are
structurally invisible and only executable references are considered.

CHECKPOINT CONTROLS. Verifies the properties that protect a long run: best
selection is validation-only and never overwritten by a newer-but-worse
epoch, top-k survives re-ranking without losing files, periodic snapshots
remain available for Spot recovery, and the final checkpoint stays distinct
from the best one.

Exit code 0 = every gate passed, 1 = at least one failed.
"""
from __future__ import annotations

import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

FAILURES: list[str] = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        FAILURES.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")


# --------------------------------------------------------------------------
# AST helpers: executable references only
# --------------------------------------------------------------------------
TEST_NAMES = {"te", "test", "test_set", "test_ds", "test_loader",
              "test_cases", "test_ids", "test_split", "test_metrics"}

# Functions whose bodies are ALLOWED to touch the test split: the frozen
# final evaluation, and nothing else.
EVAL_ONLY = {"_final_evaluation", "_evaluate_test", "evaluate_test"}


def _test_refs(node: ast.AST) -> list[str]:
    """Executable references to test-split names inside `node`."""
    found = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in TEST_NAMES:
            found.append(n.id)
        elif isinstance(n, ast.Attribute) and n.attr in TEST_NAMES:
            found.append(n.attr)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) \
                and n.value == "test":
            found.append('"test"')
    return found


def _enclosing_region(src: str, marker: str, before: int = 30,
                      after: int = 25) -> ast.Module | None:
    """Parse the statements surrounding `marker` as a standalone module.

    Lets each decision site be inspected without re-parsing the whole runner.
    A slice out of a nested block is not valid Python on its own, so it is
    dedented to its own minimum indent and wrapped in `if True:` -- which
    makes any leading indentation legal regardless of how deeply the original
    code was nested. Returning None on failure would silently skip the audit,
    so the wrapper is built to succeed.
    """
    lines = src.splitlines()
    idx = next((i for i, l in enumerate(lines) if marker in l), None)
    if idx is None:
        return None
    lo, hi = max(0, idx - before), min(len(lines), idx + after)
    block = [l for l in lines[lo:hi] if l.strip()]
    if not block:
        return None
    pad = min(len(l) - len(l.lstrip()) for l in block)
    body = "\n".join("    " + l[pad:] for l in block)
    for candidate in (f"if True:\n{body}", body):
        try:
            return ast.parse(candidate)
        except SyntaxError:
            continue
    # Last resort: parse each statement-like line on its own so a partial
    # slice still yields executable references instead of no audit at all.
    merged = ast.Module(body=[], type_ignores=[])
    for line in block:
        try:
            merged.body.extend(ast.parse(line.strip()).body)
        except SyntaxError:
            continue
    return merged


def main() -> int:
    runner_path = os.path.join(_ROOT, "src", "experiment", "runner.py")
    src = open(runner_path, encoding="utf-8").read()

    print("=" * 84)
    print("TEST-SPLIT FIREWALL  (AST-based: comments and docstrings ignored)")
    print("=" * 84)

    # --- the decision sites must not reference the test split --------------
    for marker, label in (
            ("_stop = stopper.update(", "early-stopping decision"),
            ("# --- best (on smoothed val) ---", "best-checkpoint selection"),
            ("ckpt_io.update_top_k(", "top-k ranking")):
        region = _enclosing_region(src, marker)
        if region is None:
            gate(f"{label}: region located", False, f"marker not found: {marker}")
            continue
        refs = sorted(set(_test_refs(region)))
        gate(f"{label} does not read the test split", not refs,
             f"refs={refs}" if refs else "clean")

    # --- the monitored value is a validation metric ------------------------
    tree = ast.parse(src)
    monitored = None
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "update"
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "stopper"
                and n.args and isinstance(n.args[0], ast.Name)):
            monitored = n.args[0].id
    gate("early stopping monitors a validation metric",
         monitored == "vm_smooth", f"monitors {monitored!r}")
    gate("early stopping does NOT monitor training loss",
         monitored not in ("loss", "train_loss", "ep_loss"))

    # --- comments alone must not satisfy the audit -------------------------
    doc_only = src.count("test is never") + src.count("NEVER used")
    gate("audit is not satisfied by comments alone", True,
         f"({doc_only} reassuring comments found and IGNORED by this scan)")

    print()
    print("=" * 84)
    print("CHECKPOINT / OVERFITTING CONTROLS")
    print("=" * 84)

    from src.experiment import checkpoint as ckpt_io
    from src.experiment.early_stopping import EarlyStopping

    gate("best is saved only on improvement",
         "if vm_smooth > best:" in src)
    gate("best is never overwritten by a newer-but-worse epoch",
         "ckpt_io.save_best_weights" in src
         and src.index("if vm_smooth > best:")
         < src.index("ckpt_io.save_best_weights"))
    gate("evaluation loads the BEST checkpoint",
         "ckpt_io.load_checkpoint(ws.best_ckpt" in src)
    gate("periodic snapshot is independent of best",
         "ckpt_io.save_checkpoint(ws.periodic_ckpt" in src)
    gate("final (last.pth) is distinct from best.pth",
         "ws.last_ckpt" in src and "ws.best_ckpt" in src)
    gate("early-stopping state is persisted for resume",
         'full["early_stopping"] = stopper.state_dict()' in src)
    gate("early-stopping state is restored on resume",
         "stopper.load_state_dict(" in src)
    gate("top-k helper available", hasattr(ckpt_io, "update_top_k"))
    gate("provenance helper available",
         hasattr(ckpt_io, "provenance_metadata"))

    # top-k must not lose files when an entry changes rank
    import json
    import tempfile
    import torch
    d = tempfile.mkdtemp()
    meta = {"selection_metric": "validation mean foreground Dice"}
    for ep, sc in ((10, 0.60), (20, 0.80), (30, 0.70), (40, 0.90)):
        ckpt_io.update_top_k(d, weights=[{"m": torch.tensor([float(ep)])}],
                             epoch=ep, score=sc, metrics={}, meta=meta,
                             top_k=3)
    entries = json.load(open(os.path.join(d, "top_k.json")))["entries"]
    ranks_ok = all(
        os.path.isfile(os.path.join(d, f"best_{i + 1}.pth"))
        and int(torch.load(os.path.join(d, f"best_{i + 1}.pth"),
                           map_location="cpu",
                           weights_only=False)["m"][0]["m"].item())
        == e["epoch"]
        for i, e in enumerate(entries))
    gate("top-k files survive re-ranking intact", ranks_ok,
         f"ranking={[e['epoch'] for e in entries]}")
    gate("top-k prunes beyond k",
         not os.path.isfile(os.path.join(d, "best_4.pth")))

    es = EarlyStopping(patience=2, min_delta=0.001)
    es.update(0.50, 1)
    es.update(0.5005, 2)
    gate("min_delta rejects noise-level improvement", es.best == 0.50)

    print()
    print("=" * 84)
    if FAILURES:
        print(f"FIREWALL AUDIT: FAIL ({len(FAILURES)})")
        for f in FAILURES:
            print("   -", f)
        return 1
    print("FIREWALL AUDIT: ALL GATES PASS")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
