r"""Redraw the training, test-summary and per-case figures of a finished run from its saved results."""
import argparse
import json
import os
import sys


def main(argv=None) -> int:
    r"""Read results.json and test_per_case.csv of a run and write its figures."""
    ap = argparse.ArgumentParser(description="Redraw the figures of a finished run (no GPU needed).")
    ap.add_argument("run", help="run folder with results.json")
    args = ap.parse_args(argv)

    from glo_nca.reporting import load_per_case, write_figures

    path = os.path.join(args.run, "results.json")
    if not os.path.isfile(path):
        print(f"FAILED: {path} not found (the run has not finished its final test).")
        return 2
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    settings = {"Single pass": res["test_plain"], "Ensemble + TTA": res["test"]}
    if res.get("test_tuned"):
        settings["Tuned post-processing"] = res["test_tuned"]["test"]
    write_figures(args.run, res["history"], res["best_epoch"], settings, load_per_case(args.run),
                 title=os.path.basename(os.path.normpath(args.run)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
