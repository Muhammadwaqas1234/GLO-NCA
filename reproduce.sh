#!/usr/bin/env bash
# Reproduce the reported BraTS 2021 internal test table (historical model) from its weights.
# Usage: ./reproduce.sh /path/to/BraTS2021 [run-folder]
# The run folder needs best.pth, config.yaml and split.json. Missing files are downloaded from
# GLO_RELEASE_URL when it is set; no public release exists until the weights are published.
set -euo pipefail

DATA=${1:?usage: ./reproduce.sh /path/to/BraTS2021 [run-folder]}
RUN=${2:-release/glo-nca-brats2021-res96}
URL=${GLO_RELEASE_URL:-}

mkdir -p "$RUN"
for f in best.pth config.yaml split.json; do
  if [ ! -f "$RUN/$f" ]; then
    if [ -z "$URL" ]; then
      echo "FAILED: $RUN/$f is missing. Copy the run's best.pth, config.yaml and split.json into $RUN,"
      echo "        or set GLO_RELEASE_URL to a release that contains them."
      exit 2
    fi
    curl -fsSL -o "$RUN/$f" "$URL/$f" || { echo "FAILED: could not download $URL/$f"; exit 2; }
  fi
done
echo "best.pth sha256: $(sha256sum "$RUN/best.pth" | cut -d' ' -f1)"

# The reported setting, stated explicitly: ensemble x4 + flips, thresholds and clean-up chosen on
# validation, full resolution, BraTS empty-region scoring, independent (non-nested) region masks as
# originally scored, surface HD95 in mm (medpy convention), test split.
python evaluate.py --run "$RUN" --data-root "$DATA" --splits test --ensemble-n 4 \
  --thresholds WT=0.65,TC=0.65,ET=0.6 --min-component WT=400,TC=25,ET=400 \
  --full-resolution --brats-empty --no-nested-regions --hd95-surface --hd95-convention medpy --output "$RUN/reproduce.json"

python - "$RUN/reproduce.json" <<'PY'
import json, sys
res = json.load(open(sys.argv[1]))["test"]
reported = {"WT": (0.921, 4.93), "TC": (0.895, 5.23), "ET": (0.817, 4.49)}
# The historical ensemble was not seeded: the run's own final test of the same setting gave ET 0.823
# against 0.817 here (README), so the tolerance covers that spread.
dice_tol, hd_tol = 0.010, 0.50
ok = True
print(f"{'':4}{'Dice':>8}{'reported':>10}{'HD95':>8}{'reported':>10}{'cases':>7}")
for r, (dice, hd) in reported.items():
    good = abs(res[r]["dice"] - dice) <= dice_tol and abs(res[r]["hd95"] - hd) <= hd_tol
    ok &= good
    print(f"{r:4}{res[r]['dice']:8.3f}{dice:10.3f}{res[r]['hd95']:8.2f}{hd:10.2f}"
          f"{res[r].get('cases', ''):>7}{'' if good else '  <- outside tolerance'}")
ok &= all(res[r].get("cases") == 189 for r in reported)
mean = sum(res[r]["dice"] for r in reported) / 3
print(f"mean Dice {mean:.3f} (reported 0.878); tolerance Dice +-{dice_tol}, HD95 +-{hd_tol} mm, 189 cases")
print("REPRODUCED" if ok else "DIFFERENT")
sys.exit(0 if ok else 1)
PY
