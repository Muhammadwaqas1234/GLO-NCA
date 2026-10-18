#!/usr/bin/env bash
# Reproduce the reported BraTS 2021 test table from the released weights.
# Usage: ./reproduce.sh /path/to/BraTS2021 [run-folder]
# The run folder needs best.pth, config.yaml and split.json; missing files are downloaded from the release.
set -euo pipefail

DATA=${1:?usage: ./reproduce.sh /path/to/BraTS2021 [run-folder]}
RUN=${2:-release/glo-nca-brats2021-res96}
URL=${GLO_RELEASE_URL:-https://github.com/Muhammadwaqas1234/GLO-NCA/releases/download/v1.1.0}

mkdir -p "$RUN"
for f in best.pth config.yaml split.json; do
  [ -f "$RUN/$f" ] || curl -fsSL -o "$RUN/$f" "$URL/$f"
done

# The reported setting: ensemble x4 + flips, thresholds and clean-up chosen on validation,
# full resolution, BraTS empty-region scoring and surface HD95 in mm, on the 189 test patients.
python evaluate.py --run "$RUN" --data-root "$DATA" --splits test \
  --thresholds WT=0.65,TC=0.65,ET=0.6 --min-component WT=400,TC=25,ET=400 \
  --full-resolution --brats-empty --hd95-surface --output "$RUN/reproduce.json"

python - "$RUN/reproduce.json" <<'PY'
import json, sys
res = json.load(open(sys.argv[1]))["test"]
reported = {"WT": (0.921, 4.93), "TC": (0.895, 5.23), "ET": (0.817, 4.49)}
tol = 0.005   # the ensemble passes are stochastic
ok = True
print(f"{'':4}{'Dice':>8}{'reported':>10}{'HD95':>8}{'reported':>10}")
for r, (dice, hd) in reported.items():
    print(f"{r:4}{res[r]['dice']:8.3f}{dice:10.3f}{res[r]['hd95']:8.2f}{hd:10.2f}")
    ok &= abs(res[r]["dice"] - dice) <= tol
mean = sum(res[r]["dice"] for r in reported) / 3
print(f"mean Dice {mean:.3f} (reported 0.878)")
print("REPRODUCED" if ok else f"DIFFERENT: Dice outside +-{tol} of the reported values")
sys.exit(0 if ok else 1)
PY
