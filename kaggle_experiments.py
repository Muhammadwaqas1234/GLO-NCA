r"""
================================================================================
GLO-NCA — full thesis experiment suite (Kaggle, single cell)
================================================================================
Runs everything needed for the thesis results section:

  EXP 1  Ablation      : baseline NCA (use_attention=False)
  EXP 2  GLO-NCA       : with global-context SE block (use_attention=True)
  EXP 3  GLO-NCA++     : improved config (bigger patch + more steps + more channels)
  EXP 4  Robustness    : EXP 2 config over 3 seeds -> mean +/- std

Same 140/30/30 patient split for EXP 1-3 (fair comparison). Prints a final
comparison table and saves all curves/metrics to /kaggle/working/glo_exp.

USAGE (one Kaggle cell, GPU on, BraTS dataset attached):
    !git clone -q https://github.com/Muhammadwaqas1234/GLO-NCA.git
    %run GLO-NCA/kaggle_experiments.py

Tune QUICK / EPOCHS / N_PATIENTS below to trade speed vs completeness.
================================================================================
"""
import os, sys, time, json, math, random, subprocess

# ------------------------------------------------------------------ knobs
EPOCHS       = 100      # per run
N_PATIENTS   = None     # None = all; or an int to cap (faster)
RUN_ABLATION = True
RUN_GLO      = True
RUN_IMPROVED = True
RUN_SEEDS    = True     # 3-seed robustness on the GLO config
SEEDS        = [42, 7, 123]
BASE_SEED    = 42

REPO_URL = "https://github.com/Muhammadwaqas1234/GLO-NCA.git"
REPO_DIR = "/kaggle/working/GLO-NCA"
OUT_DIR  = "/kaggle/working/glo_exp"
REGIONS  = ["WT", "TC", "ET"]
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

# ------------------------------------------------------------------ setup
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchio"], check=False)
if not os.path.isdir(os.path.join(REPO_DIR, "src")):
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import DiceCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_GLO_NCA import Agent_GLO_NCA
from src.agents.Agent import iou_score, hd95_score


# ------------------------------------------------------------------ data root
def find_data_root(base="/kaggle/input"):
    for root, dirs, files in os.walk(base):
        patient_like = 0
        for d in dirs:
            p = os.path.join(root, d)
            try:
                if any(f.endswith((".nii", ".nii.gz")) for f in os.listdir(p)):
                    patient_like += 1
            except Exception:
                pass
        if patient_like >= 2:
            return root, patient_like
    return None, 0


DATA_ROOT, n_found = find_data_root("/kaggle/input")
assert DATA_ROOT, "BraTS dataset not found under /kaggle/input — attach it (Add Input)."
print(f"DATA_ROOT = {DATA_ROOT} ({n_found} patient folders)")


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_split(seed):
    pats = sorted(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)))
    random.Random(seed).shuffle(pats)
    if N_PATIENTS:
        pats = pats[:N_PATIENTS]
    n = len(pats)
    n_tr, n_val = int(n * 0.70), int(n * 0.15)
    return pats[:n_tr], pats[n_tr:n_tr+n_val], pats[n_tr+n_val:]


def evaluate(agent, dataset, state):
    agent.exp.set_model_state(state)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    acc = {r: {"dice": [], "iou": [], "hd95": []} for r in REGIONS}
    with torch.no_grad():
        for data in loader:
            data = agent.prepare_data(data, eval=True)
            outputs, targets = agent.get_outputs(data, full_img=True)
            prob = torch.sigmoid(outputs).detach().cpu().numpy()
            gt = targets.detach().cpu().numpy()
            for i, r in enumerate(REGIONS):
                p, t = prob[..., i], gt[..., i]
                inter = np.logical_and(p >= 0.5, t >= 0.5).sum()
                d = (2 * inter) / ((p >= 0.5).sum() + (t >= 0.5).sum() + 1e-6)
                acc[r]["dice"].append(float(d)); acc[r]["iou"].append(iou_score(p, t))
                acc[r]["hd95"].append(hd95_score(p, t))
    agent.exp.set_model_state("train")
    out = {}
    for r in REGIONS:
        hd = [v for v in acc[r]["hd95"] if not math.isnan(v)]
        out[r] = {"dice": float(np.mean(acc[r]["dice"])), "iou": float(np.mean(acc[r]["iou"])),
                  "hd95": float(np.mean(hd)) if hd else float("nan")}
    return out


def run_experiment(name, use_attention, channel_n=16, hidden=64,
                   input_size=((28, 28, 20), (56, 56, 40)), steps=(10, 10), seed=BASE_SEED):
    """Train one config, return test metrics + history."""
    set_seed(seed)
    tag = f"{name}_s{seed}"
    model_path = os.path.join(OUT_DIR, tag)
    train_ids, val_ids, test_ids = make_split(seed)
    print(f"\n{'='*60}\n{name}  (attention={use_attention}, ch={channel_n}, "
          f"hid={hidden}, steps={list(steps)}, patch={input_size[-1]}, seed={seed})")
    print(f"split train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)}")

    config = [{
        "img_path": DATA_ROOT, "label_path": DATA_ROOT, "model_path": model_path,
        "device": "cuda:0", "unlock_CPU": True,
        "optimizer": "adamw", "lr": 16e-4, "lr_gamma": 0.9999,
        "betas": (0.9, 0.99), "weight_decay": 1e-4,
        "save_interval": EPOCHS*100, "evaluate_interval": EPOCHS*100, "n_epoch": EPOCHS,
        "batch_size": 1, "batch_duplication": 1,
        "channel_n": channel_n, "inference_steps": list(steps), "cell_fire_rate": 0.5,
        "input_channels": 4, "output_channels": 3, "hidden_size": hidden,
        "train_model": 1, "use_attention": use_attention,
        "input_size": [list(input_size[0]), list(input_size[1])], "scale_factor": 2,
        "data_split": [0.7, 0.15, 0.15], "keep_original_scale": True, "rescale": True,
        "patchify": True, "priotize_masks": 0.5,
    }]
    dataset = Dataset_NiiGz_3D_BraTS(); dataset.MODALITIES = MODALITIES
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ca = [BasicNCA3D(channel_n, 0.5, dev, hidden, kernel_size=7, input_channels=4, use_attention=use_attention),
          BasicNCA3D(channel_n, 0.5, dev, hidden, kernel_size=3, input_channels=4, use_attention=use_attention)]
    agent = Agent_GLO_NCA(ca)
    exp = Experiment(config, dataset, ca, agent); dataset.set_experiment(exp)
    def entry(p): return (p, p, 0)
    for sp, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        exp.data_split.images[sp] = {p: {0: entry(p)} for p in ids}
        exp.data_split.labels[sp] = {p: {0: entry(p)} for p in ids}
    exp.set_model_state("train")
    loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=1)
    loss_f = DiceCELoss()
    n_params = sum(p.numel() for m in ca for p in m.parameters())

    history = {"epoch": [], "val_mean": [], "val_WT": [], "val_TC": [], "val_ET": [], "loss": []}
    best_mean, best = -1.0, os.path.join(model_path, "best.pth")
    os.makedirs(model_path, exist_ok=True)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(EPOCHS):
        losses = []
        for data in loader:
            r = agent.batch_step(data, loss_f)
            if r:
                losses.append(sum(r.values()))
        avg = float(np.mean(losses)) if losses else 0.0
        val = evaluate(agent, dataset, "val")
        vmean = float(np.mean([val[r]["dice"] for r in REGIONS]))
        history["epoch"].append(ep+1); history["loss"].append(avg); history["val_mean"].append(vmean)
        for r in REGIONS:
            history[f"val_{r}"].append(val[r]["dice"])
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  ep {ep+1}/{EPOCHS} loss {avg:.3f} | val mean {vmean:.3f} "
                  f"(WT {val['WT']['dice']:.3f} TC {val['TC']['dice']:.3f} ET {val['ET']['dice']:.3f})",
                  flush=True)
        if vmean > best_mean:
            best_mean = vmean
            torch.save({"m0": ca[0].state_dict(), "m1": ca[1].state_dict(), "ep": ep+1}, best)
    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated()/1e9 if dev.type == "cuda" else 0

    ck = torch.load(best, map_location=dev)
    ca[0].load_state_dict(ck["m0"]); ca[1].load_state_dict(ck["m1"])
    test = evaluate(agent, dataset, "test")
    print(f"  -> TEST  WT {test['WT']['dice']:.3f}  TC {test['TC']['dice']:.3f}  "
          f"ET {test['ET']['dice']:.3f} | best ep {ck['ep']} | {train_time:.0f}s | "
          f"{peak:.2f}GB | {n_params} params", flush=True)
    return {"name": name, "use_attention": use_attention, "params": n_params,
            "test": test, "history": history, "train_time": train_time,
            "peak_vram": peak, "best_epoch": ck["ep"], "seed": seed}


# ------------------------------------------------------------------ run all
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}

    if RUN_ABLATION:
        results["baseline"] = run_experiment("Baseline_NCA", use_attention=False)
    if RUN_GLO:
        results["glo"] = run_experiment("GLO_NCA", use_attention=True)
    if RUN_IMPROVED:
        results["improved"] = run_experiment(
            "GLO_NCA_pp", use_attention=True, channel_n=24, hidden=96,
            input_size=((40, 40, 28), (80, 80, 56)), steps=(15, 15))

    seed_runs = []
    if RUN_SEEDS:
        for sd in SEEDS:
            seed_runs.append(run_experiment("GLO_seed", use_attention=True, seed=sd))
        results["seeds"] = seed_runs

    json.dump(results, open(os.path.join(OUT_DIR, "all_results.json"), "w"), indent=2, default=str)
    make_report(results)


def make_report(results):
    # ---- comparison table ----
    print("\n" + "=" * 72)
    print("COMPARISON TABLE  (Dice on held-out TEST set)")
    print("=" * 72)
    print(f"{'Method':<26}{'WT':<8}{'TC':<8}{'ET':<8}{'mean':<8}{'params':<9}")
    print("-" * 72)
    def row(label, res):
        t = res["test"]; m = np.mean([t[r]["dice"] for r in REGIONS])
        print(f"{label:<26}{t['WT']['dice']:<8.3f}{t['TC']['dice']:<8.3f}"
              f"{t['ET']['dice']:<8.3f}{m:<8.3f}{res['params']:<9}")
    if "baseline" in results: row("Baseline NCA (no attn)", results["baseline"])
    if "glo" in results:      row("GLO-NCA (ours)", results["glo"])
    if "improved" in results: row("GLO-NCA++ (big patch)", results["improved"])
    if "seeds" in results:
        import numpy as _np
        arr = {r: _np.array([s["test"][r]["dice"] for s in results["seeds"]]) for r in REGIONS}
        print("-" * 72)
        print(f"{'GLO-NCA 3-seed mean':<26}" + "".join(f"{arr[r].mean():<8.3f}" for r in REGIONS) +
              f"{_np.mean([arr[r].mean() for r in REGIONS]):<8.3f}")
        print(f"{'   +/- std':<26}" + "".join(f"{arr[r].std():<8.3f}" for r in REGIONS))
    print("=" * 72)

    # ---- ablation bar chart ----
    if "baseline" in results and "glo" in results:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(REGIONS)); w = 0.35
        b = [results["baseline"]["test"][r]["dice"] for r in REGIONS]
        g = [results["glo"]["test"][r]["dice"] for r in REGIONS]
        ax.bar(x - w/2, b, w, label="Baseline NCA", color="#9e9e9e")
        ax.bar(x + w/2, g, w, label="GLO-NCA (ours)", color="#1f77b4")
        for i in range(len(REGIONS)):
            ax.text(i - w/2, b[i]+.01, f"{b[i]:.2f}", ha="center", fontsize=9)
            ax.text(i + w/2, g[i]+.01, f"{g[i]:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(REGIONS); ax.set_ylim(0, 1)
        ax.set_title("Ablation: effect of the global-context (SE) block")
        ax.set_ylabel("Test Dice"); ax.legend(); ax.grid(axis="y", alpha=.3)
        plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "ablation.png"), dpi=130); plt.show()

    # ---- training curves (val mean) ----
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, lab, c in [("baseline", "Baseline NCA", "#9e9e9e"),
                        ("glo", "GLO-NCA", "#1f77b4"),
                        ("improved", "GLO-NCA++", "#2ca02c")]:
        if key in results:
            h = results[key]["history"]
            ax.plot(h["epoch"], h["val_mean"], label=lab, color=c)
    ax.set_title("Validation mean Dice over epochs"); ax.set_xlabel("epoch")
    ax.set_ylabel("val mean Dice"); ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "curves_compare.png"), dpi=130); plt.show()
    print("\nAll experiment outputs saved to", OUT_DIR)


if __name__ == "__main__":
    main()
