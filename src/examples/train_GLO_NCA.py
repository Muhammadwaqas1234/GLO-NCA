r"""Train GLO-NCA (Global Context-Aware NCA) on the multi-modal BraTS dataset.

This is the main training entry point for the thesis. It wires together:
  * Dataset_NiiGz_3D_BraTS  - 4 modalities (T1/T1ce/T2/FLAIR) + ET/TC/WT regions
  * BasicNCA3D(use_attention=True) - NCA with the lightweight SE global-context
        block (the thesis novelty); a 2-level coarse-to-fine cascade
  * Agent_GLO_NCA           - patch-based multi-level training (low VRAM)
  * DiceCELoss              - Dice + per-region cross-entropy for multi-class
  * AdamW optimizer         - set in the agent from config['optimizer']

The default config is tuned to fit a 6 GB GPU (RTX 3050/3060). Adjust
input_size / channel_n / inference_steps if you have more or less VRAM.

Set BOTH img_path and label_path to the BraTS dataset root (one sub-folder per
patient); the loader finds the 4 modality files and the seg file inside each.
"""
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import TverskyCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_GLO_NCA import Agent_GLO_NCA

# ============================================================================
# This config reproduces the best run (v4): WT 0.885 / TC 0.730 / ET 0.734 on
# BraTS 2024 (24k params). It bundles the winning techniques:
#   * foreground crop + nonzero per-channel z-norm  (Swin-UNETR preprocessing)
#   * SE global-context block, 24 channels, hidden 96, 20 NCA steps/level
#   * 64^3 high-res patches, cosine LR (set up below), Tversky+BCE loss
# Cosine LR is applied after building the agent (see the schedule override).
# ============================================================================
config = [{
    # --- Paths (EDIT THESE) -------------------------------------------------
    # For BraTS, point BOTH to the dataset root (one sub-folder per patient),
    # e.g. the Kaggle ".../BraTS2024_small_dataset" folder.
    'img_path':   r"PATH_TO_BRATS_ROOT",
    'label_path': r"PATH_TO_BRATS_ROOT",
    'model_path': r"runs/GLO_NCA_BraTS",
    'device': "cuda:0",
    'unlock_CPU': True,
    # --- Optimizer ----------------------------------------------------------
    'optimizer': 'adamw',
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    'weight_decay': 1e-4,
    # --- Training -----------------------------------------------------------
    'save_interval': 10,
    'evaluate_interval': 10,
    'n_epoch': 150,
    'batch_size': 1,
    'batch_duplication': 1,
    # --- Model (v4 winning values) ------------------------------------------
    'channel_n': 24,               # was 16
    'inference_steps': [20, 20],   # was [10, 10]
    'cell_fire_rate': 0.6,         # was 0.5
    'input_channels': 4,           # T1, T1ce, T2, FLAIR
    'output_channels': 3,          # WT, TC, ET
    'hidden_size': 96,             # was 64
    'train_model': 1,              # 2-level cascade (3-level trained worse)
    'use_attention': True,         # SE global-context block (thesis novelty)
    # --- Data ---------------------------------------------------------------
    'input_size': [(32, 32, 24), (64, 64, 48)],   # 64^3 high-res patches
    'scale_factor': 2,
    'data_split': [0.7, 0.15, 0.15],              # train / val / test
    'keep_original_scale': True,
    'rescale': True,
    'patchify': True,
    'priotize_masks': 0.5,
    # --- Swin-UNETR preprocessing (the TC/ET win) ---------------------------
    'foreground_crop': True,       # crop to brain bbox before resize
    'nonzero_norm': True,          # z-norm on brain voxels only, per channel
}]

# --- Build experiment -------------------------------------------------------
dataset = Dataset_NiiGz_3D_BraTS()        # 3D, multi-modal (slice=None)
device = torch.device(config[0]['device'])

# Two NCAs: coarse (large kernel, global) -> fine (small kernel, detail).
ca1 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device,
                 hidden_size=config[0]['hidden_size'], kernel_size=7,
                 input_channels=config[0]['input_channels'],
                 use_attention=config[0]['use_attention']).to(device)
ca2 = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device,
                 hidden_size=config[0]['hidden_size'], kernel_size=3,
                 input_channels=config[0]['input_channels'],
                 use_attention=config[0]['use_attention']).to(device)
ca = [ca1, ca2]

agent = Agent_GLO_NCA(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                          batch_size=exp.get_from_config('batch_size'))

# Recall-focused loss for the small ET/TC regions (Tversky beta > alpha).
loss_function = TverskyCELoss(alpha=0.25, beta=0.75, ce_weight=0.5)

# Cosine LR decay over all steps (kills the sawtooth; the v4 winning schedule).
steps_per_epoch = max(1, len(data_loader))
total_steps = exp.get_from_config('n_epoch') * steps_per_epoch
agent.scheduler = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-5)
                   for opt in agent.optimizer]

# Parameter count (lightweight NCA selling point).
print("Trainable parameters:", sum(p.numel() for m in ca for p in m.parameters() if p.requires_grad))

# Train.
agent.train(data_loader, loss_function)

# Evaluate on the test set (prints Dice / mIoU / HD95 per region WT/TC/ET).
agent.getAverageDiceScore()
