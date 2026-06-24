r"""Single-level 3D NCA baseline on BraTS (no coarse-to-fine cascade).

This is the "plain NCA" baseline for the thesis ablation: one BasicNCA3D with
no multi-level processing. Compare against train_M3D_NCA.py (multi-level +
SE global-context) to show what the global-context design adds.
Set use_attention below to also ablate the SE block on a single level.
"""
import torch
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
from src.models.Model_BasicNCA3D import BasicNCA3D
from src.losses.LossFunctions import DiceCELoss
from src.utils.Experiment import Experiment
from src.agents.Agent_NCA import Agent_NCA

config = [{
    'img_path':   r"PATH_TO_BRATS_ROOT",
    'label_path': r"PATH_TO_BRATS_ROOT",
    'model_path': r"runs/Backbone3D_NCA_BraTS",
    'device': "cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'optimizer': 'adamw',
    'lr': 16e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    'weight_decay': 1e-4,
    # Training
    'save_interval': 10,
    'evaluate_interval': 10,
    'n_epoch': 1000,
    'batch_size': 1,
    # Model
    'channel_n': 16,
    'inference_steps': 20,
    'cell_fire_rate': 0.5,
    'input_channels': 4,       # T1, T1ce, T2, FLAIR
    'output_channels': 3,      # WT, TC, ET
    'hidden_size': 64,
    'use_attention': False,    # flip to True to add SE global context
    # Data
    'input_size': (56, 56, 40),
    'data_split': [0.7, 0.0, 0.3],
    'keep_original_scale': True,
    'rescale': True,
}]

dataset = Dataset_NiiGz_3D_BraTS()
device = torch.device(config[0]['device'])
ca = BasicNCA3D(config[0]['channel_n'], config[0]['cell_fire_rate'], device,
                hidden_size=config[0]['hidden_size'], kernel_size=7,
                input_channels=config[0]['input_channels'],
                use_attention=config[0]['use_attention']).to(device)
agent = Agent_NCA(ca)
exp = Experiment(config, dataset, ca, agent)
dataset.set_experiment(exp)
exp.set_model_state('train')
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                          batch_size=exp.get_from_config('batch_size'))
loss_function = DiceCELoss()

print("Trainable parameters:", sum(p.numel() for p in ca.parameters() if p.requires_grad))

agent.train(data_loader, loss_function)
agent.getAverageDiceScore()
