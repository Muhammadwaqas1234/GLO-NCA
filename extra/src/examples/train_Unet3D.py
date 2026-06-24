r"""UNet3D baseline on multi-modal BraTS - for the NCA-vs-UNet comparison.

Trains a standard 3D U-Net on the same 4-modality BraTS input and 3-region
(WT/TC/ET) output as the M3D-NCA, so Dice / mIoU / HD95 are directly
comparable. Note the parameter count printed below: it is orders of magnitude
larger than the NCA - that contrast is the point of the thesis.
"""
from unet import UNet3D
from src.datasets.Nii_Gz_Dataset_3D import Dataset_NiiGz_3D_BraTS
from src.utils.Experiment import Experiment
import torch
from src.losses.LossFunctions import DiceCELoss
from src.agents.Agent_UNet import Agent

config = [{
    'img_path':   r"PATH_TO_BRATS_ROOT",
    'label_path': r"PATH_TO_BRATS_ROOT",
    'model_path': r"runs/UNet3D_BraTS",
    'device': "cuda:0",
    'unlock_CPU': True,
    # Optimizer
    'optimizer': 'adamw',
    'lr': 1e-4,
    'lr_gamma': 0.9999,
    'betas': (0.9, 0.99),
    'weight_decay': 1e-4,
    # Training
    'save_interval': 100,
    'evaluate_interval': 10,
    'n_epoch': 1000,
    'batch_size': 1,
    # Model / data
    'channel_n': 16,
    'cell_fire_rate': 0.5,
    'input_channels': 4,       # T1, T1ce, T2, FLAIR
    'output_channels': 3,      # WT, TC, ET
    'input_size': (64, 64, 48),
    'data_split': [0.7, 0.0, 0.3],
    'keep_original_scale': True,
    'rescale': True,
}]

dataset = Dataset_NiiGz_3D_BraTS()
device = torch.device(config[0]['device'])
# in_channels / out_classes must match the BraTS multi-modal / multi-region setup.
ca = UNet3D(in_channels=config[0]['input_channels'], padding=1,
            out_classes=config[0]['output_channels']).to(device)
agent = Agent(ca)
exp = Experiment(config, dataset, ca, agent)
exp.set_model_state('train')
dataset.set_experiment(exp)
data_loader = torch.utils.data.DataLoader(dataset, shuffle=True,
                                          batch_size=exp.get_from_config('batch_size'))
loss_function = DiceCELoss()

print("Trainable parameters:", sum(p.numel() for p in ca.parameters() if p.requires_grad))

agent.train(data_loader, loss_function)
agent.getAverageDiceScore()
