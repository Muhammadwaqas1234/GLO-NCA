# extra/ — non-M3D-NCA code (kept for future use)

The main `src/` package is now focused **only on the M3D-NCA brain-tumor
pipeline**. Everything here is not used by M3D-NCA but was part of the original
framework — kept in case it is needed later (baselines, 2D Med-NCA, tutorials).

Nothing in `src/` imports anything from this folder, so the M3D-NCA pipeline
runs without it.

## Contents
```
extra/
├── src/
│   ├── agents/
│   │   ├── Agent_Med_NCA.py        # 2D Med-NCA training agent
│   │   └── Agent_UNet.py           # UNet baseline agent
│   ├── models/
│   │   ├── Model_BasicNCA.py       # 2D base NCA
│   │   └── Model_BackboneNCA.py    # 2D backbone NCA (Med-NCA)
│   ├── datasets/
│   │   └── Nii_Gz_Dataset.py       # deprecated 2D slice loader
│   └── examples/
│       ├── train_Med_NCA.py        # 2D Med-NCA example
│       ├── train_Backbone2D_NCA.py # single 2D NCA baseline
│       ├── train_Backbone3D_NCA.py # single 3D NCA baseline (no cascade)
│       ├── train_Unet2D.py         # UNet 2D baseline
│       └── train_Unet3D.py         # UNet 3D baseline
├── notebooks/                      # the matching .ipynb runners + CA tutorial
├── Tutorial/                       # Growing-NCA + Med-NCA hippocampus tutorials
└── runPdoc_documentation.py        # API-docs generator
```

## How to use one of these again
These files use `from src.xxx import ...` paths. To run one, copy the file you
need back into the matching `src/` sub-folder (e.g. move
`extra/src/models/Model_BasicNCA.py` -> `src/models/`), then run its example.
The `src/` package still contains all the shared base classes they depend on
(`Agent`, `Agent_NCA`, `Agent_Multi_NCA`, `Dataset_Base`, `Dataset_3D`,
`Experiment`, `helper`, `LossFunctions`).
