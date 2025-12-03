# CellSamUnet: A Dual-Stream Feature Extraction Network for Cell Image Segmentation

## Introduction
This repository contains the official implementation of **CellSamUnet**, a novel cell image segmentation method that employs a dual-stream feature extraction network. By fusing the hierarchical features from the **Segment Anything Model 2 (SAM2)** image encoder with a traditional **U-Net** encoder, the architecture effectively captures both global semantic dependencies and fine-grained local details. Experimental results on the DynamicNuclearNet and MoNuSeg datasets demonstrate that CellSamUnet achieves superior segmentation performance compared to state-of-the-art baselines.

## Training on Datasets:
!gdown --id 1lVVfq_NRv-8ts8uGwzJRgqj6_RMpCnyB -O monuseg_train.zip

## Validating the trained models 
!gdown --id 141UpDydk6tRAS8OJecealaRp9dTOqP2v -O monuseg_val.zip

## Project Structure
```text
CellSamUnet/
├── checkpoints/             # Saved models (created automatically)
├── configs/
│   ├── defaults.py          # Configuration (paths, hyperparameters)
├── monuseg_data/            # Dataset folder structure
├── results/                 # Training logs and visualization
├── src/                     # Source code modules
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── trainer.py
│   └── utils.py
├── train.py                 # Main training script
├── test.py                  # Evaluation script
├── requirements.txt         # Python dependencies
└── README.md
```

## Clone the repository
git clone [https://github.com/yourusername/CellSamUnet.git](https://github.com/yourusername/CellSamUnet.git)
cd CellSamUnet

## Install Dependencies
First, install the general requirements:
pip install -r requirements.txt

## Install Segment Anything Model 2 (SAM2)
This project relies on Facebook Research's SAM2. You must clone and install it manually:
## Clone SAM2 inside the project
git clone [https://github.com/facebookresearch/sam2.git](https://github.com/facebookresearch/sam2.git)

## Enter the directory and install in editable mode
cd sam2
pip install -e .
cd ..


## Data Preparation
Please organize your dataset (e.g., MoNuSeg) as follows. You can configure the exact paths in configs/defaults.py.
```text
monuseg_data/
├── kmms_training/
│   ├── images/   # .tif or .png images
│   └── masks/    # .png binary masks
└── kmms_test/
    ├── images/
    └── masks/
```
