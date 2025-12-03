<p align="center">
  <h1 align="center">Rethinking Diffusion-Based Augmentation: Why Single Prompt Fails</h1>
  <p align="center">
    <a>Chanhee Lee*</a>
    ·
    <a>Jeonghwan Cho*</a>
    ·
    <a>Suhyun Kim*</a>
    ·
    <a>Jungyeon Kim*</a>
  </p>
  <p align="center">
    <i>Sungkyunkwan University · Department of Applied Artificial Intelligence</i><br>
    <i>2025-Fall Generative Deep Learning Course Term Project</i>
  </p>
</p>

#### [[Report]](https://www.google.com/)
---


## 📝 Abstract

Diffusion models have emerged as a powerful tool for data augmentation, synthesizing high-quality samples that preserve class semantics. However, the precise factors driving their performance—specifically in restricted settings like **single-prompt generation**—remain underexplored.

In this work, we conduct a controlled study to dissect the components of diffusion-based augmentation. We analyze the impact of **prompt diversity**, **mixing strategies**, and the integration of **synthetic fractals**. Our findings reveal that in the single-prompt regime, diffusion augmentation often fails to broaden the training distribution effectively. To address semantic inconsistencies, we introduce **CLIP-Guided Semantic Hybrid Blending**, a method that selectively replaces low-fidelity regions in generated images with original content.

---

## ⚙️ Installation

This codebase has been tested with **Python 3.10** and **CUDA 12.1**.

#### 1. Clone the Repository

```bash
git clone [https://github.com/iontail/gdl_term.git](https://github.com/iontail/gdl_term.git)
cd gdl_term
```

#### 2. Environment Setup

```bash
# Create and activate conda environment
conda create -n gdtp python=3.10 -y
conda activate gdtp

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install additional dependencies
pip install gco-wrapper matplotlib numpy six wandb tqdm gdown clip scipy

# Optional: System utilities
apt update && apt install -y tmux unzip
```

#### 3. Weights & Biases Setup

```bash
wandb login
```

## 📂 Data Preparation
We provide scripts to download the CIFAR-100 dataset and the pre-generated augmented samples used in our study.

#### 1. Download Standard Datasets

```bash
python download_cifar100.py
```

#### 2. Setup Directory Structure & Download Augmented Data
We organize the data into specific directories for original, generated, and mixed samples.

```bash
mkdir -p datasets/concat datasets/fractal datasets/blended datasets/generated datasets/original
cd datasets

# Concatenated Samples
gdown 1TsXi6THJSpcXKna3fkgZwNTJFEXA8ehZ
unzip concatenated.zip

# Fractal Images (DeviantArt)
gdown 1c7HVPiF9L0dAV3bG5Y20fDiQPrvHzhj1
unzip deviantart.zip

# Blended Samples
gdown 1oxPibnC2OiFRC_TjccH-dmPWw2RNp12v
unzip blended.zip

# Generated Samples (Diffusion Output)
gdown 1Ewb4sOfJi27VpIxBjhX_rEnPJGlQ97eG
unzip generated.zip

# Original CIFAR-100 Images (Sorted)
gdown 1BpGjSI1dTHj1SoR264LCKKXiN_GgadFY
unzip original.zip
```

## 🚀 Training
We use PreActResNet18 as the backbone architecture. You can run the training script directly or via the provided shell script.

#### Quick Start
```bash
bash script/train.sh
```

#### Custom Training Command
To run a specific configuration (e.g., using fractal mixing with an enlarged dataset):

```bash
python main.py \
    --dataset cifar100 \
    --train_org_dir ./datasets/cifar100/train \
    --train_aug_dir ./datasets/mixed \
    --test_dir ./datasets/cifar100/test \
    --root_dir ./output \
    --fractal_img_dir ./datasets/fractal \
    --workers 4 \
    --labels_per_class 500 \
    --arch preactresnet18 \
    --learning_rate 0.1 \
    --batch_size 128 \
    --momentum 0.9 \
    --decay 0.0001 \
    --epochs 300 \
    --schedule 100 200 \
    --enlarge_dataset \
    --use_wandb
```

<details>
<summary><span style="font-weight: bold;">Key Arguments</span></summary>

  #### --train_aug_dir
  Path to the augmented/mixed dataset.
  #### --fractal_img_dir
  Path to the fractal image dataset (if using fractal mixing).
  #### --enlarge_dataset
  If set, the model trains on the combined size of original + augmented data.
  #### --labels_per_class
  Number of samples per class to use from the original dataset.

</details>
<br>


## 🧩 Project Structure

```
gdl_term/
├── augmentation/
│   ├── fractal_aug.py             
│   ├── fractal_utils.py           
│   └── semantic_hybrid_blending.py
├── models/
|   ├── init.py
│   ├── preactresnet.py         # PreActResNet implementation (Default)
│   └── wide_resnet.py          # WideResNet implementation
```

## 🔍 Method Overview

#### 1. Progressive Augmentation Scheduling
We investigate how the timing and ratio of introducing augmented data affect learning. We implemented Linear, Warmup, and Step scheduling strategies to control the blend ratio $\rho(t)$ over epochs.

#### 2. Integration of Fractal Images
To introduce non-semantic structural diversity, we integrate fractal patterns into the training loop. This is handled by the FractalMixDataset class in augmentation/fractal_utils.py.

#### 3. CLIP-Guided Semantic Hybrid Blending
Unlike naive concatenation, we propose a semantic-aware blending strategy. Using CLIP, we identify regions in diffusion-generated images that lack semantic alignment with the target class and replace them with the original content. This ensures high fidelity and reduces artifacts. (See utils/semantic_hybrid_blending.py for implementation details)

## 🙏 Acknowledgements
This project is largely inspired by and built upon the following works:

[DiffuseMix: Label-Preserving Data Augmentation with Diffusion Models](https://github.com/khawar-islam/diffuseMix)

[PuzzleMix: Exploiting Saliency and Local Statistics for Optimal Mixup](https://github.com/snu-mllab/PuzzleMix)

We thank the authors for their amazing contributions to the community. 



