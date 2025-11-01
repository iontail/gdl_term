# IGL Term Project
<p align="center">
  <h1 align="center">Data Augmentation</h1>
  <p align="center">
    <a>Chanhee Lee</a>
    ·
    <a>Junghwan Cho</a>
    ·
    <a>Suhyun Kim</a>
    ·
    <a>Jungyeon Kim</a>
  </p>
  <p align="center">
    <i>Sungkyunkwan University · Department of Applied Artificial Intelligence</i><br>
    <i>2025-Fall Generative Deep Learning Course Term Project</i>
  </p>
</p>




## 📝 Abstract

blank


## 1. Installation

To set up the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/iontail/igl_term.git
   ```

2. Navigate to the project directory:

   ```bash
   cd igl_term
   ```

3. Create a new virtual environment and install dependencies:

   ```bash
   conda create -n igl_term python=3.10
   conda activate igl_term
   pip install -r requirements.txt
   ```

4. Set up the dataset and run the code:

   ```bash
   blank
   ```


## 📁 Project Structure
```
data/                      # Dataset directory

data_utils/
├── dataset.py             # Dataset functions
└── dataloader.py          # DataLoader implementation

models/
├── resnet.py              # ResNet model
├── preactresnet.py        # PreActivation ResNet model
└── get_model.py           # Model loader

utils/
├── scheduler.py           # Learning rate scheduler
└── augmentation/          # Data augmentation modules (!important)
    └── ...

train.py                   # Main training script
arguments.py               # CLI argument parser
trainer.py                 # Trainer class
text_prompt.py             # Text prompt testing
```

---
We thank [DiffuseMix.Pytorch](https://github.com/khawar-islam/diffuseMix.git) for their amazing works!