
## 📦 Installation (colab의 경우 git clone만 해도 될수도)

0.  *** puzzlemix repo 참고 ***
1.  **리포지토리 클론:**
    ```bash
    git clone https://github.com/ai-cho/GDTP.git
    cd GDTP
    ```

2.  **Conda 환경 생성 및 PyTorch 설치:**
    이 코드는 `Python 3.10` 및 `CUDA 12.1` 환경에서 테스트되었습니다.

    ```bash
    # 1. Conda 환경 생성
    conda create -n gdtp python=3.10 -y
    
    # 2. 환경 활성화
    conda activate gdtp
    
    # 3. PyTorch (CUDA 12.1) 설치
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```

3.  **추가 라이브러리 설치 및 데이터 다운로드:**
    ```bash
    pip install gco-wrapper matplotlib numpy six wandb tqdm gdown
    apt update && apt install -y tmux
    apt update && apt install -y unzip
    wandb login

    python download_cifar100.py
    mkdir datasets
    cd datasets
    mkdir concat
    mkdir fractal
    mkdir blended
    
    gdown 1TsXi6THJSpcXKna3fkgZwNTJFEXA8ehZ
    unzip concatenated.zip

    gdown 1LDh58LuQ9HkAjTliVv7tzCmgVZ9zOrCS 
    unzip fractal.zip

    gdown 1oxPibnC2OiFRC_TjccH-dmPWw2RNp12v
    unzip blended.zip

    ```

---

## 👟 Training

아래는 `preactresnet18` 아키텍처를 사용하여 CIFAR-100 데이터셋으로 모델을 학습시키는 예시 명령어입니다.

```
bash script/train.sh
```

```bash
python main.py --dataset cifar100 \
    --train_org_dir ./datasets/cifar100/train \
    --train_aug_dir ./datasets/concat \
    --test_dir ./datasets/cifar100/test \
    --root_dir ./output \
    --fractal_img_dir ./datasets/fractal \
    --workers 8 \
    --labels_per_class 500 \
    --arch preactresnet18 \
    --learning_rate 0.1 \
    --batch_size 128 \
    --momentum 0.9 \
    --decay 0.0001 \
    --epochs 300 \
    --schedule 100 200 \
    --train fractal_mixup \
    --fractal_alpha 0.2 \
    --active_lam \
    --use_wandb
