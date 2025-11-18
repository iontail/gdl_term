import torchvision
from PIL import Image
import os
from tqdm import tqdm
import shutil

print("📥 CIFAR-100 다운로드 및 저장 시작...")

# 저장 경로
train_path = './datasets/cifar100/train'
test_path = './datasets/cifar100/test'

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# CIFAR-100 다운로드
print("\n1️⃣ CIFAR-100 다운로드 중...")
trainset = torchvision.datasets.CIFAR100(
    root='/tmp/cifar100_temp',
    train=True,
    download=True
)

testset = torchvision.datasets.CIFAR100(
    root='/tmp/cifar100_temp',
    train=False,
    download=True
)

# 저장 함수
def save_dataset(dataset, save_path, split_name):
    print(f"\n2️⃣ {split_name} 데이터 저장 중...")
    
    classes = dataset.classes
    
    # 클래스별 폴더 생성
    for class_name in classes:
        os.makedirs(os.path.join(save_path, class_name), exist_ok=True)
    
    # 이미지 저장
    class_counts = {cls: 0 for cls in classes}
    
    for idx in tqdm(range(len(dataset)), desc=f"{split_name}"):
        img, label = dataset[idx]
        class_name = classes[label]
        
        filename = f"{class_name}_{class_counts[class_name]:05d}.png"
        filepath = os.path.join(save_path, class_name, filename)
        
        img.save(filepath)
        class_counts[class_name] += 1
    
    print(f"✅ {split_name} 완료: {len(dataset)}개 이미지")

# Train 저장
save_dataset(trainset, train_path, "Train")

# Test 저장
save_dataset(testset, test_path, "Test")

# 임시 파일 삭제
shutil.rmtree('/tmp/cifar100_temp', ignore_errors=True)

print("\n" + "="*70)
print("🎉 CIFAR-100 다운로드 완료!")
print("="*70)
print(f"✅ Train: {train_path}")
print(f"✅ Test: {test_path}")

# 구조 확인
print("\n📁 Train 폴더 구조:")
train_classes = os.listdir(train_path)
print(f"총 {len(train_classes)}개 클래스")
for cls in train_classes[:5]:
    count = len(os.listdir(os.path.join(train_path, cls)))
    print(f"  - {cls}/: {count}개 파일")
print("  ...")