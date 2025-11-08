import torchvision
from PIL import Image
from tqdm import tqdm
import os
import shutil



print("📥 CIFAR-100 다운로드 시작...")

# 저장 경로
base_path = './data/cifar100'
train_path = f'{base_path}/train'
test_path = f'{base_path}/test'

os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# 다운로드 (임시 위치)

trainset = torchvision.datasets.CIFAR100(
    root='./data/cifar100_temp',
    train=True,
    download=True
)



testset = torchvision.datasets.CIFAR100(
    root='./data/cifar100_temp',
    train=False,
    download=True
)

# Drive에 저장
def save_dataset_as_folders(dataset, save_path, split_name):
    print(f"\n💾 {split_name} 데이터를 저장 중...")
    
    classes = dataset.classes
    
    # 클래스별 폴더 생성
    for class_name in classes:
        class_path = os.path.join(save_path, class_name)
        os.makedirs(class_path, exist_ok=True)
    
    # 이미지 저장
    class_counts = {class_name: 0 for class_name in classes}
    
    for idx in tqdm(range(len(dataset)), desc=f"{split_name} 저장"):
        img, label = dataset[idx]
        class_name = classes[label]
        
        img_filename = f"{class_name}_{class_counts[class_name]:05d}.png"
        img_path = os.path.join(save_path, class_name, img_filename)
        
        img.save(img_path)
        class_counts[class_name] += 1
    
    print(f"✅ {split_name} 완료! 총 {len(dataset)}개 이미지 저장")

# 저장 실행
save_dataset_as_folders(trainset, train_path, "Train")
save_dataset_as_folders(testset, test_path, "Test")

# 임시 파일 삭제
shutil.rmtree('./data/cifar100_temp', ignore_errors=True)

print("\n" + "="*70)
print("🎉 CIFAR-100 다운로드 완료!")
print("="*70)
print(f"✅ Train: {train_path}")
print(f"✅ Test: {test_path}")

print(f"{len(train_path)} classes in {train_path}")
print(f"{len(test_path)} classes in {test_path}")
