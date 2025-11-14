import os
import re

def extract_numbers(base_directory='./datasets'):
    """
    Original과 Generated의 번호 범위를 비교
    """
    print("=" * 80)
    print("번호 범위 분석")
    print("=" * 80)
    
    # Original 분석
    print("\n[1] Original 데이터 번호 분석")
    print("-" * 80)
    original_dir = os.path.join(base_directory, 'cifar100/train/apple')
    
    if os.path.exists(original_dir):
        files = sorted(os.listdir(original_dir))
        orig_numbers = []
        
        for f in files:
            # apple_00000.png에서 00000 추출
            match = re.search(r'_(\d+)\.png$', f)
            if match:
                orig_numbers.append(int(match.group(1)))
        
        if orig_numbers:
            print(f"총 파일 수: {len(orig_numbers)}")
            print(f"번호 범위: {min(orig_numbers)} ~ {max(orig_numbers)}")
            print(f"처음 10개 번호: {sorted(orig_numbers)[:10]}")
            print(f"마지막 10개 번호: {sorted(orig_numbers)[-10:]}")
    
    # Generated 분석
    print("\n[2] Generated 데이터 번호 분석")
    print("-" * 80)
    generated_dir = os.path.join(base_directory, 'generated/apple')
    
    if os.path.exists(generated_dir):
        files = sorted(os.listdir(generated_dir))
        gen_numbers = []
        
        for f in files:
            # apple_s_000027.png_generated_...에서 000027 추출
            # 또는 golden_delicious_s_001167.png_generated_...에서 001167 추출
            match = re.search(r'_s_(\d+)\.png', f)
            if match:
                gen_numbers.append(int(match.group(1)))
        
        if gen_numbers:
            print(f"총 파일 수: {len(gen_numbers)}")
            print(f"번호 범위: {min(gen_numbers)} ~ {max(gen_numbers)}")
            print(f"처음 10개 번호: {sorted(gen_numbers)[:10]}")
            print(f"마지막 10개 번호: {sorted(gen_numbers)[-10:]}")
    
    # 겹치는 번호 확인
    print("\n[3] 번호 매칭 가능성 분석")
    print("-" * 80)
    
    if orig_numbers and gen_numbers:
        orig_set = set(orig_numbers)
        gen_set = set(gen_numbers)
        
        common = orig_set & gen_set
        
        print(f"Original 고유 번호 개수: {len(orig_set)}")
        print(f"Generated 고유 번호 개수: {len(gen_set)}")
        print(f"겹치는 번호 개수: {len(common)}")
        
        if common:
            print(f"겹치는 번호 예시 (최대 20개): {sorted(common)[:20]}")
        else:
            print("❌ 겹치는 번호가 하나도 없습니다!")
            print("\n이유:")
            print("  - Original은 CIFAR-100 데이터 (0~499 범위)")
            print("  - Generated는 ImageNet 데이터 (다른 ID 범위)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    extract_numbers('./datasets')