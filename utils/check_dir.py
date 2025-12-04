import os
import argparse


def collect_generated_filenames(root_dir, generated_subdir="generated",
                                exts=(".png", ".jpg", ".jpeg")):
    """
    root_dir/generated_subdir 아래의 모든 이미지 파일 이름을 재귀적으로 수집.

    ex)
        root_dir = "./datasets"
        generated_subdir = "generated"
        -> "./datasets/generated/..." 아래 파일들

    Returns:
        List[str]: 파일 이름(또는 경로) 리스트
    """
    target_root = os.path.join(root_dir, generated_subdir)

    if not os.path.exists(target_root):
        raise FileNotFoundError(f"Generated directory not found: {target_root}")

    filenames = []

    # 재귀적으로 하위 폴더까지 탐색
    for dirpath, dirnames, files in os.walk(target_root):
        for fname in files:
            if fname.lower().endswith(exts):
                # 1) 파일 이름만 원하면 이걸 사용
                # filenames.append(fname)

                # 2) 클래스/서브폴더 정보까지 포함한 상대 경로가 필요하면 이걸 사용
                rel_path = os.path.relpath(os.path.join(dirpath, fname), start=target_root)
                filenames.append(rel_path)

    return sorted(filenames)


def main():
    parser = argparse.ArgumentParser(
        description="generated 이미지 파일 이름(or 상대 경로) 리스트 추출"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./datasets",
        help="베이스 디렉토리 (ex: ./datasets 또는 ./result)"
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default="generated",
        help="root 내부의 generated 폴더 이름 (default: generated)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과를 저장할 txt 파일 경로 (지정 안 하면 stdout에만 출력)"
    )
    parser.add_argument(
        "--filenames-only",
        action="store_true",
        help="경로 말고 순수 파일 이름만 출력하고 싶으면 사용"
    )

    args = parser.parse_args()

    # 파일 목록 수집
    all_paths = collect_generated_filenames(args.root, args.subdir)

    if args.filenames_only:
        # 클래스/폴더 상관없이 순수 파일 이름만
        results = sorted({os.path.basename(p) for p in all_paths})
    else:
        # generated/ 기준 상대 경로 (ex: apple/apple_000058.png_generated_...)
        results = all_paths

    # stdout에 출력
    print(f"총 {len(results)} 개의 파일을 찾았습니다.\n")
    for name in results:
        print(name)

    # 파일로도 저장하기 원하면
    if args.output is not None:
        with open(args.output, "w", encoding="utf-8") as f:
            for name in results:
                f.write(name + "\n")
        print(f"\n➡ 결과를 '{args.output}' 에 저장했습니다.")


if __name__ == "__main__":
    main()