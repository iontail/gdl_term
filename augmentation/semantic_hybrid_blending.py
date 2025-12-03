"""
CLIP-based Semantic Hybrid Blending
Replaces regions in a generated image where the target class activation is low
with content from the original source image, guided by CLIP semantic features.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path

import clip
from scipy import ndimage


CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
    'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
    'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
    'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
    'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

class CLIPBlender:
    def __init__(self, device='cuda', clip_model='ViT-B/32'):
        self.device = device
        self.model, self.preprocess = clip.load(clip_model, device=device)
        self.model.eval()

        self.class_texts = [f"a photo of a {cls}" for cls in CIFAR100_CLASSES]
        with torch.no_grad():
            self.class_text_features = self.model.encode_text(
                clip.tokenize(self.class_texts).to(device)
            )
            self.class_text_features = F.normalize(self.class_text_features, dim=-1)

    def extract_spatial_features(self, image, patch_size=32, overlap=0.5):
        w, h = image.size
        stride = int(patch_size * (1 - overlap))
        
        num_patches_h = max(1, (h - patch_size) // stride + 1)
        num_patches_w = max(1, (w - patch_size) // stride + 1)

        patches = []
        patch_coords = []

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                left = min(j * stride, w - patch_size)
                top = min(i * stride, h - patch_size)
                right = left + patch_size
                bottom = top + patch_size

                patch = image.crop((left, top, right, bottom))
                patches.append(patch)
                patch_coords.append((i, j))

        patch_tensors = torch.stack([
            self.preprocess(p) for p in patches
        ]).to(self.device)

        with torch.no_grad():
            patch_features = self.model.encode_image(patch_tensors)
            patch_features = F.normalize(patch_features, dim=-1)
            patch_features = patch_features.float()

        feature_dim = patch_features.shape[-1]
        spatial_features = torch.zeros(num_patches_h, num_patches_w, feature_dim, dtype=torch.float32)

        for idx, (i, j) in enumerate(patch_coords):
            spatial_features[i, j] = patch_features[idx].cpu()

        return spatial_features

    def compute_class_activation_map(self, generated_features, class_idx):
        class_feature = self.class_text_features[class_idx:class_idx+1].float()  # [1, feature_dim]

        gen_flat = generated_features.float().view(-1, generated_features.shape[-1])  # [H*W, feature_dim]
        class_feature_cpu = class_feature.cpu()

        with torch.no_grad():
            similarities = gen_flat @ class_feature_cpu.T  # [H*W, 1]
            similarities = similarities.squeeze(-1)  # [H*W]

        activation_map = similarities.view(generated_features.shape[:2])

        activation_map = activation_map - activation_map.min()
        if activation_map.max() > 0:
            activation_map = activation_map / activation_map.max()

        return activation_map.numpy()

    def create_replace_mask(self, activation_map, threshold_percentile=30, smooth=True, kernel_size=5):
        threshold = np.percentile(activation_map, threshold_percentile)

        mask = (activation_map <= threshold).astype(np.float32)

        if smooth:
            mask = ndimage.gaussian_filter(mask.astype(float), sigma=kernel_size/3)
            mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

        return mask

    def blend_images(self, original_img, generated_img, mask, use_binary_replace=True):
        orig_np = np.array(original_img).astype(np.float32)
        gen_np = np.array(generated_img).astype(np.float32)

        if mask.shape != (orig_np.shape[0], orig_np.shape[1]):
            interp_mode = 'nearest' if use_binary_replace else 'bilinear'
            
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            mask_resized = F.interpolate(
                mask_tensor,
                size=(orig_np.shape[0], orig_np.shape[1]),
                mode=interp_mode,
                align_corners=None if interp_mode == 'nearest' else False
            ).squeeze().numpy()
        else:
            mask_resized = mask

        if use_binary_replace:
            mask_binary = (mask_resized > 0.5).astype(np.float32)

            if len(orig_np.shape) == 3:
                mask_3d = np.stack([mask_binary] * orig_np.shape[2], axis=-1)
            else:
                mask_3d = mask_binary

            blended = orig_np * mask_3d + gen_np * (1 - mask_3d)
        else:
            if len(orig_np.shape) == 3:
                mask_3d = np.stack([mask_resized] * orig_np.shape[2], axis=-1)
            else:
                mask_3d = mask_resized

            blended = orig_np * mask_3d + gen_np * (1 - mask_3d)

        return Image.fromarray(blended.astype(np.uint8))

    def process_pair(self, original_path, generated_path, output_path,
                     patch_size=32, threshold_percentile=30, class_name=None,
                     smooth=True, kernel_size=5, use_binary_replace=True):
        original_img = Image.open(original_path).convert('RGB')
        generated_img = Image.open(generated_path).convert('RGB')

        if original_img.size != generated_img.size:
            generated_img = generated_img.resize(original_img.size, Image.LANCZOS)

        class_idx = -1
        if class_name:
            try:
                class_idx = CIFAR100_CLASSES.index(class_name)
            except ValueError:
                print(f"Warning: Class '{class_name}' not found in CIFAR100_CLASSES.")
        
        if class_idx == -1:
            input_tensor = self.preprocess(original_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(input_tensor)
                image_features = F.normalize(image_features, dim=-1)
                similarities = (image_features @ self.class_text_features.T).squeeze(0)
                class_idx = similarities.argmax().item()
            print(f"Predicted class: {CIFAR100_CLASSES[class_idx]} (index: {class_idx})")
        
        generated_features = self.extract_spatial_features(generated_img, patch_size=patch_size)

        activation_map = self.compute_class_activation_map(generated_features, class_idx)

        h, w = original_img.size[1], original_img.size[0]
        activation_map_resized = F.interpolate(
            torch.from_numpy(activation_map).unsqueeze(0).unsqueeze(0).float(),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()

        mask = self.create_replace_mask(
            activation_map_resized,
            threshold_percentile=threshold_percentile,
            smooth=smooth,
            kernel_size=kernel_size
        )

        blended = self.blend_images(
            original_img, 
            generated_img, 
            mask, 
            use_binary_replace=use_binary_replace
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        blended.save(output_path)

        return blended, mask, activation_map_resized


def process_all_images(original_dir, generated_dir, output_dir,
                       patch_size=32, threshold_percentile=30, max_images=100,
                       smooth=True, kernel_size=5, use_binary_replace=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    blender = CLIPBlender(device=device)

    if not os.path.exists(original_dir):
        print(f"Error: Original directory '{original_dir}' does not exist.")
        return

    class_dirs = sorted([d for d in os.listdir(original_dir)
                         if os.path.isdir(os.path.join(original_dir, d))])

    total_images = 0
    processed = 0

    for class_name in tqdm(class_dirs, desc="Processing classes"):
        if processed >= max_images:
            print(f"Reached maximum image limit ({max_images}). Stopping.")
            break

        orig_class_dir = os.path.join(original_dir, class_name)
        gen_class_dir = os.path.join(generated_dir, class_name)
        out_class_dir = os.path.join(output_dir, class_name)

        if not os.path.isdir(gen_class_dir):
            continue

        orig_files = sorted([f for f in os.listdir(orig_class_dir)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for orig_file in tqdm(orig_files, desc=f"  {class_name}", leave=False):
            if processed >= max_images:
                break

            orig_path = os.path.join(orig_class_dir, orig_file)

            gen_files = [f for f in os.listdir(gen_class_dir)
                         if f.startswith(orig_file) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            if not gen_files:
                continue

            gen_file = gen_files[0]
            gen_path = os.path.join(gen_class_dir, gen_file)

            out_filename = gen_file.replace('_generated_', '_clip_blend_')
            if not out_filename.endswith('.jpg'):
                out_filename = os.path.splitext(out_filename)[0] + '.jpg'
            out_path = os.path.join(out_class_dir, out_filename)

            total_images += 1

            try:
                blender.process_pair(
                    orig_path, gen_path, out_path,
                    patch_size=patch_size,
                    threshold_percentile=threshold_percentile,
                    class_name=class_name,
                    smooth=smooth,
                    kernel_size=kernel_size,
                    use_binary_replace=use_binary_replace
                )
                processed += 1
            except Exception as e:
                print(f"Error processing {orig_path} -> {gen_path}: {e}")
                continue

    print(f"Completed: {processed} images processed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP-based Semantic Hybrid Blending")
    
    parser.add_argument("--original_dir", type=str,
                        default="result/original_resized",
                        help="Directory containing original images")
    
    parser.add_argument("--generated_dir", type=str,
                        default="result/generated",
                        help="Directory containing generated images")
    
    parser.add_argument("--output_dir", type=str,
                        default="result/clip_blended",
                        help="Directory to save blended images")
    
    parser.add_argument("--patch_size", type=int, default=32,
                        help="Patch size for feature extraction (smaller is finer but slower)")
    
    parser.add_argument("--threshold_percentile", type=int, default=30,
                        help="Percentile threshold for activation (lower means more original content)")
    
    parser.add_argument("--max_images", type=int, default=50000,
                        help="Maximum number of images to process")
    
    parser.add_argument("--no_smooth", action="store_true",
                        help="Disable mask smoothing")
    
    parser.add_argument("--kernel_size", type=int, default=5,
                        help="Gaussian blur kernel size (for smoothing)")
    
    parser.add_argument("--use_binary_replace", type=str, default="true",
                        choices=["true", "false", "True", "False", "1", "0"],
                        help="Use binary replacement (true) or soft blending (false)")

    args = parser.parse_args()

    use_binary_replace = args.use_binary_replace.lower() in ('true', '1', 'yes', 'y')

    process_all_images(
        args.original_dir,
        args.generated_dir,
        args.output_dir,
        patch_size=args.patch_size,
        threshold_percentile=args.threshold_percentile,
        max_images=args.max_images,
        smooth=not args.no_smooth,
        kernel_size=args.kernel_size,
        use_binary_replace=use_binary_replace
    )