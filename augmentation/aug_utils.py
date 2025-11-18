import os
import random
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from active_fractal import mix_fractal

class Utils:
    @staticmethod
    def load_img(dir_path, img_size, transform_compose = None):
        if transform_compose is None:
            transform = transforms.Compose([transforms.Resize((img_size, img_size))])
        else:
            transform = transforms.Compose(
                [transforms.Resize((img_size, img_size))] + transform_compose.transforms
            )

        dataset = datasets.ImageFolder(dir_path, transform=transform)
        return dataset
    
    @staticmethod
    def load_img_list(dir_path, img_size):
        fractal_img_paths = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        return [Image.open(path).convert('RGB').resize((img_size, img_size)) for path in fractal_img_paths]


    @staticmethod
    def make_mask(original_img, ratio: float = 0.5):
        """
        Random several rectangular mask
        sum of makses ~~ ratio * (W * H)

        Args:
            original_img (PIL.Image): input image
            ratio (float): ratio of patch area w.r.t original image area

        Returns:
            mask (np.ndarray): shape (H, W, 3), {0.0, 1.0}
        """
        width, height = original_img.size
        total_area = width * height

        assert 0.0 <= ratio <= 1.0

        if ratio == 1.0:
            return np.ones((height, width, 3), dtype=np.float32)

        target_area = int(total_area * ratio)

        # set K(the patch number)
        # bigger the ratio is, more possible having many patch
        min_patches = 1
        max_patches = max((1, 5, int(ratio * 5)))  # max number = 10 * ratio
        K = random.randint(min_patches, max_patches)

        # assign each patch area
        weights = np.random.dirichlet(np.ones(K)) # setting each patch's area (sum of weight = 1)
        patch_areas = np.round(weights * target_area).astype(int)
        diff = target_area - int(patch_areas.sum()) # rataining area
        patch_areas[-1] += diff # setting last patch is a part of retaining area

        mask = np.zeros((height, width), dtype=np.float32)

        for area_i in patch_areas:
            if area_i <= 0:
                continue

            # preventing too much small or big patch
            min_patch_area = max(1, int(0.05 * total_area))  # at least 10%
            max_patch_area = int(0.5 * target_area)       
            area_i = int(np.clip(area_i, min_patch_area, max_patch_area))

            placed = False
            attempts = 0
            max_attempts = 50

            while not placed and attempts < max_attempts:
                attempts += 1
                aspect = np.random.uniform(0.5, 2.0) # ratio between height and width

                h = int(np.sqrt(area_i / aspect))
                w = int(aspect * h)

                h = max(1, min(h, height))
                w = max(1, min(w, width))

                if width == w:
                    x1 = 0
                else:
                    x1 = random.randint(0, width - w)
                if height == h:
                    y1 = 0
                else:
                    y1 = random.randint(0, height - h)

                # 이미 1로 채워진 부분과 안 겹치도록
                if np.all(mask[y1:y1 + h, x1:x1 + w] == 0):
                    mask[y1:y1 + h, x1:x1 + w] = 1.0
                    placed = True

        # (H, W) → (H, W, 3)
        mask = np.repeat(mask[:, :, None], 3, axis=2).astype(np.float32)
        return mask

    @staticmethod
    def combine_img_random_mask(base_img, overlay_img, ratio: float = 0.5):
        mask = Utils.make_mask(base_img, ratio)
        original_array = np.array(base_img, dtype=np.float32) / 255.0
        augmented_array = np.array(overlay_img, dtype=np.float32) / 255.0

        blended_array = (1 - mask) * original_array + mask * augmented_array
        blended_array = np.clip(blended_array * 255, 0, 255).astype(np.uint8)

        blended_img = Image.fromarray(blended_array)
        return blended_img
    
    @staticmethod
    def mix_fractal(base_img, fractal, alpha: float = 0.2, active_lam: bool = False, retain_lam: bool = False):
        if active_lam:
            lam = np.random.uniform(0.20, 0.30).astype(np.float32)
        elif not retain_lam:
            lam = np.float32(alpha)
        else:
            lam = np.float32(1.0)

        if base_img.size != fractal.size:
            fractal = fractal.resize(base_img.size)

        base_arr = np.array(base_img, dtype=np.float32) / 255.0
        fract_arr = np.array(fractal, dtype=np.float32) / 255.0

        mix_arr = lam * fract_arr + (1.0 - lam) * base_arr
        mix_arr = np.clip(mix_arr * 255.0, 0, 255).astype(np.uint8)
        mixed_img = Image.fromarray(mix_arr)

        return mixed_img, lam
    
    @staticmethod
    def augment_pipeline(
        base_img,
        overlay_img,
        ratio: float = 0.5,
        fractal_img = None,
        alpha: float = 0.0,
        active_lam: bool = False,
        retain_lam: bool = False
    ):
        
        combined_img = Utils.combine_img_random_mask(base_img, overlay_img, ratio)

        if fractal_img is not None and ratio > 0.0:
            mixed_img, lam = Utils.mix_fractal(combined_img, fractal_img, alpha, active_lam, retain_lam)
        else:
            mixed_img = combined_img
            lam = 1

        return combined_img, mixed_img, lam

    @staticmethod
    def blend_images_with_resize(base_img, overlay_img, alpha=0.20):
        overlay_img_resized = overlay_img.resize(base_img.size)
        base_array = np.array(base_img, dtype=np.float32)
        overlay_array = np.array(overlay_img_resized, dtype=np.float32)
        assert base_array.shape == overlay_array.shape and len(base_array.shape) == 3
        blended_array = (1 - alpha) * base_array + alpha * overlay_array
        blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
        blended_img = Image.fromarray(blended_array)
        return blended_img

    @staticmethod
    def combine_images(original_img, augmented_img, blend_width=20):
        width, height = original_img.size
        combine_choice = random.choice(['horizontal', 'vertical'])

        if combine_choice == 'vertical':  # Vertical combination
            mask = np.linspace(0, 1, blend_width).reshape(-1, 1)
            mask = np.tile(mask, (1, width))  # Extend mask horizontally
            mask = np.vstack([np.zeros((height // 2 - blend_width // 2, width)), mask,
                              np.ones((height // 2 - blend_width // 2 + blend_width % 2, width))])
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

        else:
            mask = np.linspace(0, 1, blend_width).reshape(1, -1)
            mask = np.tile(mask, (height, 1))  # Extend mask vertically
            mask = np.hstack([np.zeros((height, width // 2 - blend_width // 2)), mask,
                              np.ones((height, width // 2 - blend_width // 2 + blend_width % 2))])
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))

        original_array = np.array(original_img, dtype=np.float32) / 255.0
        augmented_array = np.array(augmented_img, dtype=np.float32) / 255.0

        blended_array = (1 - mask) * original_array + mask * augmented_array
        blended_array = np.clip(blended_array * 255, 0, 255).astype(np.uint8)

        blended_img = Image.fromarray(blended_array)
        return blended_img

    @staticmethod
    def is_black_image(image):
        histogram = image.convert("L").histogram()
        return histogram[-1] > 0.9 * image.size[0] * image.size[1] and max(histogram[:-1]) < 0.1 * image.size[0] * \
            image.size[1]