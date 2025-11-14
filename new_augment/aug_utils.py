import os
import random
import numpy as np
from PIL import Image
from ..active_fractal import mix_fractal

class Utils:
    @staticmethod
    def load_fractal_images(fractal_img_dir):
        fractal_img_paths = [os.path.join(fractal_img_dir, fname) for fname in os.listdir(fractal_img_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        return [Image.open(path).convert('RGB').resize((256, 256)) for path in fractal_img_paths]


    @staticmethod
    def make_mask(original_img, augmented_img, ratio: float=0.5, blend_width=20, ):
        """
        Make mask for random mask with some ratio compared to the original image size
        Args:
            ratio (float): 0.5 means make mask with total size is half of that of original size
        Returs:
            mask
        """
        #TODO: Implement Random Mask for given ratio
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

        return mask

    @staticmethod
    def combine_img_random_mask(base_img, overlay_img, ratio: float = 0.5):
        mask = Utils.make_mask(base_img, overlay_img, ratio, blend_width=20)
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

        lam_reshape = lam

        mix_data = lam_reshape * fractal + (1.0 - lam_reshape) * base_img

        return mix_data.astype(np.float32), lam
    
    @staticmethod
    def augment_pipeline(
        base_img,
        overlay_img,
        ratio: float = 0.5,
        fractal_img = None,
        alpha: float = 0.2,
        active_lam: bool = False,
        retain_lam: bool = False
    ):
        
        combined_img = Utils.combine_img_random_mask(base_img, overlay_img, ratio)

        if fractal_img is not None:
            aug_img, lam = Utils.mix_fractal(combined_img, fractal_img, alpha, active_lam, retain_lam)
        else:
            aug_img = combined_img
            lam = 1

        return aug_img, lam

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