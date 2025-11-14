import os
from torch.utils.data import Dataset
from PIL import Image
import random
from .aug_utils import Utils


class Mixer(Dataset):
    def __init__(self, original_dataset,fractal_imgs, idx_to_class):
        self.original_dataset = original_dataset
        self.idx_to_class = idx_to_class
        self.combine_counter = 0
        self.fractal_imgs = fractal_imgs
        self.utils = Utils()
        self.augmented_images = self.generate_augmented_images()

    def generate_augmented_images(self):
        augmented_data = []

        base_directory = '.datasets/mixed'
        original_resized_dir = os.path.join(base_directory, 'original_resized')
        generated_dir = os.path.join(base_directory, 'generated')
        fractal_dir = os.path.join(base_directory, 'fractal')

        # Ensure these directories exist
        os.makedirs(original_resized_dir, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)
        os.makedirs(fractal_dir, exist_ok=True)

        for idx, (img_path, label_idx) in enumerate(self.original_dataset.samples):

            label = self.idx_to_class[label_idx]  # Use folder name as label

            original_img = Image.open(img_path).convert('RGB')
            original_img = original_img.resize((256, 256))
            img_filename = os.path.basename(img_path)

            label_dirs = {dtype: os.path.join(base_directory, dtype, str(label)) for dtype in
                          ['original_resized', 'generated', 'fractal']}

            for dir_path in label_dirs.values():
                os.makedirs(dir_path, exist_ok=True)

            original_img.save(os.path.join(label_dirs['original_resized'], img_filename))

            img = img.resize((256, 256))

            combined_img = self.utils.combine_images(original_img, img)
            random_fractal_img = random.choice(self.fractal_imgs)
            blended_img = self.utils.blend_images_with_resize(combined_img, random_fractal_img)
            blended_img_filename = f"{img_filename}_blended.jpg"
            blended_img.save(os.path.join(label_dirs['blended'], blended_img_filename))
            augmented_data.append((blended_img, label))

        return augmented_data

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        image, label = self.augmented_images[idx]
        return image, label