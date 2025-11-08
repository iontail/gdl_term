import os
from torch.utils.data import Dataset
from PIL import Image
import random
from utils import Utils


class Hybrid(Dataset):
    def __init__(self, image_dir, fractal_img_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        self.fractal_images = Utils.load_fractal_images(fractal_img_dir)
        self.augmented_images = []

        for img_path in self.image_paths:
            original_img = Image.open(img_path).convert('RGB').resize((256, 256))
            overlay_img = random.choice(self.fractal_images)
            blended_img = Utils.blend_images_with_resize(original_img, overlay_img, alpha=0.2)
            combined_img = Utils.combine_images(original_img, blended_img, blend_width=20)

            if self.transform:
                original_img = self.transform(original_img)
                combined_img = self.transform(combined_img)

            label = os.path.basename(img_path).split('_')[0]  # Assuming label is part of filename
            self.augmented_images.append((combined_img, label))

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        image, label = self.augmented_images[idx]
        return image, label
