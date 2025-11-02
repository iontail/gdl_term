import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets

from .utils.augmentation import ModelHandler, Utils

# TODO
# it is for testing the new prompt whether the Diffusion generates filtered image
# with not changing its geometry characteristics

class Diffusion_Prompt(Dataset):
    def __init__(self, original_dataset, num_images, guidance_scale, idx_to_class, prompts, model_handler):
        self.original_dataset = original_dataset
        self.idx_to_class = idx_to_class
        self.combine_counter = 0
        self.prompts = prompts
        self.model_handler = model_handler
        self.num_augmented_images_per_image = num_images
        self.guidance_scale = guidance_scale
        self.utils = Utils()
        self.augmented_images = self.generate_augmented_images()

    def generate_augmented_images(self):
        augmented_data = []

        base_directory = './result'
        original_resized_dir = os.path.join(base_directory, 'original_resized')
        generated_dir = os.path.join(base_directory, 'generated')

        # Ensure these directories exist
        os.makedirs(original_resized_dir, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)

        for idx, (img_path, label_idx) in enumerate(self.original_dataset.samples):

            label = self.idx_to_class[label_idx]  # Use folder name as label

            original_img = Image.open(img_path).convert('RGB')
            original_img = original_img.resize((256, 256))
            img_filename = os.path.basename(img_path)

            label_dirs = {dtype: os.path.join(base_directory, dtype, str(label)) for dtype in
                          ['original_resized', 'generated']}

            for dir_path in label_dirs.values():
                os.makedirs(dir_path, exist_ok=True)

            original_img.save(os.path.join(label_dirs['original_resized'], img_filename))

            for prompt in self.prompts:
                augmented_images =  self.model_handler.generate_images(prompt, img_path, self.num_augmented_images_per_image,
                                                          self.guidance_scale)

                for i, img in enumerate(augmented_images):
                    img = img.resize((256, 256))
                    generated_img_filename = f"{img_filename}_generated_{prompt}_{i}.jpg"
                    img.save(os.path.join(label_dirs['generated'], generated_img_filename))

                    if not self.utils.is_black_image(img):
                        augmented_data.append((img, label))
        return augmented_data

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        image, label = self.augmented_images[idx]
        return image, label


if __name__ == "__main__":

    # please assign the "data_dir", "prompts", "guidance_scale" variables
    model_id = "timbrooks/instruct-pix2pix"
    data_dir = ""
    test_dataset = datasets.ImageFolder(root=data_dir)
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    prompts: list[str] = ['', '']
    guidance_scale = 4

    augmented_dataset = Diffusion_Prompt(
        original_dataset=test_dataset,
        num_images=1,
        guidance_scale=guidance_scale,
        idx_to_class=idx_to_class,
        prompts=prompts,
        model_handler=ModelHandler(model_id, device='cuda')
    )

    for idx, (image, label) in enumerate(augmented_dataset):
        image.save(f'augmented_images/{idx}.png')
        pass

