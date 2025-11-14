import os
from torch.utils.data import Dataset
from PIL import Image
import random
from .aug_utils import Utils


class Mixer(Dataset):
    def __init__(self,
                 img_size,
                 load: bool = True,
                 ratio: float = 0.5,
                 alpha=0.2,
                 active_lam=False,
                 retain_lam=False
                 ):
        
        self.combine_counter = 0
        self.img_size = img_size
        self.utils = Utils()
        self.ratio = ratio
        self.alpha = alpha
        self.active_lam = active_lam
        self.retain_lam = retain_lam

        if load:
            self.augmented_dataset = self.utils.load_img('./datasets/mixed', self.img_size)
        else:
            self.augmented_dataset = None

    def generate_augmented_imgs(self):
        augmented_data = []

        base_directory = './datasets'
        original_dir = os.path.join(base_directory, 'original')
        generated_dir = os.path.join(base_directory, 'generated')
        fractal_dir = os.path.join(base_directory, 'fractal')
        

        # Ensure these directories exist
        os.makedirs(base_directory, exist_ok=True)
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)
        os.makedirs(fractal_dir, exist_ok=True)

        original_dataset = self.utils.load_img(original_dir, self.img_size)
        generated_dataset = self.utils.load_img(generated_dir, self.img_size)
        fractal_dataset = self.utils.load_img(fractal_dir, self.img_size)

        self.idx_to_class = {v: k for k, v in original_dataset.class_to_idx.items()}

        # matching generated images to their originals
        # to do that, we create a mapping from (label, original_stem) to generated image path
        gen_index = {}

        for gen_path, gen_label_idx in generated_dataset.samples:
            gen_label = self.idx_to_class[gen_label_idx]  #
            gen_fname = os.path.basename(gen_path)        # 'apple_s_000058.png_generated_rainbow_0.jpg'

            gen_root, _ = os.path.splitext(gen_fname)     # 'apple_s_000058.png_generated_rainbow_0'
            prefix = gen_root.split('_generated')[0]      # 'apple_s_000058.png'
            orig_stem, _ = os.path.splitext(prefix)       # 'apple_s_000058'

            key = (gen_label, orig_stem)
            gen_index[key] = gen_path # only one generated image per original (we use random one prompt per one original data sample)
        

        label_dirs = {dtype: os.path.join(base_directory, dtype, str(label)) for dtype in
                      ['original', 'generated', 'combined', 'mixed']}

        for dir_path in label_dirs.values():
            os.makedirs(dir_path, exist_ok=True)


        match_fails = 0
        for idx, (img_path, label_idx) in enumerate(original_dataset.samples):
            label = self.idx_to_class[label_idx]  # Use folder name as label

            label_dirs = {dtype: os.path.join(base_directory, dtype, str(label)) for dtype in 
                          ['original', 'generated', 'combined', 'mixed']}
            for dir_path in label_dirs.values():
                os.makedirs(dir_path, exist_ok=True)

            origin_file_name = os.path.basename(img_path)  # 'apple_s_000058.png'
            origin_stem, _ = os.path.splitext(origin_file_name)  # 'apple_s_000058'

            key = (label, origin_stem)
            gen_path = gen_index.get(key, None)
            if gen_path is None:
                match_fails += 1
                continue

            original_img = Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size))
            gen_img = Image.open(gen_path).convert('RGB').resize((self.img_size, self.img_size))
            #original_img.save(os.path.join(label_dirs['original_resized'], origin_file_name))


            rand_idx = random.randint(0, len(fractal_dataset) - 1)
            random_fractal_img = fractal_dataset[rand_idx][0]
            combined_img, mixed_img, lam = self.utils.augment_pipeline(
                original_img,
                gen_img,
                ratio=self.ratio,
                fractal_img=random_fractal_img,
                alpha=self.alpha,
                active_lam=self.active_lam,
                retain_lam=self.retain_lam
            )

            combined_img.save(os.path.join(label_dirs['combined'], f"{origin_stem}_combined.jpg"))
            mixed_img.save(os.path.join(label_dirs['mixed'], f"{origin_stem}_mixed.jpg"))
            augmented_data.append((mixed_img, label_idx))

        self.augmented_datalist = augmented_data
        print(f"Total match fails: {match_fails}")
        return augmented_data


    def subset_loader(self,
                      batch_size,
                      workers,
                      dataset,
                      data_train_org_dir,
                      data_train_aug_dir=None,
                      data_test_dir=None,
                      labels_per_class=100,
                      valid_labels_per_class=500,
                      mixup_alpha=1,
                      train_mode='vanilla',
                      fractal_dataset=None
                      ):
        

        pass
    
    def __len__(self):
        if self.augmented_dataset is not None:
            return len(self.augmented_dataset)
        return len(self.augmented_datalist)

    def __getitem__(self, idx):
        if self.augmented_dataset is not None:
            img, label_idx = self.augmented_dataset[idx]
            return img, label_idx
        else:
            img, label_idx = self.augmented_datalist[idx]
            return img, label_idx