import os
import random
import torch
import numpy as np

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from functools import reduce
from operator import __or__

from .aug_utils import Utils


class FractalMixDataset(Dataset):
    def __init__(self, main_dataset, fractal_dataset):
        self.main_dataset = main_dataset
        self.fractal_dataset = fractal_dataset
        self.fractal_len = len(self.fractal_dataset)
        
        # 'targets' and 'classes' attributes are needed to apply SubsetRandomSampler
        if hasattr(main_dataset, 'targets'):
            self.targets = main_dataset.targets
        if hasattr(main_dataset, 'classes'):
            self.classes = main_dataset.classes

    def __len__(self):
        return len(self.main_dataset)

    def __getitem__(self, idx):
        main_img, main_label = self.main_dataset[idx]

        fractal_idx = random.randint(0, self.fractal_len - 1)
        fractal_img, _ = self.fractal_dataset[fractal_idx]
        return main_img, fractal_img, main_label


class Mixer(Dataset):
    def __init__(self,
                 img_size,
                 load: bool = True,
                 ratio: float = 0.5,
                 alpha=0.0,
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

    def generate_augmented_imgs(self, base_directory='./datasets', train_datadir='original'):
        augmented_data = []

        train_data_dir = os.path.join(base_directory, train_datadir)
        generated_dir = os.path.join(base_directory, 'generated')
        fractal_dir = os.path.join(base_directory, 'fractal')
        

        # Ensure these directories exist
        os.makedirs(base_directory, exist_ok=True)
        os.makedirs(generated_dir, exist_ok=True)
        os.makedirs(fractal_dir, exist_ok=True)


        original_dataset = self.utils.load_img(train_data_dir, self.img_size)
        generated_dataset = self.utils.load_img(generated_dir, self.img_size)
        fractal_dataset = self.utils.load_img_list(fractal_dir, self.img_size)

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


        match_fails = 0
        for idx, (img_path, label_idx) in tqdm(enumerate(original_dataset.samples), leave=False):
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


            if fractal_dataset is None or len(fractal_dataset) == 0:
                random_fractal_img = None
            else:
                rand_idx = random.randint(0, len(fractal_dataset) - 1)
                random_fractal_img = fractal_dataset[rand_idx]

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

        print(f"Total match fails: {match_fails}")
        return augmented_data


    def subset_loader(self,
                      batch_size,
                      workers,
                      dataset,
                      train_data_org_dir,
                      aug_data_dir=None,
                      test_data_dir=None,
                      labels_per_class=100,
                      valid_labels_per_class=500,
                      train_mode='vanilla',
                      fractal_dataset=None,
                      enlarge_dataset=False
                      ):
        
        if dataset == 'cifar10':
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
        elif dataset == 'cifar100':
            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]
        else:
            assert False, "Unknow dataset : {}".format(dataset)

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)])


        if dataset == 'cifar10':
            train_data = datasets.CIFAR10(train_data_org_dir,
                                        train=True,
                                        transform=train_transform,
                                        download=True)
            test_data = datasets.CIFAR10(test_data_dir,
                                        train=False,
                                        transform=test_transform,
                                        download=True)
            num_classes = 10

        elif dataset == 'cifar100':
            if not os.path.exists(test_data_dir):
                    raise FileNotFoundError(f"Test directory not found: {test_data_dir}")

            if aug_data_dir == 'None':
                train_data = datasets.ImageFolder(train_data_org_dir, transform=train_transform)
                print(f"Loading train data from: {train_data_org_dir}")
            else:
                from torch.utils.data import ConcatDataset
                
                train_root_1 = train_data_org_dir
                train_root_2 = aug_data_dir

                if not os.path.exists(train_root_1) or not os.path.exists(train_root_2):
                    raise FileNotFoundError(f"One of the train directories not found. Check paths.")
                

                print(f"Loading train data from: {train_root_1}")
                train_data_1 = datasets.ImageFolder(train_root_1, transform=train_transform)

                print(f"Loading augmented data from: {train_root_2}")
                if self.augmented_dataset is not None:
                    self.augmented_dataset.transform = train_transform
                    train_data_2 = self.augmented_dataset
                elif aug_data_dir is not None:
                    train_data_2 = datasets.ImageFolder(train_root_2, transform=train_transform)
                else:
                    raise ValueError("aug_data_dir must be provided if argument 'load' is False when initializing Mixer.")

                if train_data_1.classes != train_data_2.classes:
                    raise ValueError("Class list/order mismatch between the two train directories. "
                                    "This will cause incorrect labels.")
                    
                train_data = ConcatDataset([train_data_1, train_data_2])
                print(f"Combined two train datasets. Total size: {len(train_data)}")

                train_data.targets = np.concatenate([train_data_1.targets, train_data_2.targets])
                train_data.classes = train_data_1.classes

            test_data = datasets.ImageFolder(test_data_dir, transform=test_transform)
            
            num_classes = len(train_data.classes)
            print(f"Found {num_classes} classes total.")

        else:
            assert False, 'Unknown dataset : {}'.format(dataset)


        # new code
        if train_mode == 'fractal_mixup':
            if fractal_dataset is None:
                raise ValueError("Fractal_dataset must be provided when train_mode is 'fractal_mixup'")
            print(f"Wrapping main dataset with {len(fractal_dataset)} fractal images.")
            train_data = FractalMixDataset(main_dataset=train_data, fractal_dataset=fractal_dataset)

        n_labels = num_classes

        # random sampler
        def get_sampler(labels, n=None, n_valid=None):
            # Only choose digits in n_labels
            # n = number of labels per class for training
            # n_val = number of lables per class for validation
            (indices, ) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
            np.random.shuffle(indices)

            indices_valid = np.hstack([
                list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)
            ])
            indices_train = np.hstack([
                list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid + n]
                for i in range(n_labels)
            ])
            indices_unlabelled = np.hstack(
                [list(filter(lambda idx: labels[idx] == i, indices))[:] for i in range(n_labels)])

            indices_train = torch.from_numpy(indices_train)
            indices_valid = torch.from_numpy(indices_valid)
            indices_unlabelled = torch.from_numpy(indices_unlabelled)
            sampler_train = SubsetRandomSampler(indices_train)
            sampler_valid = SubsetRandomSampler(indices_valid)
            sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
            return sampler_train, sampler_valid, sampler_unlabelled

        if enlarge_dataset:
            print("Enlarging dataset by using all augmented data for training.")
            labelled = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers,
                                                pin_memory=True)
            validation = None
            unlabelled = None
            test = torch.utils.data.DataLoader(test_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=workers,
                                            pin_memory=True)

        else:
            train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.targets, labels_per_class, valid_labels_per_class)


            labelled = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                sampler=train_sampler,
                                                shuffle=False,
                                                num_workers=workers,
                                                pin_memory=True)
            validation = torch.utils.data.DataLoader(train_data,
                                                    batch_size=batch_size,
                                                    sampler=valid_sampler,
                                                    shuffle=False,
                                                    num_workers=workers,
                                                    pin_memory=True)
            unlabelled = torch.utils.data.DataLoader(train_data,
                                                    batch_size=batch_size,
                                                    sampler=unlabelled_sampler,
                                                    shuffle=False,
                                                    num_workers=workers,
                                                    pin_memory=True)
            test = torch.utils.data.DataLoader(test_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=workers,
                                            pin_memory=True)
        print(f"Actual train dataset size: {len(labelled.dataset)}")

        return labelled, validation, unlabelled, test, num_classes

    
    def __len__(self):
        if self.augmented_dataset is not None:
            return len(self.augmented_dataset)
        return len(self.augmented_datalist)

    def __getitem__(self, idx):
        if self.augmented_dataset is not None:
            img, label_idx = self.augmented_dataset[idx]
            return img, label_idx
        else:
            raise ValueError("self.augmented_dataset is None. Cannot get item.")
        


if __name__ == "__main__":
    img_size = 32
    ratio = 0.5
    alpha = 0.2
    active_lam = False
    retain_lam = False

    mixer = Mixer(img_size=img_size,
                  load=False,
                  ratio=ratio,
                  alpha=alpha,
                  active_lam=active_lam,
                  retain_lam=retain_lam)
    
    mixer.generate_augmented_imgs(base_directory='./datasets')

    print("Augmentation process completed.")
