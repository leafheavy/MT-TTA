"""
GitHub Address: 
@INPROCEEDINGS{9674124,
author={Tunio, Muhammad Hanif and Jianping, Li and Butt, Muhammad Hassaan Farooq and Memon, Imran},
booktitle={2021 18th International Computer Conference on Wavelet Active Media Technology and Information Processing (ICCWAMTIP)},
title={Identification and Classification of Rice Plant Disease Using Hybrid Transfer Learning},
year={2021},
volume={},
number={},
pages={525-529},
doi={10.1109/ICCWAMTIP53232.2021.9674124}}

"""

import os
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt

import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from corruption_method import *
from utils import *

args = parse_args()

# Defining dataset class
class MyDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        labels = self.data.iloc[idx, 1:].values.astype(int)
        if self.transform:
            image = self.transform(image)
        return image, labels

# Define data transformation operations
def data_transforms(image_size=args.image_size, corruption=None, severity=1):
    """
    - args:
        - image_size: Image size for resizing
        - corruption: Add noise (options: gaussian_noise, shot_noise, impulse_noise, defocus_blur, brightness, contrast)
        - severity: Noise intensity (1-5)
    - return:
        - transforms: Data augmentation/Normalization operations
    """

    transforms_list = [transforms.Resize((image_size, image_size))]

    if corruption is not None:
        transforms_list.extend([
            transforms.Lambda(lambda img: np.array(img)),  # PIL -> numpy (0-255)
            transforms.Lambda(lambda x: corruption(x, severity)),  # corrupt original images
            transforms.Lambda(lambda x: x.astype(np.float32) / 255.0),  # normalization
        ])

    transforms_list.append(transforms.ToTensor())

    return transforms.Compose(transforms_list)

transforms_clean = data_transforms(corruption=None) # No corruption
# transforms_corrpution = data_transforms(corruption=gaussian_noise, severity=5) # Gaussian Noise
# transforms_corrpution = data_transforms(corruption=shot_noise, severity=3) # Shot Noise
# transforms_corrpution = data_transforms(corruption=impulse_noise, severity=3) # Impulse Noise
# transforms_corrpution = data_transforms(corruption=defocus_blur, severity=3) # Defocus Blur
# transforms_corrpution = data_transforms(corruption=brightness, severity=3) # Brightness
# transforms_corrpution = data_transforms(corruption=contrast, severity=3) # Contrast

# Generate PlantData dataset
def create_plantdata(root, split, batch_size=args.batch_size, transforms=transforms_clean, proportion=None, shuffle=True):
    """
    - args:
        - root: Dataset path
        - split: Dataset split ('train', 'test')
            - 'train': Training set
            - 'test': Test set
        - batch_size: Batch size
        - transforms: Data augmentation/Normalization operations
        - proportion: Proportion of the entire dataset to use for the specified split
            - None (default): Use default split (80% train, 20% test)
            - 0.7: Use 70% of entire dataset for the specified split
        - shuffle: Whether to shuffle the dataset
    - return:
        - plantdata_loader: PlantData data loader
    """

    image_path = os.path.join(root, 'PlantData', 'pictures')
    if split not in ['train', 'test']:
        raise ValueError("split must be either 'train' or 'test'")
        
    # Use default split
    if split == 'train' and proportion is None:
        file_name =  os.path.join(root, 'PlantData', 'balanced_train.csv') # Please replace with actual CSV filename
        plantdata = pd.read_csv(file_name) # Read CSV file containing image names and labels
        
        return plantdata_loader

    elif split == 'test' and proportion is None:
        file_name =  os.path.join(root, 'PlantData', 'balanced_test.csv') # Please replace with actual CSV filename
        plantdata = pd.read_csv(file_name) # Read CSV file containing image names and labels

        return plantdata_loader

    # User-defined split
    else:
        train_file =  os.path.join(root, 'PlantData', 'balanced_train.csv')  # 请将文件名替换为实际的CSV文件名
        test_file =  os.path.join(root, 'PlantData', 'balanced_test.csv')   # 请将文件名替换为实际的CSV文件名
        train_plantdata = pd.read_csv(train_file)
        test_plantdata = pd.read_csv(test_file)
        # Merge training and test data
        full_data = pd.concat([train_plantdata, test_plantdata], ignore_index=True)

        # Custom split: Random sampling from entire dataset by proportion
        total_samples = int(len(full_data) * proportion)
        
        random_indices = np.random.permutation(len(full_data))
      
        if split == 'train':
            selected_indices = random_indices[:total_samples]
        # Test set takes all remaining samples
        else:
            selected_indices = random_indices[total_samples:]
            
        # Shuffle dataset again
        plantdata = full_data.iloc[selected_indices].reset_index(drop=True)
        print(f"Selected {len(plantdata)} samples from {len(full_data)} total samples for {split} set.")

    plantdata_dataset = MyDataset(plantdata, root_dir=image_path, transform=transforms)
    plantdata_loader = DataLoader(plantdata_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True, num_workers=4)

    return plantdata_loader

# Generate CelebA dataset
def create_celeba(root, split, batch_size=args.batch_size, transforms=transforms_clean, selected_attrs=None, shuffle=True, max_samples=None):
    """
    - args:
        - root: Dataset path
        - split: Dataset split ('train': 70%, 'test': 30%)
        - batch_size: Batch size
        - transforms: Data augmentation/Normalization operations
        - selected_attrs: List of selected attributes (default None uses all 40 attributes)
        - shuffle: Whether to shuffle the dataset
        - max_samples: Number of samples in dataset (for taking subset during debugging)
    - return:
        - celeba_dataloader: CelebA data loader
    """

    def get_celeba_attr_names(data_path):
        attr_file = os.path.join(data_path, 'celeba', 'list_attr_celeba.txt')
        with open(attr_file, 'r') as f:
            f.readline() # Generate CelebA dataset
            attr_names = f.readline().strip().split() # Second line contains attribute names
        return attr_names

    # Get attribute names list in advance
    all_attr_names = get_celeba_attr_names(root)

    # If no specific attributes selected, use all 40
    if selected_attrs is None:
        selected_attrs = all_attr_names
        target_transform = None  # No transformation, directly get all attributes
    else:
        class SelectAttributes:
            def __init__(self, target_names, all_attr_names):
                self.target_indices = [all_attr_names.index(name) for name in target_names]
                self.target_names = target_names
            
            def __call__(self, attr_tensor):
                return attr_tensor[self.target_indices]

        target_transform = SelectAttributes(selected_attrs, all_attr_names)

    # Task names are the selected attributes
    task_names = selected_attrs
        
    celeba_dataset = datasets.CelebA(
        root=root,
        split="all",
        target_type='attr',                # Default use attribute labels (options: 'attr', 'identity', 'bbox', 'landmarks')
        transform=transforms,
        target_transform=target_transform, # Select specific attributes
        download=False
    )

    # Following the Multi_Tent paper, we split the dataset into 70% train and 30% test.
    total_size = len(celeba_dataset)
    train_size = int(0.7 * total_size)
    if split == 'train':
        indices = list(range(train_size))
    else:
        indices = list(range(train_size, total_size))

    celeba_dataset = Subset(celeba_dataset, indices)

    if max_samples is not None:
        subset_size = min(max_samples, total_size) # Prevent exceeding dataset size
        print(f"Random Sampling {subset_size} data from {total_size}")
        
        indices = torch.randperm(subset_size).tolist() # Randomly select subset
        celeba_dataset = Subset(celeba_dataset, indices)

    # Create DataLoader
    celeba_dataloader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)

    return celeba_dataloader, task_names