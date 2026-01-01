import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random


class CustomHaze:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation='test'):
        # Paths to train and test list files
        train_data_path = os.path.join(self.config.data.data_dir, 'data', 'custom_haze')
        
        # Training dataset
        train_dataset = CustomHazeDataset(
            train_data_path,
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            transforms=self.transforms,
            filelist='train.txt',
            parse_patches=parse_patches
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.config.training.batch_size,
            shuffle=True, 
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )
        
        # Validation dataset
        val_dataset = CustomHazeDataset(
            train_data_path,
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            transforms=self.transforms,
            filelist='test.txt',
            parse_patches=False
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader


class CustomHazeDataset(torch.utils.data.Dataset):
    def __init__(self, dir, n, patch_size, transforms, filelist, parse_patches=True):
        super().__init__()
        self.dir = dir
        self.transforms = transforms
        self.n = n
        self.patch_size = patch_size
        self.parse_patches = parse_patches

        self.imgs = []
        self.gt_imgs = []

        # Read the file list
        with open(os.path.join(dir, filelist), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    self.imgs.append(parts[0])
                    self.gt_imgs.append(parts[1])

        self.img_options = self.imgs

        print("Total training examples:", len(self.imgs))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        index_ = index % len(self.imgs)
        img_name = self.imgs[index_]
        gt_name = self.gt_imgs[index_]

        # Load images
        inp_img = PIL.Image.open(img_name).convert('RGB')
        tar_img = PIL.Image.open(gt_name).convert('RGB')

        # Get patches if needed
        if self.parse_patches:
            inp_img, tar_img = self._get_patch(inp_img, tar_img)

        # Apply transforms
        inp_img = self.transforms(inp_img)
        tar_img = self.transforms(tar_img)

        # Concatenate for conditional input
        img = torch.cat([inp_img, tar_img], dim=0)

        return img, img_name

    def _get_patch(self, inp_img, tar_img):
        w, h = inp_img.size
        p = self.patch_size
        
        if w < p or h < p:
            # If image is smaller than patch, resize
            inp_img = inp_img.resize((p, p), PIL.Image.BICUBIC)
            tar_img = tar_img.resize((p, p), PIL.Image.BICUBIC)
        else:
            # Random crop
            x = random.randint(0, w - p)
            y = random.randint(0, h - p)
            inp_img = inp_img.crop((x, y, x + p, y + p))
            tar_img = tar_img.crop((x, y, x + p, y + p))
        
        return inp_img, tar_img
