import os
import torch
import math
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        images_dir = Path(root)
        self.samples = list(file for file in images_dir.iterdir() if file.is_file() and file.suffix in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"])        
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(Image.open(self.samples[index]).convert("RGB"))

    def __len__(self):
        return len(self.samples)

class ImageFolderCSV(ImageFolder):
    def __init__(self, root, csv_file, transform=None):
        self.samples = pd.read_csv(csv_file)["image_path"].tolist()
        self.samples = [os.path.join(root, sample[1:]) for sample in self.samples] # The csv file has a '/' at the beginning of the path
        self.transform = transform

class ImageFolders(ImageFolder):
    def __init__(self, root, transform=None):
        images_dir = Path(root)
        subdirs = [x for x in images_dir.iterdir() if x.is_dir()]
        self.samples = []
        for subdir in subdirs:
            self.samples.extend(list(file for file in subdir.iterdir() if file.is_file() and file.suffix in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"])) 
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(Image.open(self.samples[index]).convert("RGB"))

    def __len__(self):
        return len(self.samples)

class ConcatDataset(ImageFolder):
    def __init__(self, datasets,transform=None, patch_size=(256, 256)):
        self.datasets = datasets
        self.transform = transform
        self.samples = []
        for d in self.datasets:
            self.samples.extend(d.samples)
            self.patch_size = patch_size  
        self.check_images()
    def check_images(self):
        # drop images that are smaller than the patch size from the samples
        images_sizes = [Image.open(s).size for s in self.samples]
        samples_new = []
        for i, s in enumerate(self.samples):
            if images_sizes[i][0] >= self.patch_size[0] and images_sizes[i][1] >= self.patch_size[1]:
                samples_new.append(s)
        self.samples = samples_new