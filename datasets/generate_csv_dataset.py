import os
import sys 
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)
sys.path.pop(0) # remove the path of the current file
import pandas as pd
import numpy as np
from compressai.datasets import Vimeo90kDataset
from torchvision.datasets import ImageNet
from pathlib import Path
from PIL import ImageFile, Image
from torchvision import transforms
from torch.utils.data import Dataset
from utils.args import train_options
from datasets.datasets import ImageFolder, ImageFolders
    

config = train_options()

main_path = os.path.join(config.main_path, config.dataset)
patch_size = config.patch_size
csv_folder = os.path.join(repo_path, "datasets", "csv")
os.makedirs(csv_folder, exist_ok=True)
csv_file = os.path.join(repo_path, "datasets", "csv", config.dataset_csv + ".csv")

dataset_transforms = transforms.Compose([transforms.ToTensor()])
print("Creating the CSV dataset")
print("processing the image folder")
train_dataset1 = ImageFolder(os.path.join(main_path, 'train'), transform=dataset_transforms)
print("processing the vimeo dataset")
train_dataset_Vimeo90k = Vimeo90kDataset(os.path.join(main_path, 'vimeo_septuplet'), split="train", transform=dataset_transforms, tuplet=7)
test_dataset_Vimeo90k = Vimeo90kDataset(os.path.join(main_path, 'vimeo_septuplet'), split="valid", transform=dataset_transforms, tuplet=7)
print("processing the imageNet dataset")
train_dataset_ImageNet = ImageFolders(root=os.path.join(main_path, 'imageNet'), transform=dataset_transforms)

samples = []
samples.extend(train_dataset1.samples)
samples.extend(train_dataset_Vimeo90k.samples)
samples.extend(test_dataset_Vimeo90k.samples)
samples.extend(train_dataset_ImageNet.samples)

# save the samples to a text file
samples = [str(s) for s in samples]

with open(csv_file.replace(".csv", ".txt"), "w") as file:
    for s in samples:
        file.write(s + "\n")

# remove images that are smaller than the patch size
samples_new = [] 
for s in samples:
    if Image.open(s).size[0] >= patch_size[0] and Image.open(s).size[1] >= patch_size[1]:
        samples_new.append(s)
        print(s)
samples = samples_new

# remove the main path from the samples
samples = [str(s).replace(main_path, "") for s in samples]
df = pd.DataFrame(samples, columns=["image_path"])
df.to_csv(csv_file, index=False)
