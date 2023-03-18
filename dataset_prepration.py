import os
import random
from torch.utils.data import Dataset
import torch.nn as nn
from PIL import Image
from pathlib import Path
import numpy as np
import cv2 as cv

data_dir = ".\data\Images"
class_dict = {}
base_dir = Path.cwd()
train_images = []
val_images = []
test_images = []

# Create class dictionary
for i, d in enumerate(os.listdir(data_dir)):
    if os.path.isdir(os.path.join(data_dir, d)):
        class_dict[d] = i

# Create a list of image paths and labels
class_path = []
for class_name in class_dict.keys():
    class_path = Path(os.path.join(data_dir, class_name))
    class_images = []
    for image_file in os.listdir(class_path):
        if image_file.endswith(".jpg"):
            if " " in str(image_file):
                old_name = image_file
                image_file = image_file.replace(" ", "__")
                os.rename(str(os.path.join(class_path, old_name)), str(os.path.join(class_path, image_file)))
            image_path = os.path.join(class_path, image_file)
            class_images.append((str(base_dir.joinpath(image_path)), class_dict[class_name]))
    # Shuffle and split the images into train, validation, and test sets
    random.shuffle(class_images)
    n_train = int(len(class_images) * 0.8)
    n_val = int(len(class_images) * 0.2)

    train_images.extend(class_images[0: n_train])
    val_images.extend(class_images[n_train: n_train + n_val])
    #test_images.extend(class_images[n_train + n_val:])


# Write the data to text files
with open("data/train.txt", "w") as f:
    for c in train_images:
        image_path, label = c
        f.write(f"{image_path} {label}\n")

with open("data/val.txt", "w") as f:
    for c in val_images:
        image_path, label = c
        f.write(f"{image_path} {label}\n")

#with open("data/test.txt", "w") as f:
    #for c in test_images:
        #image_path, label = c
        #f.write(f"{image_path} {label}\n")


# Define the custom dataset class
class MITIndoorDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.data_file = data_file
        self.transform = transform
        with open(self.data_file, "r") as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].strip()
        if line is not None:
            try:
                img_path, label = line.split(" ")
                #img = cv.imread(Path(img_path))
                #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = Image.open(img_path).convert('RGB')
                if img is not None:
                    if self.transform:
                        img = self.transform(img)
                    return img, int(label)
            except:
                print(line)

# if __name__ == "__main__":
#    MITIndoorDataset("data/train.txt").__getitem__(0)
