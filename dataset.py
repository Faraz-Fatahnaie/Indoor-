from torch.utils.data import Dataset
from PIL import Image


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
                img = Image.open(img_path).convert('RGB')
                if img is not None:
                    if self.transform:
                        img = self.transform(img)
                    return img, int(label)
            except:
                print('NOT PROCESSED: ', line)
                pass


if __name__ == "__main__":
    MITIndoorDataset("data/train.txt").__getitem__(0)
