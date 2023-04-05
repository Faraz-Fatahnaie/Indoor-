import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


def cutmix(data, targets, alpha):
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)
    batch_size = data.size()[0]
    index = torch.randperm(batch_size)
    x1, x2 = data, data[index]
    y1, y2 = targets, targets[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    x1[:, :, bbx1:bbx2, bby1:bby2] = x2[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    y1 = (lam * y1) + ((1 - lam) * y2)
    return x1, y1


def rand_bbox(size, lam):
    # generate a random square bbox
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def dataset_mean_var():
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(root="./data/Images", transform=transformations)

    train_loader = DataLoader(dataset, batch_size=1, num_workers=4)

    mean = 0.
    var = 0.
    samples = 0.
    for data, _ in tqdm(train_loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        var += data.var(2).sum(0)
        samples += batch_samples

    mean /= samples
    var /= samples

    print(f'Mean: {mean}\nVariance: {var}')

# if __name__=="__main__":
