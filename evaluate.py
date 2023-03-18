import pickle
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from dataset_prepration import MITIndoorDataset

transformations_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

test_dataset = MITIndoorDataset("data/test.txt", transformations_test)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=True)

model = torch.load('trained_model/best_model.pt')
model.to('cuda:0')

labels = []
preds = []
confs = []

with torch.no_grad():
    for idx in tqdm(range(len(test_dataset)), total=len(test_dataset)):
        model.eval()
        image, target = test_dataset[idx]
        image = image.to('cuda:0')
        image = torch.unsqueeze(image, dim=0)
        pred = model(image)
        confs.append(pred.detach().cpu().view(-1).numpy())

        pred = 1 if pred > args.thresh else 0
        preds.append(pred)
        labels.append(target.detach().cpu().numpy())
