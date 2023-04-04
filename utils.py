import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


if __name__=="__main__":
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