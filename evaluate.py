import os
import pickle
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from sklearn.metrics import balanced_accuracy_score
from dataset import MITIndoorDataset
from argparse import Namespace, ArgumentParser
from pathlib import Path


def evaluate(args: Namespace):
    # LOAD MODEL
    model_path: Path = args.model
    root_path = Path(model_path).resolve().parent

    submission_path = str(model_path).split("\\")[-1].split(".")[-2].split("-")
    del submission_path[0]
    submission_path = str(submission_path[0]) + '-' + str(submission_path[1])

    model = torch.load(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transformations_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_dataset = MITIndoorDataset("data/val.txt", transformations_test)

    with open('data/class_dict.pkl', 'rb') as f:
        class_dict = pickle.load(f)

    labels = []
    preds = []
    miss_classified_samples = []
    acc = 0
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), total=len(test_dataset)):
            model.eval()
            image, target = test_dataset[idx]
            image = image.to('cuda:0')
            image = torch.unsqueeze(image, dim=0)
            pred = model(image)
            preds.append(pred.argmax(dim=1).detach().cpu().view(-1).numpy())
            labels.append(target)

            acc += (preds[-1] == target).sum().item()
            if preds[-1] != target:
                miss_classified_samples.append([image, target, preds[-1]])

    labels = np.array(labels, dtype='int32')
    preds = np.array(preds, dtype='int32')
    preds = preds[:, 0]

    # print('PREDs', np.shape(preds), preds.dtype)
    # print('LABELS', np.shape(labels), labels.dtype)

    balanced_acc = balanced_accuracy_score(labels, preds)
    print('BALANCE ACCURACY: ', round(balanced_acc, 4))

    submission = pd.DataFrame({'prediction': preds, 'label': labels})
    if not os.path.exists(root_path.joinpath('submission')):
        os.mkdir(root_path.joinpath('submission'))
    submission_file_name = f'SUBMISSION_{submission_path}_BALANCED-ACC-{round(balanced_acc, 4)}'
    submission_file_path = Path(root_path).joinpath('submission', submission_file_name + '.csv')
    submission.to_csv(submission_file_path, index=False)

    with open(str(Path(root_path).joinpath('submission', submission_file_name)), "wb") as fp:
        pickle.dump(miss_classified_samples, fp)


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate Trained Model on Test Set')
    parser.add_argument('--model', help="model weights", type=Path)

    evaluate(args=parser.parse_args())
