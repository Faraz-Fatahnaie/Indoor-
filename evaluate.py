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
import matplotlib.pyplot as plt
from collections import Counter
import operator
import random


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
    submission_file_name = f'SUBMISSION_{submission_path}_BALANCED-ACC-{round(balanced_acc, 4)}.csv'
    submission_file_path = Path(root_path).joinpath('submission', submission_file_name)
    submission.to_csv(submission_file_path, index=False)

    top_n_error_class(miss_classified_samples, class_dict)
    error_analyzer(miss_classified_samples, class_dict)


def top_n_error_class(miss_classified_samples, class_dict, n=10):
    """Given a list of misclassified samples and a dictionary mapping class,
        returns the names of the top n classes with the highest misclassified samples.

        Args:
        - miss_classified_samples (list): a list of triplets (img, true_label, predicted_label)
            representing miss-classified samples.
        - class_dict (dict): A dictionary mapping class names to integer indices.
        - n (int): The number of top classes to return. Defaults to 10.

        Returns:
        - top_classes (list): A list of the highest misclassified samples.
    """

    miss_classified_true_label = [lbl[1] for lbl in miss_classified_samples]
    class_list = tuple(Counter(miss_classified_true_label).keys())
    freq = list(Counter(miss_classified_true_label).values())
    indexed = list(enumerate(freq))
    top_n = sorted(indexed, key=operator.itemgetter(1))[-n:]
    k = list(reversed([i for i, v in top_n]))
    print(f'TOP {n} CLASSES WITH THE HIGHEST MISCLASSIFIED SAMPLES')
    counter = 1
    for u in k:
        for cls_name, index in class_dict.items():
            if index == u:
                print(f'{counter}- {cls_name}')
                counter += 1


def error_analyzer(miss_classified_samples, class_dict):
    """
       Display a grid of images and their corresponding true and predicted labels
       for a sample of miss-classified samples from test dataset.

       Args:
       - miss_classified_samples (list): a list of triplets (img, true_label, predicted_label)
         representing miss-classified samples.
       - class_dict (dict): a dictionary mapping class names (strings) to their corresponding
         integer labels (integers).

       Returns: None
       """

    rows = 6
    cols = 6
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.6, wspace=0.3)

    mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    denormalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    ])

    random_index = random.sample(range(len(miss_classified_samples)), rows * cols)

    invrs_class_dict = {}
    for cls_name in class_dict.keys():
        index = class_dict[cls_name]
        invrs_class_dict[index] = cls_name

    for i, ax in enumerate(axes.flat):
        # show the image
        img, t, p = miss_classified_samples[random_index[i]]
        img = img.squeeze().permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        ax.imshow(denormalize(img).permute(1, 2, 0))
        # set the title as the true and predicted labels
        ax.set_title(f"True: {invrs_class_dict[t]}\nPred: {invrs_class_dict[p.item()]}")
        ax.axis("off")

    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser(description='Evaluate Trained Model on Test Set')
    parser.add_argument('--model', help="model weights", type=Path)

    evaluate(args=parser.parse_args())
