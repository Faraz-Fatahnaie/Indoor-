import pickle
import matplotlib.pyplot as plt
from collections import Counter
import operator
import random
import torch
import torchvision.transforms as transforms
from argparse import Namespace, ArgumentParser
from pathlib import Path


def top_n_error_class(miss_classified_samples, class_dict, n=10):
    """
        Given a list of misclassified samples and a dictionary mapping class,
        returns the names of the top n classes with the highest misclassified samples.

        Args:
        - miss_classified_samples (list): a list of triplets (img, true_label, predicted_label)
            representing miss-classified samples.
        - class_dict (dict): A dictionary mapping class names to integer indices.
        - n (int): The number of top classes to return.

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
    cols = 10
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
        img, t, p = miss_classified_samples[random_index[i]]
        img = img.squeeze().permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        ax.imshow(denormalize(img).permute(1, 2, 0))
        ax.set_title(f"True: {invrs_class_dict[t]}\nPred: {invrs_class_dict[p.item()]}")
        ax.axis("off")
    plt.show()


def load_files(args: Namespace):
    class_dict_path: Path = args.mcso

    with open(f"{class_dict_path}", "rb") as fp:
        miss_classified_samples = pickle.load(fp)

    with open('data/class_dict.pkl', 'rb') as f:
        class_dict = pickle.load(f)

    return miss_classified_samples, class_dict


if __name__ == "__main__":
    parser = ArgumentParser(description='Evaluate Trained Model on Test Set')
    parser.add_argument('--mcso', help="miss-classified samples object created in evaluate.py", type=Path)

    mcso, cls_dict = load_files(args=parser.parse_args())

    top_n_error_class(mcso, cls_dict)
    error_analyzer(mcso, cls_dict)