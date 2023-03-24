import os
import random
from pathlib import Path
import pickle


def data_prepration(root_dir: str):
    """
    This Function is for Processing MIT Indoor Scene Recognition and Should Run Once.
    :param root_dir: Path to Root Folder of Images which Contains 67 Classes
    :return: None (Two .txt File will Create Contained Image Path and Corresponding Label. Moreover, a Dictionary will
    Create which the Keys are Class Name and the Values are an Integer Related to that Class Name)
    """

    class_dict = {}
    base_dir = Path.cwd()
    train_images = []
    val_images = []

    # Create class dictionary
    for i, d in enumerate(os.listdir(root_dir)):
        if os.path.isdir(os.path.join(root_dir, d)):
            class_dict[d] = i

    # Create a list of image paths and labels
    for class_name in class_dict.keys():
        class_path = Path(os.path.join(root_dir, class_name))
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

    # Write the data to text files
    with open("data/train.txt", "w") as f:
        for c in train_images:
            image_path, label = c
            f.write(f"{image_path} {label}\n")

    with open("data/val.txt", "w") as f:
        for c in val_images:
            image_path, label = c
            f.write(f"{image_path} {label}\n")

    with open('data/class_dict.pkl', 'wb') as f:
        pickle.dump(class_dict, f)


if __name__ == "__main__":
    data_prepration(root_dir=".\data\Images")
