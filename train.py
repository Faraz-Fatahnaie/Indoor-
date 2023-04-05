import torch
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from dataset import MITIndoorDataset
from models.CNN import CNN, SimpleCNN
from models.ResNet import ResNet18, ResNet50
from models.EfficientNetV2 import efficientnet_v2_s
from timm.models import vision_transformer as vits
from timm.data import Mixup
import os
from argparse import Namespace, ArgumentParser
from pathlib import Path
import json
from config.setting import setting
from utils import cutmix


def setup(args: Namespace):
    i = 1
    flag = True
    SAVE_PATH_ = ''
    TRAINED_MODEL_PATH_ = ''
    CHECKPOINT_PATH_ = ''
    config = {}
    BASE_DIR = Path(__file__).resolve().parent
    while flag:
        if args.model_dir is None:

            config, config_file = setting()

            TEMP_PATH = BASE_DIR.joinpath(
                f"session/{config['MODEL_NAME']}-{i}")
            if os.path.isdir(TEMP_PATH):
                i += 1
            else:
                flag = False

                os.mkdir(BASE_DIR.joinpath(f"session/{config['MODEL_NAME']}-{i}"))
                os.mkdir(BASE_DIR.joinpath(f"session/{config['MODEL_NAME']}-{i}/trained_models"))
                SAVE_PATH_ = BASE_DIR.joinpath(f"session/{config['MODEL_NAME']}-{i}")
                TRAINED_MODEL_PATH_ = BASE_DIR.joinpath(f"session/{config['MODEL_NAME']}-{i}/trained_models")

                os.mkdir(BASE_DIR.joinpath(f'{SAVE_PATH_}/model_checkpoint'))
                CHECKPOINT_PATH_ = SAVE_PATH_.joinpath(f"model_checkpoint/{config['MODEL_NAME']}-{i}.pt")

                with open(f'{SAVE_PATH_}/MODEL_CONFIG.json', 'w') as f:
                    json.dump(config_file, f)

                print(f'MODEL SESSION: {SAVE_PATH_}')
        else:
            flag = False
            SAVE_PATH_ = args.model_dir
            model_dir_name = str(SAVE_PATH_).split(os.sep)[-1]  # .pt file is saving with name of folder name in session
            CHECKPOINT_PATH_ = SAVE_PATH_.joinpath(f'model_checkpoint/{model_dir_name}.pt')
            TRAINED_MODEL_PATH_ = SAVE_PATH_.joinpath('trained_models')

            CONFIGS = open(SAVE_PATH_.joinpath('MODEL_CONFIG.json'))
            CONFIGS = json.load(CONFIGS)
            print('JSON CONFIG FILE LOADED')
            config, _ = setting(CONFIGS)

    transformations = transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'], config['IMAGE_SIZE'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-20, 20)),
        transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.3),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.4843, 0.4240, 0.3648], [0.0558, 0.0536, 0.0518])
    ])

    transformations_test = transforms.Compose([
        transforms.Resize((config['IMAGE_SIZE'], config['IMAGE_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize([0.4843, 0.4240, 0.3648], [0.0558, 0.0536, 0.0518])
    ])

    train_ds = MITIndoorDataset("data/train.txt", transformations)
    val_ds = MITIndoorDataset("data/val.txt", transformations_test)

    # MODEL CONFIGURATION
    model_catalog = {
        # 'efficientNet': efficientnet_b0(in_channels=config['N_CHANNEL'], pretrained=True, num_classes=1),
        # 'efficientNet_C': EfficientNetB0C(in_channels=config['N_CHANNEL'], pretrained=True, num_classes=1),
        # 'Vgg16FCorrelation': Vgg16FCorrelation(num_classes=1),
        'resnet-18': ResNet18(num_classes=67, dropout=0.5),
        'efficientnet_v2_s': efficientnet_v2_s(),
        'ViT': vits.vit_base_patch16_224(pretrained=False, num_classes=67)
    }

    # OPTIMIZER CONFIGURATION
    opt = {
        'Adam': optim.Adam,
        'AdamW': optim.AdamW,
        'RMSprop': optim.RMSprop,
    }

    device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _epoch = 1
    if args.model_dir is None:
        net = model_catalog[config['MODEL_NAME']]
        net.to(device_)
        print('Operation On:', device_)
        # summary(model=net, input_size=(3, config['IMAGE_SIZE'], config['IMAGE_SIZE']))
        with open(f"{SAVE_PATH_}/{config['MODEL_NAME']}_summary.txt", 'a') as f:
            print(net, file=f)

        optimizer_ = opt[config['OPTIMIZER']](net.parameters(), lr=config['LR'])
    else:
        net = model_catalog[config['MODEL_NAME']]
        net_checkpoint = torch.load(Path(CHECKPOINT_PATH_))
        net.load_state_dict(net_checkpoint['model_state_dict'])
        net.to(device_)
        print('Operation On:', device_)

        optimizer_ = opt[config['OPTIMIZER']](net.parameters(), lr=config['LR'])
        optimizer_.load_state_dict(net_checkpoint['optimizer_state_dict'])

        epoch_ = net_checkpoint['epoch']
        epoch_ += 1

    # SCHEDULER CONFIGURATION
    scheduler_opt = {
        'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer_, mode='max', factor=config['FACTOR'],
            patience=config['PATIENCE'], threshold=1e-3,
            min_lr=config['MIN_LR'], verbose=True),
        'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_, gamma=config['FACTOR']),
        'lr_scheduler': lr_scheduler.MultiStepLR(optimizer=optimizer_, milestones=[5], gamma=5e-5, last_epoch=-1)
    }
    scheduler_ = scheduler_opt[config['SCHEDULER']]

    # LOSS FUNCTION CONFIGURATION
    criterion_dict = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(),
        'MSELoss': nn.MSELoss()
    }
    criterion_ = criterion_dict[config['LOSS_FUNCTION']]

    return net, train_ds, val_ds, optimizer_, scheduler_, criterion_, device_, SAVE_PATH_, \
           TRAINED_MODEL_PATH_, CHECKPOINT_PATH_, config


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_dir', help='path to model in session folder in order to resume training using '
                                            'checkpoint files', type=Path, required=False)

    # CREATE SESSION AND CONFIGURE FOR TRAINING
    model, train_dataset, val_dataset, optimizer, scheduler, criterion, device, SAVE_PATH, TRAINED_MODEL_PATH, \
    CHECKPOINT_PATH, config = setup(args=parser.parse_args())

    writer = SummaryWriter(log_dir=f'{SAVE_PATH}')

    best_valid_acc = 0
    epoch_since_improvement = 0

    # CREATE DATA LOADER FOR TRAIN, VALIDATION AND TEST
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True,
                              num_workers=config['NUM_WORKER'], drop_last=True, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False,
                            num_workers=config['NUM_WORKER'], pin_memory=True)

    mix_up = Mixup(mixup_alpha=.2, num_classes=67)

    # TRAINING LOOP
    print('START TRAINING ...')
    for epoch in range(config['EPOCHS']):
        # TRAIN THE MODEL
        epoch_iterator_train = tqdm(train_loader)
        train_loss = 0
        train_acc = 0
        for step, batch in enumerate(epoch_iterator_train):
            model.train()
            images, labels = batch[0], batch[1]

            if config['augmentation'] == 1:  # MIXUP
                if len(images) % 2 != 0:
                    images = images[:len(images) - 1, :, :, :]
                    labels = labels[:len(labels) - 1]

                images, labels = mix_up(images, labels)

            if config['augmentation'] == 2:  # CUTMIX
                inputs, labels = cutmix(images, labels, alpha=1.0)

            images, labels = images.to(device).float(), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)

            if config['augmentation'] == 1:  # if mixup is used
                loss = criterion(outputs, labels.argmax(1))
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            epoch_iterator_train.set_postfix(
                batch_loss=(loss.item()), loss=(train_loss / (step + 1))
            )
            if config['augmentation'] == 1:
                # TODO: this should be fixed
                train_acc += (outputs == labels).sum().item()
            else:
                train_acc += (outputs.argmax(dim=1) == labels).sum().item()
        train_loss /= len(train_dataset)
        train_acc /= len(train_dataset)

        # VALIDATION THE MODEL
        val_loss = 0
        val_acc = 0
        val_predictions = []
        val_labels = []
        epoch_iterator_val = tqdm(val_loader)
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                model.eval()
                images, labels = batch[0].to(device).float(), batch[1].to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                epoch_iterator_val.set_postfix(
                    batch_loss=(loss.item()), loss=(val_loss / (step + 1))
                )
                val_acc += (outputs.argmax(dim=1) == labels).sum().item()
                val_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_dataset)
        val_acc /= len(val_dataset)
        valid_balanced_acc = balanced_accuracy_score(val_labels, val_predictions)

        # Print epoch results
        print(f"Epoch {epoch + 1}/{config['EPOCHS']}:")
        print(f"LR: {scheduler.optimizer.param_groups[0]['lr']}")
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Valid Loss: {val_loss:.4f}, Valid Accuracy: {val_acc:.4f}, Valid Balanced Accuracy:'
              f' {valid_balanced_acc:.4f}')

        # Save best trained_model
        if valid_balanced_acc > best_valid_acc:
            best_valid_acc = valid_balanced_acc
            epoch_since_improvement = 0
            torch.save(model, TRAINED_MODEL_PATH.joinpath(f'VAL_BALANCED_ACC-{valid_balanced_acc:.4f}-'
                                                          f'EPOCH-{epoch + 1}.pth'))
            print('BEST MODEL SAVED')
            print(f'VALIDATION ACCURACY IMPROVED TO {valid_balanced_acc:.4f}.')

        else:
            epoch_since_improvement += 1
            # Check if we should stop training early
            if epoch_since_improvement >= config['EARLY_STOP']:
                early_stop = config['EARLY_STOP']
                print(f'VALIDATION ACCURACY DID NOT IMPROVE FOR {early_stop} EPOCHS. TRAINING STOPPED.')
                break
            print(
                f'VALIDATION ACCURACY DID NOT IMPROVE. EPOCHS SINCE LAST LAST IMPROVEMENT: {epoch_since_improvement}.')

        # Update learning rate scheduler
        scheduler.step(valid_balanced_acc)

        # LOG TENSORBOARD
        writer.add_scalar('/Loss_train', train_loss, epoch)
        writer.add_scalar('/Loss_validation', val_loss, epoch)

        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, CHECKPOINT_PATH)
        except Exception as e:
            print('MODEL DID NOT SAVE!')
