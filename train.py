import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from dataset_prepration import MITIndoorDataset
from models import CNN, SimpleCNN

if __name__ == "__main__":
    # Define the transformations for data augmentation
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    transformations_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = MITIndoorDataset("data/train.txt", transformations)
    val_dataset = MITIndoorDataset("data/val.txt", transformations_test)

    # HYPER-PARAMETERS
    batch_size = 32
    lr = 0.001
    num_epochs = 50
    patience = 10

    best_valid_acc = 0
    epoch_since_improvement = 0

    # CREATE DATA LOADER FOR TRAIN, VALIDATION AND TEST
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    print(len(train_loader))
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    # DEFINE MODEL, OPTIMIZER AND LOSS FUNCTION
    model = SimpleCNN(num_classes=67)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # DEFINE LEARNING RATE SCHEDULER
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, verbose=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # TRAINING LOOP
    for epoch in range(num_epochs):
        # TRAIN THE MODEL
        epoch_iterator_train = tqdm(train_loader)
        train_loss = 0
        train_acc = 0
        for step, batch in enumerate(epoch_iterator_train):
            model.train()
            images, labels = batch[0].to(device).float(), batch[1].to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            epoch_iterator_train.set_postfix(
                batch_loss=(loss.item()), loss=(train_loss / (step + 1))
            )
            train_acc += (outputs.argmax(dim=1) == labels).sum().item()
        train_loss /= len(train_dataset)
        train_acc /= len(train_dataset)

        # VALIDATION THE MODEL
        valid_loss = 0
        valid_acc = 0
        valid_predictions = []
        valid_labels = []
        epoch_iterator_val = tqdm(valid_loader)
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                model.eval()
                images, labels = batch[0].to(device).float(), batch[1].to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                epoch_iterator_val.set_postfix(
                    batch_loss=(loss.item()), loss=(valid_loss / (step + 1))
                )
                valid_acc += (outputs.argmax(dim=1) == labels).sum().item()
                valid_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                valid_labels.extend(labels.cpu().numpy())
        valid_loss /= len(val_dataset)
        valid_acc /= len(val_dataset)
        valid_balanced_acc = balanced_accuracy_score(valid_labels, valid_predictions)

        # Print epoch results
        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.4f}, Valid Balanced Accuracy:'
              f' {valid_balanced_acc:.4f}')

        # Save best model
        if valid_balanced_acc > best_valid_acc:
            best_valid_acc = valid_balanced_acc
            epoch_since_improvement = 0
            torch.save(model.state_dict(), 'model/best_model-2.pt')
            print(f'VALIDATION ACCURACY IMPROVED TO {valid_balanced_acc:.4f}.')
            print('BEST MODEL SAVED')
        else:
            epoch_since_improvement += 1
            print(
                f'VALIDATION ACCURACY DID NOT IMPROVE. EPOCHS SINCE LAST LAST IMPROVEMENT: {epoch_since_improvement}.')

        # Update learning rate scheduler
        scheduler.step(valid_balanced_acc)

        # Check if we should stop training early
        if epoch_since_improvement > patience:
            print(f'VALIDATION ACCURACY DID NOT IMPROVE FOR {patience} EPOCHS. TRAINING STOPPED.')
            break
