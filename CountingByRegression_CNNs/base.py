import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, models

class LeafcountingDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the comma separated csv file without header. The 1st column is image file name and the 2nd column is the annotation/label. 
            root_dir (string): Directory with all the images.
        """
        self.csv = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.csv.iloc[idx, 0]
        image = Image.open(self.root_dir/img_name)
        if len(image.getbands()) == 4:
            image = image.convert('RGB')
        label = self.csv.iloc[idx, 1].astype('float32').reshape(-1,)

        if self.transform:
            image = self.transform(image)

        return image, label, img_name

def train_model_regression(model, dataloaders, criterion, optimizer, model_name_prefix, 
                           patience=10, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loss_history, valid_loss_history = [], []
    
    early_stopping = EarlyStopping(model_name_prefix, patience=patience, verbose=True, min_loss_cutoff=0.8)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        model.train()
        train_running_loss = 0.0
        for inputs, labels, _ in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_running_loss += loss.item() * inputs.size(0)
        epoch_loss_train = train_running_loss / len(dataloaders['train'].dataset)
        print('Train Loss: {:.4f}'.format(epoch_loss_train))
        train_loss_history.append(epoch_loss_train)

        if 'valid' in dataloaders: # if validation data is available
            model.eval()
            valid_running_loss = 0.0
            for inputs, labels, _ in dataloaders['valid']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_running_loss += loss.item() * inputs.size(0)
            epoch_loss_valid = valid_running_loss / len(dataloaders['valid'].dataset)
            print('Validation Loss: {:.4f}'.format(epoch_loss_valid))
            valid_loss_history.append(epoch_loss_valid)
            early_stopping(epoch_loss_valid, model)
        else:
            early_stopping(epoch_loss_train, model)

        if early_stopping.early_stop:
            print('Early stopping')
            break
    print('Best val loss: {:4f}'.format(early_stopping.val_loss_min))
    model.load_state_dict(torch.load('%s.pt'%model_name_prefix))
    return model, train_loss_history, valid_loss_history

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name='resnet18', num_classes=1, feature_extract=True, use_pretrained=True, inputsize=224):
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = inputsize
    else:
        print("Invalid model name, exiting...")
        exit()
    return model_ft, input_size

def image_transforms(input_size=224):
    
    image_transforms_dict = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            #transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])# Imagenet standards  
            ]),
    
        # Validation does not use augmentation
        'valid':
        transforms.Compose([
            transforms.Resize(size=(input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return image_transforms_dict

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, mn_prefix, patience=20, verbose=True, delta=0, min_loss_cutoff=0):
        """
        Args:
            mn_prefix (str): the prefix of the saved model name.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            min_loss_cutoff (float): set the minimum loss cutoff if there is no validation process during training.
                For leaf counting problem with MSE as the loss, set 0.81 consdering human error is 0.5.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.mn_prefix = mn_prefix
        self.min_loss_cutoff = -min_loss_cutoff

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None: # the first epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            if self.best_score > self.min_loss_cutoff: # the model performs better than human being
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), '%s.pt'%self.mn_prefix)
        self.val_loss_min = val_loss