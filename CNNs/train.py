"""
train CNNs to estimate leaf numbers
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from base import EarlyStopping, image_transforms, initialize_model, train_model_regression, LeafcountingDataset

parser = argparse.ArgumentParser(description='Training CNNs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# required arguments
parser.add_argument('training_csv', help='training csv file')
parser.add_argument('training_img_dir', help='directory where training images reside')
parser.add_argument('model_name_prefix', help='the prefix of the output model name ')

# positional arguments
parser.add_argument('--valid_csv', help='csv file for validation if available')
parser.add_argument('--valid_img_dir', help='directory where validation images reside')
parser.add_argument('--inputsize', default=224, type=int, help='the input size of image. At least 224 if using pretrained models')
parser.add_argument('--batchsize', default=60, type=int,  help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='number of total epochs')
parser.add_argument('--patience', default=30, type=int, help='patience in early stopping')
parser.add_argument('--learning_type', default='finetuning', choices=['feature_extractor', 'finetuning'], help='transfer learning type.')
parser.add_argument('--pretrained_mn', help='specify your own pretrained model as feature extractor')
                    
args = parser.parse_args()

def train(args):
    train_csv, train_dir, model_name_prefix = args.training_csv, args.training_img_dir, args.model_name_prefix
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = LeafcountingDataset(train_csv, train_dir, image_transforms(input_size=args.inputsize)['train'])
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize)
    dataloaders_dict = {'train': train_loader}
    if args.valid_csv and args.valid_img_dir:
        valid_dataset = LeafcountingDataset(args.valid_csv, args.valid_img_dir, image_transforms(input_size=args.inputsize)['valid'])
        valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize)
        dataloaders_dict['valid'] = valid_loader 
    feature_extract = True if args.learning_type=='feature_extractor' else False
    
    if args.pretrained_mn:
        model, input_size = initialize_model(feature_extract=True,
                                         use_pretrained=False,
                                         inputsize=args.inputsize)
        model.load_state_dict(torch.load(args.pretrained_mn, map_location=device))
    else:
        model, input_size = initialize_model(feature_extract=feature_extract, 
                                         inputsize=args.inputsize)
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    sgd_optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9) # optimizer
    criterion = nn.MSELoss() # loss
    since = time.time()
    model_ft, train_hist, valid_hist = train_model_regression(model, dataloaders_dict, 
                                                            criterion, sgd_optimizer,
                                                            model_name_prefix, 
                                                            patience=args.patience, 
                                                            num_epochs=args.epoch)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":
    train(args)
