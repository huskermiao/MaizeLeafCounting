# -*- coding: UTF-8 -*-

"""
Leaf tip detection using Faster-RCNN
"""

import os
import time
import datetime
import argparse
import torch
from torch.utils.data import DataLoader
import base
import utils
from base import ObjectDetectionDataset, get_model
from engine import train_one_epoch, evaluate

parser = argparse.ArgumentParser(description='Training CNNs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# required arguments
parser.add_argument('training_csv', help='tab separated training csv file')
parser.add_argument('training_img_dir', help='directory where training images reside')
parser.add_argument('valid_csv', help='tab separated csv file for validation')
parser.add_argument('valid_img_dir', help='directory where validation images reside')
parser.add_argument('model_name_prefix', help='the prefix of the output model name ')

# positional arguments
parser.add_argument('--batchsize', default=10, type=int,  help='batch size')
parser.add_argument('--epoch', default=30, type=int, help='number of total epochs')
args = parser.parse_args()

def get_transform(train):
    transforms = []
    transforms.append(base.ToTensor())
    if train:
        transforms.append(base.RandomHorizontalFlip(0.5))
    return base.Compose(transforms)

def train(args):
    train_csv, train_dir, valid_csv, valid_dir, model_name_prefix = args.training_csv, args.training_img_dir, args.valid_csv, args.valid_img_dir, args.model_name_prefix
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('device: %s'%device)
    train_dataset = ObjectDetectionDataset(train_csv, train_dir, get_transform(train=True))
    valid_dataset = ObjectDetectionDataset(valid_csv, valid_dir, get_transform(train=False))
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, collate_fn=utils.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batchsize, collate_fn=utils.collate_fn)
    dataloaders_dict = {'train': train_loader, 'valid': valid_loader}
    model = get_model() # healthy leaf tip, cut leaf tip, background
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad] # 'requires_grad' default is Ture (will be trained)
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3,gamma=0.1)
    print('start training')
    start_time = time.time()
    for epoch in range(args.epoch):
        train_one_epoch(model, optimizer, dataloaders_dict['train'], device, epoch, print_freq=10)
        lr_scheduler.step()
        utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join('.', f'{model_name_prefix}_{epoch}.pth'))
        evaluate(model, dataloaders_dict['valid'], device=device)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    train(args)