# -*- coding: UTF-8 -*-

"""
Leaf tip detection using Faster-RCNNs
"""

import os
import time
import datetime
import argparse
import torch
import utils
from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate
from base import ObjectDetectionDataset, get_model, get_transform, EarlyStopping


parser = argparse.ArgumentParser(description='Training Faster R-CNNs to detect leaf tips', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# required arguments
parser.add_argument('training_csv', help='tab separated training csv file')
parser.add_argument('training_img_dir', help='directory where training images reside')
parser.add_argument('valid_csv', help='tab separated csv file for validation')
parser.add_argument('valid_img_dir', help='directory where validation images reside')
parser.add_argument('model_name_prefix', help='the prefix of the output model name')

# positional arguments
parser.add_argument('--batchsize', default=10, type=int,  help='batch size')
parser.add_argument('--epoch', default=30, type=int, help='number of total epochs')
parser.add_argument('--patience', default=4, type=int, help='patience in early stopping')
args = parser.parse_args()

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
    print('start training...')
    start_time = time.time()

    early_stopping = EarlyStopping(model_name_prefix, patience=args.patience, verbose=True, delta=0.01)
    for epoch in range(args.epoch):
        loss_dict, total_loss = train_one_epoch(model, optimizer, dataloaders_dict['train'], device, epoch, print_freq=1)
        print('sum of losses: %s'%total_loss)
        print('loss dict:\n', loss_dict)
        lr_scheduler.step()
        coco_evaluator = evaluate(model, dataloaders_dict['valid'], device=device)
        ap = coco_evaluator.coco_eval['bbox'].stats[0]
        #ious = coco_evaluator.coco_eval['bbox'].ious
        print('average AP: %s'%ap)
        valid_loss = 1-ap
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print('Early stopping')
            break
        print('Best average precision: {:0.3f}'.format(1-early_stopping.val_loss_min))
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    train(args)