# -*- coding: UTF-8 -*-

"""
Detect leaf tips using a trained Faster-RCNN
"""

import os
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import base
from base import ObjectDetectionDataset, get_model, show_box
import utils
from engine import train_one_epoch, evaluate

parser = argparse.ArgumentParser(description='Training CNNs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# required arguments
parser.add_argument('model_fn', help='specify the filename of the trained model')
parser.add_argument('testing_csv', help='tab separated csv file for testing')
parser.add_argument('testing_img_dir', help='directory where testing images reside')
parser.add_argument('output_prefix', help='the prefix of output files')

# positional arguments
parser.add_argument('--score_cutoff', type=float, default=0.9, help='set score cutoff')
args = parser.parse_args()

def get_transform(train):
    transforms = []
    transforms.append(base.ToTensor())
    if train:
        transforms.append(base.RandomHorizontalFlip(0.5))
    return base.Compose(transforms)

def predict(args):
    saved_model, test_csv, test_dir, output = args.model_fn, args.testing_csv, args.testing_img_dir, args.output_prefix
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device detected: {device}')
    model = get_model(num_classes=3)
    checkpoint = torch.load(saved_model, map_location={'cuda:0': 'cpu'})
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    test_dataset = ObjectDetectionDataset(test_csv, test_dir, get_transform(train=False), only_image=True)
    test_loader = DataLoader(test_dataset, batch_size=1)

    filenames, boxes, labels, scores, lcs = [], [], [], [], []
    for imgs, _, fns in test_loader:
        imgs = imgs.to(device)
        results = model(imgs)
        fn = fns[0]
        print(fn)
        boxes, labels, scores = results[0]['boxes'], results[0]['labels'], results[0]['scores']
        boxes = np.array([i.to(device).tolist() for i in boxes])
        labels = np.array([i.to(device).tolist() for i in labels])
        scores = np.array([i.to(device).tolist() for i in scores])
        idxs = np.argwhere(scores>args.score_cutoff).squeeze()

        img = show_box(Path(test_dir)/fn, boxes[idxs], labels[idxs], scores[idxs])
        img_out_fn = fn.replace('.png', '.prd.jpg') if fn.endswith('.png') else fn.replace('.jpg', '.prd.jpg')
        out_dir = Path(output)
        if not out_dir.exists():
            out_dir.mkdir()
        img.save(out_dir/img_out_fn)
        filenames.append(fn)
        lcs.append(len(idxs))
    pd.DataFrame(dict(zip(['fn', 'lc'], [filenames, lcs]))).to_csv('%s.pred.csv'%output, index=False)
    print('Done! Check %s.pred.csv and images under %s folder...'%(output, output))


if __name__ == "__main__":
    predict(args)