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
import utils
from torch.utils.data import DataLoader
from base import ObjectDetectionDataset, get_model, show_box, get_transform
from engine import train_one_epoch, evaluate, idx_cleanboxes

parser = argparse.ArgumentParser(description='Training CNNs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# required arguments
parser.add_argument('model_fn', help='specify the filename of the trained model')
parser.add_argument('testing_csv', help='tab separated csv file for testing')
parser.add_argument('testing_img_dir', help='directory where testing images reside')
parser.add_argument('output_prefix', help='the prefix of output files')

# positional arguments
parser.add_argument('--score_cutoff', type=float, default=0.5, help='set score cutoff')
parser.add_argument('--second_cutoff', type=float, default=0.83,
                help='cutoff for solving overlapped bounding boxes')
args = parser.parse_args()

def predict(args):
    saved_model, test_csv, test_dir, output_prefix = args.model_fn, args.testing_csv, args.testing_img_dir, args.output_prefix
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device detected: {device}')
    model = get_model(num_classes=3)
    model.load_state_dict(torch.load(saved_model, map_location=device))
    model.eval()
    test_dataset = ObjectDetectionDataset(test_csv, test_dir, get_transform(train=False), only_image=True, sep=',')
    test_loader = DataLoader(test_dataset, batch_size=1)

    filenames, lcs = [], [] # lcs: leaf counts
    print('start prediction...')
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
        boxes, labels, scores = boxes[idxs], labels[idxs], scores[idxs]

        # post-process boxes to remvoe redundancy
        final_idx = idx_cleanboxes(boxes, scores, second_cutoff=args.second_cutoff)
        boxes, labels, scores = boxes[final_idx], labels[final_idx], scores[final_idx]
        
        print('saving box coordinates, label, and score...')
        npy_prefix = '.'.join(fn.split('.')[0:-1])
        df = pd.DataFrame(boxes)
        df.columns = ['x0', 'y0', 'x1', 'y1']
        df['label'] = labels
        df['score'] = scores
        df.to_csv(npy_prefix+'.info.csv', index=False) 
        print('adding predicted box on the original images...')
        img = show_box(Path(test_dir)/fn, boxes, labels, scores)
        img_out_fn = fn.replace('.png', '.prd.jpg') if fn.endswith('.png') else fn.replace('.jpg', '.prd.jpg')
        img.save(img_out_fn)

        filenames.append(fn)
        lcs.append(len(final_idx))
    pd.DataFrame(dict(zip(['fn', 'lc'], [filenames, lcs]))).to_csv(output_prefix+'.prediction.csv', index=False)
    print('Done! check leaf counting results in %s.prediction.csv'%output_prefix)

if __name__ == "__main__":
    predict(args)