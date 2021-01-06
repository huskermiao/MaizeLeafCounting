import sys
import json
import torch
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

font_dir = Path(__file__).parent.absolute()
c_dict = {1:'#e41a1c', 2:'#377eb8', 3:'#984ea3', 4:'#ff7f00', 5:'#f781bf'}

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

def show_box(img_fn, boxes, labels, scores):
    
    import matplotlib.pyplot as plt

    original_image = Image.open(img_fn).convert('RGB')
    annotated_image = original_image
    draw = ImageDraw.Draw(original_image)
    font = ImageFont.truetype(str(font_dir/"calibril.ttf"), 10)
    
    for box, label, score in zip(boxes, labels, scores):
        #print(box, label, score)
        if not isinstance(box, list):
            box = list(box)
        color = c_dict[label]
        draw.rectangle(xy=box, outline=color) 
        draw.rectangle(xy=[l + 1. for l in box], outline=color)  # a second rectangle at an offset of 1 pixel to increase line thickness

        text = '[%s] %.2f'%(label, score)
        text_size = font.getsize(text.upper())
        text_location = [box[0] + 2., box[1] - text_size[1]]
        textbox_location = [box[0], box[1] - text_size[1], box[0] + text_size[0] + 4., box[1]]
        draw.rectangle(xy=textbox_location, fill=color)
        draw.text(xy=text_location, text=text.upper(), fill='white', font=font)
    del draw
    return annotated_image

def get_model(num_classes=3):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class ObjectDetectionDataset(Dataset):
    def __init__(self, csv_fn, root_dir, transforms, sep='\t', only_image=False):
        '''
        csv_fn: tab separated csv file with:
            1st column('fn'): image file name
            2st column('targets): target information including x, y, and label in json format
        root_dir:
            where the images in csv file are located
        '''
        self.csv_df = pd.read_csv(csv_fn, sep=sep)
        if only_image:
            if 'fn' not in self.csv_df.columns:
                sys.exit("Couldn't find 'fn' in the csv header.")
        else:
            if 'fn' not in self.csv_df.columns or 'targets' not in self.csv_df.columns:
                sys.exit("Couldn't find 'fn' and 'targets' in the csv header.")

        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.only_image = only_image

    def __len__(self):
        return len(self.csv_df)

    def __getitem__(self, idx):
        #print(f'idx: {idx}')
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_fn = self.csv_df.loc[idx, 'fn']
        #print(f'img_fn: {img_fn}')
        img = Image.open(self.root_dir/img_fn)
        target = {}
        if len(img.getbands()) == 4:
            img = img.convert('RGB')
        
        if self.only_image:
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target, img_fn

        # map each label to a class with name started from 1
        tips = json.loads(self.csv_df.loc[idx, 'targets'])
        #print(tips)
        label_dict = {'intact':1, 'cut':2}

        boxes, labels = [], []
        for x, y, label in zip(tips['x'], tips['y'], tips['label']):
            boxes.append([x-15, y-15, x+15, y+15]) # square with side length==30
            labels.append(label_dict[label])
        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels)
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([idx])
        # zeros: False, ones: True (will not be used in evaluation)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        #print(boxes.size(), labels.size(), image_id.size(), areas.size(), iscrowd.size())

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target , img_fn

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, mn_prefix, patience=20, verbose=True, delta=0):
        """
        Args:
            mn_prefix (str): the prefix of the saved model name.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.mn_prefix = mn_prefix

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

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), '%s.pt'%self.mn_prefix)
        self.val_loss_min = val_loss