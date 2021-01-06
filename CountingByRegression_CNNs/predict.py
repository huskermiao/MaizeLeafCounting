"""
make predictions on testing images using trained model
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from base import image_transforms, initialize_model, LeafcountingDataset

parser = argparse.ArgumentParser(description='Training CNNs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# required arguments
parser.add_argument('model_fn', help='specify the filename of the trianed model')
parser.add_argument('testing_csv', help='testing csv file')
parser.add_argument('testing_img_dir', help='directory where testing images reside')
parser.add_argument('output_fn', help='filename of prediction results')

# positional arguments
parser.add_argument('--inputsize', default=224, type=int, help='the input size of image for the trained model')
parser.add_argument('--batchsize', default=60, type=int,  help='batch size')

args = parser.parse_args()

def prediction(args):
    saved_model, test_csv, test_dir, output = args.model_fn, args.testing_csv, args.testing_img_dir, args.output_fn
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('devicd: %s'%device)
    model, input_size = initialize_model(feature_extract=True, 
                                        use_pretrained=False, 
                                        inputsize=args.inputsize)
    for param in model.parameters():
        param.requires_grad = False
    model.load_state_dict(torch.load(saved_model, map_location=device))
    model.eval()
    test_dataset = LeafcountingDataset(test_csv, test_dir, image_transforms(input_size=args.inputsize)['valid'])
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize)

    ground_truths, predicts, filenames = [],[],[]
    for idx, (inputs, labels, fns) in enumerate(test_loader, 1): # fns is a tuple
        inputs = inputs.to(device)
        outputs = model(inputs)
        ground_truths.append(labels.squeeze().numpy())
        filenames.append(np.array(fns))
        if torch.cuda.is_available():
            predicts.append(outputs.squeeze().to('cpu').numpy())
        else:
            predicts.append(outputs.squeeze().numpy())
    ground_truths = np.concatenate(ground_truths)
    predicts = np.concatenate(predicts)
    filenames = np.concatenate(filenames)
    df = pd.DataFrame(dict(zip(['fn', 'groundtruth', 'prediction'], [filenames, ground_truths, predicts])))
    df.to_csv(output, index=False)
    print('Prediction complete, check %s...'%output)

if __name__ == "__main__":
    prediction(args)