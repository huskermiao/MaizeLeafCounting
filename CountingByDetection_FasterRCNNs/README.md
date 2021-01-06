`train.py`: training leaf detection Faster R-CNN models
- run `python train.py -h` for help
- example command: `python train.py example_training.csv example_images example_validation.csv example_images test_fastercnn --batchsize 11 --epoch 10 --patience 2`

`predict.py`: detect leaf tips in testing iamges
- run `python predict.py -h` for help
- example prediction command: `python predict.py test_fastercnn.pt example_testing.csv example_images test_pred --score_cutoff 0.1`

Example training, validation, and testing csv files were provided:
- `example_training.csv`
- `example_validation.csv`
- `example_testing.csv`

The images in the above csv files can be found under `example_images` folder. These datasets are just for testing the code. To achieve best model performance, you should include more training images.