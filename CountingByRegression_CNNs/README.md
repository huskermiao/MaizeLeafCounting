`train.py`: training CNNs
- run `python train.py -h` for help
- an example training command: `python train.py example_training.csv example_images test_cnn --valid_csv example_validation.csv --valid_img_dir example_images --epoch 100 --patience 3`

`predict.py`: make predictions on new images using a trained CNN
- run `python predict.py -h` for help
- an example prediction command: `python predict.py test_cnn.pt example_testing.csv example_images test_prediction.csv`

Example training, validation, and testing csv files were provided:
- `example_training.csv`
- `example_validation.csv`
- `example_testing.csv`

The images in the above csv files can be found under `example_images` folder. These datasets are just for testing the code. To achieve best model performance, you should include more training images.