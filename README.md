# mobilenet-v2-custom-dataset
Using Keras MobileNet-v2 model with your custom images dataset

The Keras implementation of MobileNet-v2 (from Keras-Application package) uses by default famous datasets such as imagenet, cifar...

In this package, we use our own custom dataset to train the model. The dataset have the folowing folder structure: each class has its own folder containing .jpg images.

## How to use?

1. Configure the parameters in config.json
2. Train the model using `python train.py`
3. Evaluate the model on test dataset using: `python test.py`
