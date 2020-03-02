# CS 182 Vision Project - Generalizable Classifiers

UC Berkeley, Spring 2020. The goal is to design and train a classifier on top of the Tiny ImageNet dataset, a classification dataset with 200 classes, each with 500 training images (100,000 images) that are 64x64 RGB images. The validation dataset has 50 images per class (10,000 images).

## First time setup:

Run ```get_data.sh```. This will download the Tiny ImageNet dataset to your local machine. If needed, run ```pip install -r requirements.txt``` to install the required modules. You may need to add a ```--user``` option. Then run ```python -i tools.py``` and call ```prepare_data()```. This will process the images into a format which can be fed into Keras models.

## Training:

To design a new model, create a new function in ```models.py``` that returns a Keras model. To train a model, simply run ```python -i train.py``` and call ```train()```, passing in arguments as appropriate. It is recommended to create a Jupyter Notebook environment to train models and run visualizations.
