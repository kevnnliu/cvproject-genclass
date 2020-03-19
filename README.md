# CS 182 Vision Project - Generalizable Classifiers

UC Berkeley, Spring 2020. The goal is to design and train a classifier on top of the Tiny ImageNet dataset, a classification dataset with 200 classes, each with 500 training images (100,000 images) that are 64x64 RGB images. The validation dataset has 50 images per class (10,000 images).

## First time setup:

Run ```get_data.sh```. This will download the Tiny ImageNet dataset to your local machine. If needed, run ```pip install -r requirements.txt``` to install the required modules. You may need to add a ```--user``` option. Then run ```python process_data.py```. This will process the images into a format which can be fed into Keras models.

## Training:

To design a new model, create a new function in ```models.py``` that returns a Keras model. To train a model, simply call ```train()``` from ```train.py```, passing in arguments as appropriate. Note that you can pass in either a default Keras optimizer name or a Keras optimizer instance to the ```optim``` argument. It is recommended to use Jupyter Notebook to train models and run visualizations.
