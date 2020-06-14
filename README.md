## CS 182: Deep Neural Networks, Vision Project - Generalizable Classifiers

UC Berkeley, Spring 2020. The goal was to design and train a classifier on top of the Tiny ImageNet dataset, a classification dataset with 200 classes, each with 500 training images (100,000 images) that are 64x64 RGB images. The validation set has 50 images per class (10,000 images). This project primarily uses [Keras](https://keras.io/) and [Tensorflow](https://www.tensorflow.org/).

## First time setup:

Run `data/get_data.sh`. This will download the Tiny ImageNet dataset to your local machine. If needed, run ```pip install -r requirements.txt``` to install the required modules. You may need to add a `--user` option. It is recommended to run this in a virtual environment. Then run `python process_data.py`. This will process the images into a format which can be fed into Keras models. To enable GPU, make sure you have CUDA 10.1 and the right cuDNN libraries installed.

## Training:

To design a new model, create a new function in `models.py` that returns a Keras model. To train a model, simply call `train()` from `train.py`, passing in arguments as appropriate. Note that you can pass in either a default Keras optimizer name or a Keras optimizer instance to the `optim` argument. It is recommended to use Jupyter Notebook to train models and run visualizations. `Training.ipynb` is a good example of this.

## Loading a model:

To load a model from the models folder, simply import `load_model` from `keras.models` and call `load_model(model_path, custom_objects)`. `model_path` should be a relative path to the `.h5` file of the model you want to load. `custom_objects` should be a dictionary of custom metrics like top-k accuracy. This function returns a Keras model.

## Saliency:

To view saliency maps for your model, simply use `Saliency.ipynb` and load your model in the appropriate cell.

## Data augmentation:

To see the results of applying different data augmentations, use `Augmentation.ipynb` and change the arguments of the `ImageDataGenerator` object as desired.

## Testing

`test_submission.py` takes the path to a CSV file as a command line argument. Each line in the CSV file should be in the format `image_id (int), image_path (str), image_height (int), image_width (int), image_channels (int)`. This will create a file called `eval_classified.csv` where each line is in the format `image_id (int), image_class (str)`.
