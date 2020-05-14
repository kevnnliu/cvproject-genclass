import sys
import pathlib
from keras.models import load_model
import numpy as np
from models import AlphaStack
from tools import get_label_dict, read_resize
from train import top3_acc, top5_acc


def main():
    # Get labels
    labels = get_label_dict()

    # Load the model
    custom_metrics = {"top3_accuracy": top3_acc, "top5_accuracy": top5_acc}
    model_path = "models/AlphaStack/AlphaStack.h5"
    ensemble = load_model(model_path, custom_objects=custom_metrics)

    # Open the evaluation CSV file for writing
    with open("eval_classified.csv", "w") as eval_output_file:
        # Open the input CSV file for reading
        for line in pathlib.Path(sys.argv[1]).open():
            # Extract CSV info
            image_id, image_path, image_height, image_width, image_channels = line.strip(
            ).split(",")

            print(image_id, image_path, image_height, image_width,
                  image_channels)

            # Preprocess our data
            img = read_resize(image_path)

            # Generate a prediction
            output = ensemble.predict(np.expand_dims(img, 0))
            prediction = int(np.argmax(output.reshape(-1)))

            # Write the prediction to the output file
            eval_output_file.write('{},{}\n'.format(image_id,
                                                    labels[prediction]))


if __name__ == "__main__":
    main()
