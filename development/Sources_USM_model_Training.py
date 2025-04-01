import os
import sys
import argparse

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current dir ", current_dir)
src_dir = os.path.abspath(os.path.join(current_dir, "../"))
print("src dir ", src_dir)
sys.path.append(src_dir)

# Imports from this project
from development.lib.models_functions import train_USM_models


def main(data_path):
    # RUN ########################################################################
    dataset_path = os.path.join(
        data_path, "files/USM_CLAP_dataset/USM_CLAP_dataset.json"
    )  # path to json file containing USM-extended
    algorithms ={
        "birds": "linear",
        "construction": "linear",
        "dogs": "linear",
        "human": "linear",
        "music": "l_r",
        "nature": "linear",
        "siren": "linear",
        "vehicles": "l_r",
    }  # list of best performing algorithms for each source_USM (same order as sources list)
    saving_path = os.path.join(
        data_path, "models/sources_USM"
    )  # path to folder where to save the models

    train_USM_models(dataset_path, algorithms, saving_path)
    ##############################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Random Forest Regression models for the prediction of Pleasantness and Eventfulness with determined configuration. Saves models."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data folder",
    )
    # Parse arguments
    args = parser.parse_args()
    data_path = args.data_path

    # Call main function
    main(data_path)

# Run command  example (where sens-sensor/data is where the data is found):
# python development/Sources_USM_model_Training.py --data_path data
