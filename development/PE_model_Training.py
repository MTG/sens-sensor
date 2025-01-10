"""
This script trains and saves the best performing models for each feature set. 

train_RFR() and train_EN() are the main function, each for one algorithm type.
Best performing models are saved when you run code.
Uncomment code at the bottom in order to train and save models according to the 
specified input configuration.
"""

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
from development.lib.models_functions import train_RFR
from development.lib.auxiliars import (
    clap_features,
)


def main(data_path):

    dataset_path = os.path.join(
        data_path, "files/ARAUS_CLAP_dataset/ARAUS_CLAP_dataset.csv"
    )
    data_foldFs_path = os.path.join(
        data_path, "files/fold_Fs_CLAP_dataset/fold_Fs_CLAP_dataset.csv"
    )
    saving_folder = os.path.join(data_path, "models/models_delete_noPCA")

    ############# RUN ###################################################################
    # MODEL FOR PLEASANTNESS PREDICTION
    input_dict = {
        "train_dataset_path": dataset_path,
        "test_dataset_path": data_foldFs_path,
        "features": clap_features,
        "predict": "P",
        "params": [30],
        "saving_folder_path": saving_folder,
        "model_name": "model_pleasantness",
    }
    # train_RFR(input_dict)
    # MODEL FOR EVENTFULNESS PREDICTION
    input_dict = {
        "train_dataset_path": dataset_path,
        "test_dataset_path": data_foldFs_path,
        "features": clap_features,
        "predict": "E",
        "params": [50],
        "saving_folder_path": saving_folder,
        "model_name": "model_eventfulness",
    }
    train_RFR(input_dict)
    #####################################################################################


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
# python development/PE_model_Training.py --data_path data
