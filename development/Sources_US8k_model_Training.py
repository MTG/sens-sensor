import json
import pandas as pd
import numpy as np
import os
import sys
from joblib import dump
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
)
import argparse

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current dir ", current_dir)
src_dir = os.path.abspath(os.path.join(current_dir, "../"))
print("src dir ", src_dir)
sys.path.append(src_dir)

# Imports from this project
from development.lib.models_functions import train_US8k_models


def main(data_path):
    # RUN ########################################################################
    dataset_path = os.path.join(
        data_path, "files/UrbanSound8K_CLAP_dataset/UrbanSound8K_CLAP_dataset.json"
    )  # data/files/IDMT-US8k_CLAP_dataset/IDMT-US8k_CLAP_dataset.json # TO INCLUDE IDMT VEHIcLES
    algorithms = {
        "air_conditioner": "linear",
        "car_horn": "linear",
        "children_playing": "linear",
        "dog_bark": "linear",
        "drilling": "linear",
        "engine_idling": "linear",
        "gun_shot": "linear",
        "jackhammer": "linear",
        "siren": "linear",
        "street_music": "linear",
        "construction": "linear",
    }  # "vehicles_IDMT": "linear", # TO INCLUDE IDMT VEHIcLES
    saving_path = os.path.join(
        data_path, "models/sources_US8k"
    )  # models/sources_US8k-IDMT # TO INCLUDE IDMT VEHIcLES

    train_US8k_models(dataset_path, algorithms, saving_path)
    ##############################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train indicated models for the prediction of the specified sound sources. Saves models."
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
# python development/Sources_US8k_model_Training.py --data_path data
