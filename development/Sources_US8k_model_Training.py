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
    )  # path to json file containing USM-extended
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
    }  # list of best performing algorithms for each source_USM (same order as sources list)
    saving_path = os.path.join(
        data_path, "models/sources_US8k"
    )  # path to folder where to save the models

    train_US8k_models(dataset_path, algorithms, saving_path)
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

""" # Inputs
data_path = "data/files/UrbanSound8K_embeddings.json"
saving_path = "data/models/KNN_classification_model_complete.joblib"

############################## PREPARE DATA ##################################
# Import data from json
with open(data_path, "r") as f:
    data = json.load(f)
    # Convert JSON data back to DataFrame
    df = pd.DataFrame(data)

##############################################################################


def KNN_model_save(df: pd.DataFrame, n_neighbors: int, saving_path: str):

    # Create model
    n_neighbors = n_neighbors
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Get ground-truth labels
    Y_train = df["classID"].values
    # print("Y_train ", Y_train.shape)

    # Get feature matrices
    X_train = df[clap_features].values
    # print("X_train ", X_train.shape)

    # Fit model
    model.fit(X_train, Y_train)

    # Save model
    dump(model, saving_path)


KNN_model_save(df, 30, saving_path) """
