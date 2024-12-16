"""
This script contains a collection of functions used for model related operations.

The purpose of this script is to provide a comprehensive set of utilities and methods that facilitate the entire machine learning workflow. It includes functions for:
- Finding and tuning hyperparameters to optimize model performance.
- Training models using different algorithms and configurations.
- Saving and loading model states and configurations for reproducibility.
- Testing and evaluating model performance on validation or test datasets.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.exceptions import ConvergenceWarning
import warnings
import json
from joblib import dump, load
import copy
import os
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)


def clip(x, x_min=-1, x_max=1):
    """
    Clips the input vector `x` to be within the range `[x_min, x_max]`.

    Args:
        x (numpy.ndarray): Input vector to be clipped.
        x_min (float, optional): Minimum value for clipping. Defaults to -1.
        x_max (float, optional): Maximum value for clipping. Defaults to 1.

    Returns:
        numpy.ndarray: Clipped vector.
    """
    clipped_x = np.where(x < x_min, x_min, x)
    clipped_x = np.where(clipped_x > x_max, x_max, clipped_x)
    return clipped_x


def test_model(model_path: str, config_file_path: str, df: pd.DataFrame):
    """Function to test new data with already trained models. Specifically,
    in this project the input "df" consists on the Fold-0 variations in calibration,
    to test if the variations generate higher errors. Resulting MAE
    values are printed through terminal"""

    # Load the JSON data
    with open(config_file_path, "r") as file:
        config_dict = json.load(file)
    features = config_dict["features"]
    maskers_active = config_dict["maskers_active"]
    masker_gain = config_dict["masker_gain"]
    masker_transform = config_dict["masker_transform"]
    std_mean_norm = config_dict["std_mean_norm"]
    min_max_norm = config_dict["min_max_norm"]
    predict = config_dict["predict"]
    min = config_dict["min"]
    max = config_dict["max"]
    mean = config_dict["mean"]
    std = config_dict["std"]

    # Prepare data for inputting model
    if maskers_active:
        """features = features + [ # Already saved in "features" field
            "info.masker_bird",
            "info.masker_construction",
            "info.masker_silence",
            "info.masker_traffic",
            "info.masker_water",
            "info.masker_wind",
        ]"""
        if masker_transform == "-1,1":
            df["info.masker_bird"] = (df["info.masker_bird"] * 2 - 1) * masker_gain
            df["info.masker_construction"] = (
                df["info.masker_construction"] * 2 - 1
            ) * masker_gain
            df["info.masker_traffic"] = (
                df["info.masker_traffic"] * 2 - 1
            ) * masker_gain
            df["info.masker_silence"] = (
                df["info.masker_silence"] * 2 - 1
            ) * masker_gain
            df["info.masker_water"] = (df["info.masker_water"] * 2 - 1) * masker_gain
            df["info.masker_wind"] = (df["info.masker_wind"] * 2 - 1) * masker_gain
        else:
            df["info.masker_bird"] = (df["info.masker_bird"]) * masker_gain
            df["info.masker_construction"] = (
                df["info.masker_construction"] * masker_gain
            )
            df["info.masker_traffic"] = df["info.masker_traffic"] * masker_gain
            df["info.masker_silence"] = df["info.masker_silence"] * masker_gain
            df["info.masker_water"] = df["info.masker_water"] * masker_gain
            df["info.masker_wind"] = df["info.masker_wind"] * masker_gain

    # Get X and Y arrays
    X_test = df[features].values
    if predict == "P":
        Y_test = df["info.P_ground_truth"].values
    elif predict == "E":
        Y_test = df["info.E_ground_truth"].values

    # If needed, apply normalization to data
    if std_mean_norm:
        X_test = (X_test - np.array(mean)) / (np.array(std))
    if min_max_norm:
        X_test = (X_test - np.array(min)) / (np.array(max) - np.array(min))

    # Load the model from the .joblib file
    model = load(model_path)

    # Do predictions
    Y_prediction = clip(model.predict(X_test))
    # Get MSEs
    MSE_test = np.mean((Y_prediction - Y_test) ** 2)
    MAE_test = np.mean(np.abs(Y_prediction - Y_test))

    # Print metrics
    print("|   MSE   |   MAE     |")
    print("|---------+-----------|")
    print(f"|  {MSE_test:.4f} |   {MAE_test:.4f}  |")


def train_RFR(input_dict):
    """Function to train Random Forest Regressor model with speficied parameters and input data"""

    # Load dataframes from file paths
    df = pd.read_csv(input_dict["train_dataset_path"])
    df_Fs = pd.read_csv(input_dict["test_dataset_path"])

    # Saving folder
    if not os.path.exists(input_dict["saving_folder_path"]):
        os.makedirs(input_dict["saving_folder_path"])

    txt_name = os.path.join(
        input_dict["saving_folder_path"], input_dict["model_name"] + ".txt"
    )
    with open(txt_name, "a") as f:

        # Suppress ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        # Store interesting input data in output dictionary
        output_dict = {
            "model_name": input_dict["model_name"],
            "features": input_dict["features"],
            "predict": input_dict["predict"],
            "params": input_dict["params"],
        }

        f.write(
            "     |         Mean squared error        |             Mean  error            |"
        )
        f.write("\n")
        f.write(
            "Fold |--------+--------+--------+--------|--------+--------+--------|---------|"
        )
        f.write("\n")
        f.write(
            "     | Train  |   Val  |  Test  |Test(f6)| Train  |   Val  |  Test  | Test(f6)|"
        )
        f.write("\n")
        f.write(
            "-----+--------+--------+--------+--------+--------+--------+--------+----------"
        )
        f.write("\n")
        # Get parameter
        n_estimators = input_dict["params"][0]

        f.write(f"Number of estimators {n_estimators}")
        f.write("\n")
        f.flush()

        # Auxiliary variables to save once best model is chosen
        prev_mean = 9999
        val_fold_chosen = 0
        features = input_dict["features"]

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)

        MSEs_train = []
        MSEs_val = []
        MSEs_test = []
        MSEs_foldFs = []
        MEs_train = []
        MEs_val = []
        MEs_test = []
        MEs_foldFs = []

        for val_fold in [1, 2, 3, 4, 5]:

            # Extract dataframes
            df_train = df[
                (df["info.fold"] != val_fold) & (df["info.fold"] > 0)
            ]  # For the training set, use all samples that are not in the test set (fold 0) and current validation fold.
            df_val = df[df["info.fold"] == val_fold]
            df_test = df[df["info.fold"] == 0]

            # Get ground-truth labels
            if input_dict["predict"] == "P":
                Y_train = df_train["info.P_ground_truth"].values  # [0:10]
                Y_val = df_val["info.P_ground_truth"].values
                Y_test = df_test["info.P_ground_truth"].values
                Y_foldFs = df_Fs["info.P_ground_truth"].values
            elif input_dict["predict"] == "E":
                Y_train = df_train["info.E_ground_truth"].values  # [0:10]
                Y_val = df_val["info.E_ground_truth"].values
                Y_test = df_test["info.E_ground_truth"].values
                Y_foldFs = df_Fs["info.E_ground_truth"].values

            # Get feature matrices
            X_train = df_train[features].values  # [:,0:100]
            X_val = df_val[features].values  # [:,0:100]
            X_test = df_test[features].values  # [:,0:100]
            X_foldFs = df_Fs[features].values  # [:,0:100]

            # Fit model
            model.fit(X_train, Y_train)
            print("Model fit done")

            # Get MSEs
            MSE_train = np.mean((clip(model.predict(X_train)) - Y_train) ** 2)
            MSE_val = np.mean((clip(model.predict(X_val)) - Y_val) ** 2)
            MSE_test = np.mean((clip(model.predict(X_test)) - Y_test) ** 2)
            MSE_foldFs = np.mean((clip(model.predict(X_foldFs)) - Y_foldFs) ** 2)
            ME_train = np.mean(np.abs(clip(model.predict(X_train)) - Y_train))
            ME_val = np.mean(np.abs(clip(model.predict(X_val)) - Y_val))
            ME_test = np.mean(np.abs(clip(model.predict(X_test)) - Y_test))
            ME_foldFs = np.mean(np.abs(clip(model.predict(X_foldFs)) - Y_foldFs))

            # Add metrics
            MSEs_train.append(MSE_train)
            MSEs_val.append(MSE_val)
            MSEs_test.append(MSE_test)
            MSEs_foldFs.append(MSE_foldFs)
            MEs_train.append(ME_train)
            MEs_val.append(ME_val)
            MEs_test.append(ME_test)
            MEs_foldFs.append(ME_foldFs)

            f.write(
                f"fold{val_fold} | {(MSE_train):.4f} | {(MSE_val):.4f} | {(MSE_test):.4f} | {(MSE_foldFs):.4f} | {(ME_train):.4f} | {(ME_val):.4f} | {(ME_test):.4f} | {(ME_foldFs):.4f} |"
            )
            f.write("\n")
            f.write(
                "-----+--------+--------+--------+--------+--------+--------+--------+----------"
            )
            f.write("\n")
            f.flush()

            # Check if validation fold provides the best results
            current_mean = (ME_val + ME_test + ME_foldFs) / 3
            if current_mean < prev_mean:
                prev_mean = current_mean
                model_chosen = copy.deepcopy(model)
                val_fold_chosen = val_fold

        f.write(
            f"Mean | {np.mean(MSEs_train):.4f} | {np.mean(MSEs_val):.4f} | {np.mean(MSEs_test):.4f} | {np.mean(MSEs_foldFs):.4f} | {np.mean(MEs_train):.4f} | {np.mean(MEs_val):.4f} | {np.mean(MEs_test):.4f} | {np.mean(MEs_foldFs):.4f} |"
        )
        f.write("\n")
        f.write(
            "-----+--------+--------+--------+--------+--------+--------+--------+----------"
        )
        f.write("\n")
        f.write(f"N_estimators {n_estimators}, best validation fold {val_fold_chosen}")
        f.write("\n")
        f.flush()

        # Save data to given path
        output_dict["val_fold"] = val_fold_chosen
        json_path_name = os.path.join(
            input_dict["saving_folder_path"], input_dict["model_name"] + "_config.json"
        )

        with open(json_path_name, "w") as json_file:
            json.dump(output_dict, json_file, indent=4)

        # Save model to given path
        model_path_name = os.path.join(
            input_dict["saving_folder_path"], input_dict["model_name"] + ".joblib"
        )
        dump(model_chosen, model_path_name)
