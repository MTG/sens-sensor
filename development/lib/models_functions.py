"""
This script contains a collection of functions used for model related operations.

The purpose of this script is to provide a comprehensive set of utilities and methods that facilitate the entire machine learning workflow. It includes functions for:
- Finding and tuning hyperparameters to optimize model performance.
- Training models using different algorithms and configurations.
- Saving and loading model states and configurations for reproducibility.
- Testing and evaluating model performance on validation or test datasets.
"""

import numpy as np
import warnings
import json
import copy
import os
import pandas as pd
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from joblib import dump
from sklearn.metrics import (
    accuracy_score,
    precision_score,
)
from joblib import dump, load
import joblib

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from development.lib.auxiliars import clap_features

# region ARAUS dataset - Pleasantness and Eventfulness Predictions


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

        if input_dict["pca"]:
            # Load the saved PCA model
            pca = joblib.load("data/models/pca_model.pkl")

        for val_fold in [1, 2, 3, 4, 5]:

            # Extract dataframes
            df_train = df[
                (df["info.fold"] != val_fold) & (df["info.fold"] > 0)
            ]  # For the training set, use all samples that are not in the test set (fold 0) and current validation fold.
            df_val = df[df["info.fold"] == val_fold]
            df_test = df[df["info.fold"] == 0]

            # For fold 0 group data
            # Drop string columns
            df_test = df_test.drop("info.file", axis=1)
            df_test = df_test.drop("info.participant", axis=1)
            df_test = df_test.groupby(
                ["info.soundscape", "info.masker", "info.smr"]
            ).mean()  # .reset_index()  # For the test set, the same 48 stimuli were shown to all participants so we take the mean of their ratings as the ground truth

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
            print("X val shape ", X_val.shape)
            X_test = df_test[features].values  # [:,0:100]
            print("X test shape ", X_test.shape)
            X_foldFs = df_Fs[features].values  # [:,0:100]

            if input_dict["pca"]:
                # Apply PCA
                X_train = pca.transform(X_train)
                X_val = pca.transform(X_val)
                X_test = pca.transform(X_test)
                X_foldFs = pca.transform(X_foldFs)

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


# endregion

# region USM dataset - Sound Sources Predictions


def train_USM_models(data_path: str, algorithms: dict, saving_path: str):
    """
    Trains prediction models for a list of sound sources using specified algorithms based on the USM-extended dataset.

    Parameters:
    - data_path (str):
        The path to the JSON file containing the generated USM-extended dataset.
    - algorithms (dict):
        Dict of the pairs source-algorithms, specifying which algorithm to use for training the model to predict each source.
    - saving_path (str):
        The folder path where the trained models will be saved after training.

    Returns:
    - None:
        The function saves the trained models to the specified directory and does not return any value.
    """

    # Check if the directory exists
    if not os.path.exists(saving_path):
        # If it doesn't exist, create it
        os.makedirs(saving_path)
        print(f"Directory {saving_path} created.")

    # Import data from json
    with open(data_path, "r") as f:
        data = json.load(f)
        # Convert JSON data back to DataFrame
        df = pd.DataFrame(data)

    for source in algorithms:

        # Train
        train_USM_model(df, source, saving_path, algorithms[source])


def train_USM_model(
    df: pd.DataFrame, one_class: str, saving_model_path: str, model_choice: str
):
    """
    Trains a prediction model for a specified sound source using a given algorithm and saves the trained model.

    Parameters:
    - df (pandas.DataFrame):
        The DataFrame containing the USM-extended dataset, with CLAP embeddings and binary multilabels for various sources.
    - one_class (str):
        The specific sound source that the model will be trained to predict. This should match a column in the DataFrame
        that contains the binary labels for the presence of this sound source.
    - saving_model_path (str):
        The complete file path where the trained model will be saved. This should include the filename and extension (e.g., '.pkl').
    - model_choice (str):
        The algorithm to be used for training the model. Options might include "l_r" for logistic regression, "r_f" for random forest,
        "KNN" or "linear" for SVC with linear kernel.

    Returns:
    - None:
        The function trains the model and saves it to the specified path. It does not return any value.
    """

    txt_name = os.path.join(saving_model_path, one_class + "_model_info.txt")
    with open(txt_name, "a") as f:

        # Extract dataframes
        df_train = df[df["fold"] == "train"]
        df_val = df[df["fold"] == "val"]
        df_eval = df[df["fold"] == "eval"]

        # Get ground-truth
        Y_train = df_train[one_class].values
        Y_val = df_val[one_class].values
        Y_eval = df_eval[one_class].values
        # print("Y_train ", Y_train.shape)
        # print("Y_val ", Y_val.shape)
        # print("Y_eval ", Y_eval.shape)

        # Get feature matrices
        X_train = df_train[clap_features].values
        X_val = df_val[clap_features].values
        X_eval = df_eval[clap_features].values
        # print("X_train ", X_train.shape)
        # print("X_val ", X_val.shape)
        # print("X_eval ", X_eval.shape)

        # Initialize the classifier
        if model_choice == "l_r":
            f.write("Model type: l_r")
            f.write("\n")
            model = LogisticRegression(solver="lbfgs", max_iter=10000)
        elif model_choice == "r_f":
            f.write("Model type: r_f")
            f.write("\n")
            model = RandomForestClassifier(10)
        elif model_choice == "KNN":
            f.write("Model type: knn")
            f.write("\n")
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_choice == "linear":
            f.write("Model type: linear")
            f.write("\n")
            model = SVC(kernel="linear", probability=True)

        # Train the model
        model.fit(X_train, Y_train)

        # Predict on the test set
        Y_train_pred = model.predict(X_train)
        Y_val_pred = model.predict(X_val)
        Y_eval_pred = model.predict(X_eval)

        # Evaluate the model
        accuracy_train = accuracy_score(Y_train, Y_train_pred)
        accuracy_val = accuracy_score(Y_val, Y_val_pred)
        accuracy_eval = accuracy_score(Y_eval, Y_eval_pred)
        precision_train = precision_score(Y_train, Y_train_pred, zero_division=0)
        precision_val = precision_score(Y_val, Y_val_pred, zero_division=0)
        precision_eval = precision_score(Y_eval, Y_eval_pred, zero_division=0)
        f.write(f"Accuracy score TRAIN: {accuracy_train}")
        f.write("\n")
        f.write(f"Accuracy score VAL: {accuracy_val}")
        f.write("\n")
        f.write(f"Accuracy score EVAL: {accuracy_eval}")
        f.write("\n")
        f.write(f"Precision score TRAIN: {precision_train}")
        f.write("\n")
        f.write(f"Precision score VAL: {precision_val}")
        f.write("\n")
        f.write(f"Precision score EVAL: {precision_eval}")
        f.write("\n")
        """ print("Accuracy score TRAIN:", accuracy_score(Y_train, Y_train_pred))
        print("Accuracy score VAL:", accuracy_score(Y_val, Y_val_pred))
        print("Accuracy score EVAL:", accuracy_score(Y_eval, Y_eval_pred))
        print("Precision score TRAIN:", precision_score(Y_train, Y_train_pred))
        print("Precision score VAL:", precision_score(Y_val, Y_val_pred))
        print("Precision score EVAL:", precision_score(Y_eval, Y_eval_pred)) """

        # Define model saving path
        saving_model_path = os.path.join(saving_model_path, one_class + ".joblib")
        # Save model to given path
        dump(model, saving_model_path)

        print("Saved model for class:", one_class)


# endregion

# region US8k


def train_US8k_models(data_path: str, algorithms: dict, saving_path: str):
    """
    Trains prediction models for a list of sound sources using specified algorithms based on the USM-extended dataset.

    Parameters:
    - data_path (str):
        The path to the JSON file containing the generated US8k-CLAP dataset.
    - algorithms (dict):
        Dict of the pairs source-algorithms, specifying which algorithm to use for training the model to predict each source.
    - saving_path (str):
        The folder path where the trained models will be saved after training.

    Returns:
    - None:
        The function saves the trained models to the specified directory and does not return any value.
    """

    # Check if the directory exists
    if not os.path.exists(saving_path):
        # If it doesn't exist, create it
        os.makedirs(saving_path)
        print(f"Directory {saving_path} created.")

    # Import data from json
    with open(data_path, "r") as f:
        data = json.load(f)
        # Convert JSON data back to DataFrame
        df = pd.DataFrame(data)

    for source in algorithms:

        # Train
        train_US8k_model(df, source, saving_path, algorithms[source])


def train_US8k_model(
    df: pd.DataFrame, one_class: str, saving_model_path: str, model_choice: str
):
    """
    Trains a prediction model for a specified sound source using a given algorithm and saves the trained model.

    Parameters:
    - df (pandas.DataFrame):
        The DataFrame containing the USM-extended dataset, with CLAP embeddings and binary multilabels for various sources.
    - one_class (str):
        The specific sound source that the model will be trained to predict. This should match a column in the DataFrame
        that contains the binary labels for the presence of this sound source.
    - saving_model_path (str):
        The complete file path where the trained model will be saved. This should include the filename and extension (e.g., '.pkl').
    - model_choice (str):
        The algorithm to be used for training the model. Options might include "l_r" for logistic regression, "r_f" for random forest,
        "KNN" or "linear" for SVC with linear kernel.

    Returns:
    - None:
        The function trains the model and saves it to the specified path. It does not return any value.
    """

    # Load the saved PCA model
    pca = joblib.load("data/models/pca_model.pkl")

    txt_name = os.path.join(saving_model_path, one_class + "_model_info.txt")
    with open(txt_name, "a") as f:
        highest_precision_val = 0
        # Iterate over folds (1 to 10 if fold column has values 1-10)
        for fold in sorted(df["fold"].unique()):
            f.write(f"Fold {fold}")
            f.write("\n")

            # Extract dataframes
            df_train = df[df["fold"] != fold]
            df_val = df[df["fold"] == fold]

            # Get ground-truth
            Y_train = df_train[one_class].values
            Y_val = df_val[one_class].values
            # print("Y_train ", Y_train.shape)
            # print("Y_val ", Y_val.shape)

            # Get feature matrices
            X_train = df_train[clap_features].values
            X_val = df_val[clap_features].values

            # Apply PCA
            X_train = pca.transform(X_train)
            X_val = pca.transform(X_val)

            # print("X_train ", X_train.shape)
            # print("X_val ", X_val.shape)

            # Initialize the classifier
            if model_choice == "l_r":
                f.write("Model type: l_r")
                f.write("\n")
                model = LogisticRegression(solver="lbfgs", max_iter=10000)
            elif model_choice == "r_f":
                f.write("Model type: r_f")
                f.write("\n")
                model = RandomForestClassifier(10)
            elif model_choice == "KNN":
                f.write("Model type: knn")
                f.write("\n")
                model = KNeighborsClassifier(n_neighbors=5)
            elif model_choice == "linear":
                f.write("Model type: linear")
                f.write("\n")
                model = SVC(kernel="linear", probability=True)

            # Train the model
            model.fit(X_train, Y_train)

            # Predict on the test set
            Y_train_pred = model.predict(X_train)
            Y_val_pred = model.predict(X_val)

            # Evaluate the model
            accuracy_train = accuracy_score(Y_train, Y_train_pred)
            accuracy_val = accuracy_score(Y_val, Y_val_pred)
            precision_train = precision_score(Y_train, Y_train_pred, zero_division=0)
            precision_val = precision_score(Y_val, Y_val_pred, zero_division=0)

            f.write(f"Accuracy score TRAIN: {accuracy_train}")
            f.write("\n")
            f.write(f"Accuracy score VAL: {accuracy_val}")
            f.write("\n")
            f.write(f"Precision score TRAIN: {precision_train}")
            f.write("\n")
            f.write(f"Precision score VAL: {precision_val}")
            f.write("\n")
            f.flush()  # update file content

            if precision_val > highest_precision_val:
                highest_precision_val = precision_val
                model_to_save = copy.deepcopy(model)
                fold_used = fold
        f.write("- - - - - - - - ")
        f.write(
            f"Model saved with validation fold {fold_used}, the other folds were used for training."
        )
        f.write("\n")
        # Define model saving path
        saving_model_path = os.path.join(saving_model_path, one_class + ".joblib")
        # Save model to given path
        dump(model_to_save, saving_model_path)

        print("Saved model for class:", one_class)


# endregion
