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
from development.lib.models_functions import train_PE
from development.lib.auxiliars import (
    clap_features,
)


def main(data_path):

    dataset_path = os.path.join(
        data_path,
        "files/ARAUS_CLAP_dataset/ARAUS_CLAP_dataset.csv",  # files/ARAUS_CLAP_dataset/
    )
    data_foldFs_path = os.path.join(
        data_path,
        "files/fold_Fs_CLAP_dataset/fold_Fs_CLAP_dataset.csv",  # files/fold_Fs_CLAP_dataset/
    )
    saving_folder = os.path.join(data_path, "models_variations_KNN")

    estimators = [2, 5, 7, 10, 15, 20, 30, 50, 100, 150, 200]
    for estimator in estimators:

        ############# RUN ###################################################################
        # MODEL FOR PLEASANTNESS PREDICTION - WITH PCA
        input_dict = {
            "train_dataset_path": dataset_path,
            "test_dataset_path": data_foldFs_path,
            "features": clap_features,
            "predict": "P",
            "params": [estimator],
            "saving_folder_path": saving_folder,
            "model_name": "model_pleasantness_pca_" + str(estimator),
            "pca": True,
            "model": "KNN",
            "data_path": data_path,
        }
        train_PE(input_dict)
        # MODEL FOR EVENTFULNESS PREDICTION  - WITH PCA
        input_dict = {
            "train_dataset_path": dataset_path,
            "test_dataset_path": data_foldFs_path,
            "features": clap_features,
            "predict": "E",
            "params": [estimator],
            "saving_folder_path": saving_folder,
            "model_name": "model_eventfulness_pca_" + str(estimator),
            "pca": True,
            "model": "KNN",
            "data_path": data_path,
        }
        train_PE(input_dict)
        # MODEL FOR PLEASANTNESS PREDICTION - No PCA
        input_dict = {
            "train_dataset_path": dataset_path,
            "test_dataset_path": data_foldFs_path,
            "features": clap_features,
            "predict": "P",
            "params": [estimator],
            "saving_folder_path": saving_folder,
            "model_name": "model_pleasantness_" + str(estimator),
            "pca": False,
            "model": "KNN",
            "data_path": data_path,
        }
        train_PE(input_dict)
        # MODEL FOR EVENTFULNESS PREDICTION  - No PCA
        input_dict = {
            "train_dataset_path": dataset_path,
            "test_dataset_path": data_foldFs_path,
            "features": clap_features,
            "predict": "E",
            "params": [estimator],
            "saving_folder_path": saving_folder,
            "model_name": "model_eventfulness_" + str(estimator),
            "pca": False,
            "model": "KNN",
            "data_path": data_path,
        }
        train_PE(input_dict)
        #####################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
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
# python development/PE_RFR_variations.py --data_path data
