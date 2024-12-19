import time
import os
import sys
import pandas as pd
import argparse

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current dir ", current_dir)
src_dir = os.path.abspath(os.path.join(current_dir, "../"))
print("src dir ", src_dir)
sys.path.append(src_dir)

# Imports from this project
from development.lib.dataset_functions import generate_features_US8k


def main(data_path):
    # Inputs
    dataset_path = os.path.join(data_path, "files/UrbanSound8K_adapted.csv")
    saving_folder = "files/UrbanSound8K_CLAP_dataset"

    generate_features_US8k(
        data_path=data_path, dataset_path=dataset_path, saving_folder=saving_folder
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate feature set (CLAP embeddings) on UrbanSoundscapes8k dataset audios."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data directory.",
    )

    # Parse arguments
    args = parser.parse_args()
    data_path = args.data_path

    # Call main function
    main(data_path)

# Run command  example (where sens-sensor/data is where the data is found):
# python development/Sources_US8k_dataset_Generation.py --data_path data
