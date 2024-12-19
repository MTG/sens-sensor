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
from development.lib.dataset_functions import generate_USM_extension_dataset
from development.lib.auxiliars import sources


def main(data_path):
    # Declare input paths
    clap_model_path = os.path.join(
        data_path, "models/630k-fusion-best.pt"
    )  # CLAP model
    val_directory_path = os.path.join(data_path, "USM/val")  # VAL files folder
    eval_directory_path = os.path.join(data_path, "USM/eval")  # EVAL files folder
    train_directory_path = os.path.join(data_path, "USM/train")  # TRAIN files folder
    saving_path = os.path.join(
        data_path, "files/USM_CLAP_dataset"
    )  # JSON file path to save

    # Call function
    generate_USM_extension_dataset(
        [val_directory_path, eval_directory_path, train_directory_path],
        clap_model_path,
        sources,
        saving_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SENS feature set (ARAUS features, FreesoundExtractor features and CLAP embeddings) on augmented soundscapes from ARAUS dataset or on new set of audios (listening_test_audios in this case)."
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
# python development/Sources_USM_dataset_Generation.py --data_path data
