import sys
import os
from scipy.io.wavfile import WavFileWarning
import warnings
import pandas as pd
import argparse

# Suppress WavFileWarning
warnings.filterwarnings("ignore", category=WavFileWarning)

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current dir ", current_dir)
src_dir = os.path.abspath(os.path.join(current_dir, "../"))
print("src dir ", src_dir)
sys.path.append(src_dir)

# Imports from this project
from development.lib.dataset_functions import (
    generate_dataset,
)


def main(data_path, dataset_type):

    if dataset_type == "araus" or dataset_type == "both":
        ############### Code to generate ARAUS extended dataset ##################################
        # Inputs
        audios_path = os.path.join(data_path, "soundscapes_augmented")
        csv_path = os.path.join(data_path, "files/responses_adapted.csv")
        saving_path = os.path.join(data_path, "files/ARAUS_extended_CLAP")

        csv_file = pd.read_csv(csv_path)
        # Call function
        generate_dataset(
            audios_path,
            csv_file,
            saving_path,
            "ARAUS_original",
            1.5,  # 6.44 before
            1,
        )

        #########################################################################################

    if dataset_type == "new" or dataset_type == "both":

        # Code to generate features for new data (listenig tests audios) ########################
        # Inputs
        audios_path = os.path.join(data_path, "listening_test_audios")
        csv_path = os.path.join(data_path, "files/responses_fold_Fs.csv")
        saving_path = os.path.join(data_path, "files/fold_Fs_CLAP")

        csv_file = pd.read_csv(csv_path, delimiter=";")
        # Call function
        generate_dataset(
            audios_path,
            csv_file,
            saving_path,
            "new_data",
            1.5,  # 6.44 before
            1,
        )

        #########################################################################################


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
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        help="Type of dataset to generate: 'araus' for ARAUS-extended, 'new' for new dataset fold-Fs (make sure responses_fold_Fs.csv exists and contains the necessary fields) or 'both' for both.",
    )

    # Parse arguments
    args = parser.parse_args()
    data_path = args.data_path
    dataset_type = args.type

    # Call main function
    main(data_path, dataset_type)

# Run command  example (where sens-sensor/data is where the data is found):
# python development/PE_dataset_Generation.py --data_path data
