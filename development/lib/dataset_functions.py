"""
This script contains a set of functions to create the new dataset 'ARAUS extension'.

generate_features() is the main function, the other functions adapt the output of
the first function for re-use or processing.
"""

import pandas as pd
import json
import os
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
import warnings
from CLAP.src.laion_clap import CLAP_Module
from scipy.signal import resample
import sys
import wave
import matplotlib.pyplot as plt
import soundfile as sf

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current dir ", current_dir)
src_dir = os.path.abspath(os.path.join(current_dir, "../"))
print("src dir ", src_dir)
sys.path.append(src_dir)

# Imports from this project
from development.lib.auxiliars import (
    clap_features,
)

# region ARAUS dataset - Pleasantness and Eventfulness Predictions


def import_json_to_dataframe(json_path: str, save: bool, saving_path: str):
    """
    Import data from a JSON file and transform it into a DataFrame.

    This function reads a JSON file containing a dataset generated using generate_features()
    from the specified path and converts its content into a pandas DataFrame. Optionally,
    the resulting DataFrame can be saved to a specified path as a CSV file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file to be imported.
    save : bool
        If True, the DataFrame will be saved to the specified saving path.
    saving_path : str
        Path where the DataFrame will be saved if the save parameter is True.

    Outputs
    -------
    dataframe : pandas.DataFrame
        A DataFrame containing the data imported from the JSON file.
    """

    # Load the JSON data
    with open(json_path, "r") as file:
        data = json.load(file)

    # Generate column names list of strings
    for file in data:
        df_row = pd.json_normalize(data[file])
        column_names = df_row.columns.tolist()

    # Generate empty dataframe with column names
    df = pd.DataFrame(columns=column_names)

    # Add each entry in JSON to row of dataframe
    for file in data:
        df_row = pd.json_normalize(data[file])
        df = pd.concat([df, df_row])

    # Expand CLAP embedding
    df = expand_CLAP_features(df)

    if save:
        df.to_csv(saving_path, index=False)
    return df


def import_jsons_to_dataframe(jsons_path: list, save: bool, saving_path: str):
    """
    Import data from a list of JSON files and combine them into a single DataFrame.

    This function reads multiple JSON files (each containing the features generated with
    generate_features() for a single audio file) from the specified list of paths, all
    of which shares the same keys and format, and combines their content into a single
    pandas DataFrame. Optionally, the resulting DataFrame can be saved to a specified path
    as a CSV file.

    Parameters
    ----------
    jsons_path : list of str
        List of paths to the JSON files to be imported. Each JSON file should have the same
        structure and keys.
    save : bool
        If True, the combined DataFrame will be saved to the specified saving path.
    saving_path : str
        Path where the DataFrame will be saved if the save parameter is True.

    Outputs:
    -------
    df : pandas.DataFrame
        A DataFrame containing the combined data from all the JSON files.
    """
    jsons = sorted(os.listdir(jsons_path))
    dfs = []

    for json_file in jsons:
        if json_file.endswith(".json"):
            json_path = os.path.join(jsons_path, json_file)
            print(json_path)
            # Load the JSON data
            with open(json_path, "r") as file:
                data = json.load(file)
            for file in data:
                # Add each entry in JSON to row of dataframe
                df_row = pd.json_normalize(data[file])
                dfs.append(df_row)
    # Concatenate all dataframes
    df = pd.concat(dfs, ignore_index=True)

    if save:
        df.to_csv(saving_path, index=False)
    return df


def import_dataframe_to_json(df, save: bool, saving_path: str):
    """
    Convert a DataFrame to JSON format and optionally save it to a file.

    This function converts a pandas DataFrame containing the dataset of features into a
    JSON format. If specified, the JSON data can be saved to a file at the given path.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be converted to JSON format.
    save : bool
        If True, the JSON data will be saved to the specified saving path.
    saving_path : str
        Path where the JSON file will be saved if the save parameter is True.

    Outputs
    -------
    None
        This function does not return a value. The DataFrame is converted to JSON and
        optionally saved to a file.
    """

    print("Converting dataframe to json")
    json_file = {}
    columns = df.columns.tolist()
    for index, row in df.iterrows():
        row_json = {}
        # For each row of the dataframe
        for column_name in columns:
            keys = column_name.split(".")
            count_keys = len(keys)
            if count_keys == 1:
                if keys[0] not in row_json:
                    row_json[keys[0]] = {}
                row_json[keys[0]] = row[column_name]
            if count_keys == 2:
                if keys[0] not in row_json:
                    row_json[keys[0]] = {}
                if keys[1] not in row_json[keys[0]]:
                    row_json[keys[0]][keys[1]] = {}
                row_json[keys[0]][keys[1]] = row[column_name]
            elif count_keys == 3:
                if keys[0] not in row_json:
                    row_json[keys[0]] = {}
                if keys[1] not in row_json[keys[0]]:
                    row_json[keys[0]][keys[1]] = {}
                if keys[2] not in row_json[keys[0]][keys[1]]:
                    row_json[keys[0]][keys[1]][keys[2]] = {}
                row_json[keys[0]][keys[1]][keys[2]] = row[column_name]
            elif count_keys == 4:
                if keys[0] not in row_json:
                    row_json[keys[0]] = {}
                if keys[1] not in row_json[keys[0]]:
                    row_json[keys[0]][keys[1]] = {}
                if keys[2] not in row_json[keys[0]][keys[1]]:
                    row_json[keys[0]][keys[1]][keys[2]] = {}
                if keys[3] not in row_json[keys[0]][keys[1]][keys[2]]:
                    row_json[keys[0]][keys[1]][keys[2]][keys[3]] = {}
                row_json[keys[0]][keys[1]][keys[2]][keys[3]] = row[column_name]
            elif count_keys == 5:
                if keys[0] not in row_json:
                    row_json[keys[0]] = {}
                if keys[1] not in row_json[keys[0]]:
                    row_json[keys[0]][keys[1]] = {}
                if keys[2] not in row_json[keys[0]][keys[1]]:
                    row_json[keys[0]][keys[1]][keys[2]] = {}
                if keys[3] not in row_json[keys[0]][keys[1]][keys[2]]:
                    row_json[keys[0]][keys[1]][keys[2]][keys[3]] = {}
                if keys[4] not in row_json[keys[0]][keys[1]][keys[2]][keys[3]]:
                    row_json[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]] = {}
                row_json[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]] = row[column_name]
            elif count_keys == 6:
                if keys[0] not in row_json:
                    row_json[keys[0]] = {}
                if keys[1] not in row_json[keys[0]]:
                    row_json[keys[0]][keys[1]] = {}
                if keys[2] not in row_json[keys[0]][keys[1]]:
                    row_json[keys[0]][keys[1]][keys[2]] = {}
                if keys[3] not in row_json[keys[0]][keys[1]][keys[2]]:
                    row_json[keys[0]][keys[1]][keys[2]][keys[3]] = {}
                if keys[4] not in row_json[keys[0]][keys[1]][keys[2]][keys[3]]:
                    row_json[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]] = {}
                if keys[5] not in row_json[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]]:
                    row_json[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]][keys[5]] = {}
                row_json[keys[0]][keys[1]][keys[2]][keys[3]][keys[4]][keys[5]] = row[
                    column_name
                ]
        json_file[index] = row_json
    if save:
        with open(saving_path, "w") as file:
            json.dump(json_file, file, indent=4)


def import_jsons_to_json(jsons_path: list, save: bool, saving_path: str):
    """
    Combine multiple JSON files into a single JSON file.

    This function reads multiple JSON files (each containing the features generated with
    generate_features() for a single audio file) from the specified list of paths, each of which
    shares the same keys and format, and combines their content into a single JSON object.
    Optionally, the combined JSON data can be saved to a specified path.

    Parameters
    ----------
    jsons_path : list of str
        List of paths to the JSON files to be combined. Each JSON file should have the same
        structure and keys.
    save : bool
        If True, the combined JSON data will be saved to the specified saving path.
    saving_path : str
        Path where the combined JSON file will be saved if the save parameter is True.

    Returns
    -------
    single_json : dict
        A dictionary containing the combined data from all the JSON files.
    """

    jsons = sorted(os.listdir(jsons_path))
    single_json = {}
    count = 0
    for json_file in jsons:
        if json_file.endswith(".json"):
            json_path = os.path.join(jsons_path, json_file)
            print(json_path)
            # Load the JSON data
            with open(json_path, "r") as file:
                data = json.load(file)
                single_json[count] = data
            count = count + 1
    if save:
        with open(saving_path, "w") as file:
            json.dump(single_json, file, indent=4)
    return single_json


def file_origin_info(file, participant, gain, audio_info, origin):
    """
    Adapts metadata information from the dataset when generate_features() is called.

    We can work with three different origins of data to generate a dataset with generate_features():
    - "new_data": new audios not found in ARAUS dataset. CSV file of metadata only contains Leq,
        wav_gain, punctuation from listening tests and maskers information.
    - "ARAUS_original": original audios found in ARAUS dataset. CSV file of metadata contains the very
        same content as the original (constitutes responses_SoundLights.csv).
    - "ARAUS_extended": once a new dataset has been generated using generate_features(), any changes on this
        file, is done with this option.
    This function is called when generate_features() is called. It reads input metadata and adapts it depending
    on the data origin so that the resulting datastets, regardless of the origin, have the same metadata format.

    Parameters
    ----------
    file : str
        Name of file.
    participant : str
        Participant information that labelled current file.
    gain : float
        wav_gain information that transforms digital signal to peak-Pascals signal.
    audio_info: dict
        Dictionary with metadata of current file. Its keys depend on the origin.
    origin: str
        Possible origin of file (see explanation above).

    Returns
    -------
    audio_info_json : dict
        Adapted metadata dictionary.
    """

    audio_info_json = {}

    if origin == "new_data":
        # Calculate mean Pleasantness and Eventfulness values
        P, E = calculate_P_E(audio_info)
        # Add basic info about audio to dictionary
        audio_info_json["info"] = {
            "file": file,
            "fold": int(6),
            "wav_gain": float(gain),
            "Leq_R_r": float(audio_info["info.Leq_R_r"].values[0].replace(",", ".")),
            "P_ground_truth": P,
            "E_ground_truth": E,
            "masker_bird": int(audio_info["info.masker_bird"].values[0]),
            "masker_construction": int(
                audio_info["info.masker_construction"].values[0]
            ),
            "masker_silence": int(audio_info["info.masker_silence"].values[0]),
            "masker_traffic": int(audio_info["info.masker_traffic"].values[0]),
            "masker_water": int(audio_info["info.masker_water"].values[0]),
            "masker_wind": int(audio_info["info.masker_wind"].values[0]),
        }
    elif origin == "ARAUS_original":
        audio_info_json["info"] = {
            "file": file,
            "participant": participant,
            "fold": int(audio_info["fold_r"].values[0]),
            "soundscape": audio_info["soundscape"].values[0],
            "masker": audio_info["masker"].values[0],
            "smr": int(audio_info["smr"].values[0]),
            "stimulus_index": int(audio_info["stimulus_index"].values[0]),
            "wav_gain": float(gain),
            "time_taken": audio_info["time_taken"].values[0],
            "is_attention": int(audio_info["is_attention"].values[0]),
            "pleasant": int(audio_info["pleasant"].values[0]),
            "eventful": int(audio_info["eventful"].values[0]),
            "chaotic": int(audio_info["chaotic"].values[0]),
            "vibrant": int(audio_info["vibrant"].values[0]),
            "uneventful": int(audio_info["uneventful"].values[0]),
            "calm": int(audio_info["calm"].values[0]),
            "annoying": int(audio_info["annoying"].values[0]),
            "monotonous": int(audio_info["monotonous"].values[0]),
            "appropriate": int(audio_info["appropriate"].values[0]),
            "P_ground_truth": audio_info["P_ground_truth"].values[0],
            "E_ground_truth": audio_info["E_ground_truth"].values[0],
            "Leq_R_r": audio_info["Leq_R_r"].values[0],
            "masker_bird": int(audio_info["masker_bird"].values[0]),
            "masker_construction": int(audio_info["masker_construction"].values[0]),
            "masker_silence": int(audio_info["masker_silence"].values[0]),
            "masker_traffic": int(audio_info["masker_traffic"].values[0]),
            "masker_water": int(audio_info["masker_water"].values[0]),
            "masker_wind": int(audio_info["masker_wind"].values[0]),
        }
    elif origin == "ARAUS_extended":
        # Add basic info about audio to dictionary
        audio_info_json["info"] = {
            "file": file,
            "participant": participant,
            "fold": int(audio_info["info.fold"].values[0]),
            "soundscape": audio_info["info.soundscape"].values[0],
            "masker": audio_info["info.masker"].values[0],
            "smr": int(audio_info["info.smr"].values[0]),
            "stimulus_index": int(audio_info["info.stimulus_index"].values[0]),
            "wav_gain": float(gain),
            "time_taken": audio_info["info.time_taken"].values[0],
            "is_attention": int(audio_info["info.is_attention"].values[0]),
            "pleasant": int(audio_info["info.pleasant"].values[0]),
            "eventful": int(audio_info["info.eventful"].values[0]),
            "chaotic": int(audio_info["info.chaotic"].values[0]),
            "vibrant": int(audio_info["info.vibrant"].values[0]),
            "uneventful": int(audio_info["info.uneventful"].values[0]),
            "calm": int(audio_info["info.calm"].values[0]),
            "annoying": int(audio_info["info.annoying"].values[0]),
            "monotonous": int(audio_info["info.monotonous"].values[0]),
            "appropriate": int(audio_info["info.appropriate"].values[0]),
            "P_ground_truth": audio_info["info.P_ground_truth"].values[0],
            "E_ground_truth": audio_info["info.E_ground_truth"].values[0],
            "Leq_R_r": audio_info["info.Leq_R_r"].values[0],
            "masker_bird": int(audio_info["info.masker_bird"].values[0]),
            "masker_construction": int(
                audio_info["info.masker_construction"].values[0]
            ),
            "masker_silence": int(audio_info["info.masker_silence"].values[0]),
            "masker_traffic": int(audio_info["info.masker_traffic"].values[0]),
            "masker_water": int(audio_info["info.masker_water"].values[0]),
            "masker_wind": int(audio_info["info.masker_wind"].values[0]),
        }
    return audio_info_json


def generate_dataset(
    audioFolderPath: str,
    csv_file: pd.DataFrame,
    saving_path: str,
    origin: str,
    norm_gain: float = 1,
    variation_gain: float = 1,
):
    """
    Function to generate CLAP embeddings from any input set of audios.

    Parameters
    ----------
    audioFolderPath: str
        Relative path to the folder that contains the audios (.wav files)
    csv_file: pandas.Dataframe
        File that contains metadata information of the audios
    saving_path: str
        Saving path where output JSON or dataset is desired to be saved
    origin: str
        Origin of the generated features "new_data", "ARAUS_original", "ARAUS_extended"

    Outputs:
        output: JSON file / dictionary containing selected features of the corresponding audio files.
        It is saved in automatically in the specified path. Output can be imported as a Pandas dataframe
        using import_json_to_dataframe() function.
    """
    output = {}
    files_count = 0

    # Find the first and last WAV files for json name
    first_wav = None
    last_wav = None

    # Run only once - Load the model
    print("------- code starts -----------")
    model = CLAP_Module(enable_fusion=True)
    print("------- clap module -----------")
    model.load_ckpt("data/models/630k-fusion-best.pt")
    print("------- model loaded -----------")

    # Go over each audio file
    files = sorted(os.listdir(audioFolderPath))
    for file in files:
        if file.endswith(".mp3") or file.endswith(".wav"):
            print("File ", file)

            # Find the first and last WAV files for json name
            if first_wav is None:
                first_wav = file
            last_wav = file

            # if variation_gain code= 9999 means that variation gain should be a random number
            if variation_gain == 9999:
                variation_gain = np.random.uniform(1, 10)

            # Check if this audio had already been processed by checking if json exists
            individual_json_path = os.path.join(saving_path, "individual_jsons")
            csv_base_name = file.split(".")[0]
            json_name = os.path.join(individual_json_path, str(csv_base_name) + ".json")
            if os.path.exists(json_name):
                continue

            audio_path = os.path.join(audioFolderPath, file)
            # Find the row in csv that the audio filename matches to get wav gain
            if origin == "new_data":
                audio_info = csv_file[csv_file["info.file"] == file]
                gain = float(audio_info["info.wav_gain"].values[0].replace(",", "."))
                participant = "mean_of_all"
            elif origin == "ARAUS_extended":
                file_split = file.split("_")
                file_fold = int(file_split[1])
                participant = "ARAUS_" + file_split[3]
                file_stimulus = int(file_split[5].split(".")[0])
                audio_info = csv_file[csv_file["info.fold"] == file_fold]
                audio_info = audio_info[
                    audio_info["info.stimulus_index"] == file_stimulus
                ]
                audio_info = audio_info[audio_info["info.participant"] == participant]
                gain = audio_info["info.wav_gain"].values[0]
            elif origin == "ARAUS_original":
                file_split = file.split("_")
                file_fold = int(file_split[1])
                participant = "ARAUS_" + file_split[3]
                file_stimulus = int(file_split[5].split(".")[0])
                audio_info = csv_file[csv_file["fold_r"] == file_fold]
                audio_info = audio_info[audio_info["stimulus_index"] == file_stimulus]
                audio_info = audio_info[audio_info["participant"] == participant]
                gain = audio_info["wav_gain"].values[0]

            audio_info = file_origin_info(
                file, participant, gain * variation_gain, audio_info, origin
            )

            audio_r, fs = load(audio_path, wav_calib=gain * variation_gain, ch=1)  # R
            audio_l, fs = load(audio_path, wav_calib=gain * variation_gain, ch=0)  # L

            # Normalisation gain to avoid a lot of clipping (because audio variables
            # are in Pascal peak measure, we need "digital version")
            adapted_audio_r = audio_r / norm_gain
            adapted_audio_l = audio_l / norm_gain
            adapted_signal = np.column_stack((adapted_audio_l, adapted_audio_r))
            max_gain = np.max(adapted_audio_r)
            min_gain = np.min(adapted_audio_r)
            # Clipping?
            if max_gain > 1 or min_gain < -1:
                adapted_signal = np.clip(adapted_signal, -1, 1)
            # Save audio provisionally
            provisional_savingPath = os.path.join(saving_path, "provisional")
            if not os.path.exists(provisional_savingPath):
                os.makedirs(provisional_savingPath)
            provisional_saving_path_complete = os.path.join(
                provisional_savingPath, file
            )
            save_wav(adapted_signal, fs, provisional_saving_path_complete)
            # This audio is used to generate CLAP embedding group features

            ## EMBEDDING EXTRACTION
            embedding = extract_CLAP_embeddings(provisional_saving_path_complete, model)
            audio_info["CLAP"] = embedding  # FIXME CLAP EMBEDDING NAMES

            # Delete provisional audio
            if os.path.exists(provisional_savingPath):
                delete_wav(provisional_saving_path_complete)

            # Add this audio's dict to general dictionary
            output[int(files_count)] = audio_info

            # Save info in individual JSON for current audio
            if not os.path.exists(individual_json_path):
                os.makedirs(individual_json_path)
            with open(json_name, "w") as json_file:
                json.dump(audio_info, json_file, indent=4)

            print("Done audio ", files_count)
            files_count = files_count + 1
            print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - -")

    # Save in json
    if first_wav != None:
        first_wav = first_wav.split(".")[0]
        last_wav = last_wav.split(".")[0]
        base_name = "from_" + first_wav + "_to_" + last_wav
        # Check if the saving directory exists, create it if it doesn't
        json_name = os.path.join(saving_path, "Sounds_" + base_name + ".json")
        with open(json_name, "w") as json_file:
            json.dump(output, json_file, indent=4)
            print(f"Saved in json {json_name}")

    # Save in CSV
    csv_name = os.path.join(saving_path, "Sounds_" + base_name + ".csv")
    import_json_to_dataframe(json_name, True, csv_name)

    return output


def expand_CLAP_features(df):
    """
    Expand the 'CLAP' column in the DataFrame into multiple columns.

    This function processes a DataFrame where one of the columns, 'CLAP', contains a vector
    of numbers. The vector is split into individual components, with each component being placed
    into a new column. This transformation allows for more manageable and accessible data for
    subsequent analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame or dataset. Each entry in the 'CLAP' column is a list or array-like structure
        with numerical values, generated using generate_features().

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with the 'CLAP' column expanded into multiple columns. Each component of the
        original 'CLAP' vectors is now in its own separate column (column names determined in
        'clap_features' variable)
    """

    full_list = []
    for index, row in df.iterrows():
        string_list = row["CLAP"]
        del row["CLAP"]
        clap_list = [float(item) for item in string_list]
        complete_new_row = list(row.values) + clap_list
        full_list.append(complete_new_row)

    all_columns = list(df.columns)
    all_columns.remove("CLAP")
    all_columns.extend(clap_features)

    df = pd.DataFrame(data=full_list, columns=all_columns)
    return df


def extract_CLAP_embeddings(audio_path: str, model):

    # Extract embedding
    embedding = model.get_audio_embedding_from_filelist([audio_path], use_tensor=False)

    return embedding[0].tolist()


def calculate_P_E(data):
    attributes = [
        "info.pleasant",
        "info.eventful",
        "info.chaotic",
        "info.vibrant",
        "info.uneventful",
        "info.calm",
        "info.annoying",
        "info.monotonous",
    ]  # Define attributes to extract from dataframes
    ISOPl_weights = [
        1,
        0,
        -np.sqrt(2) / 2,
        np.sqrt(2) / 2,
        0,
        np.sqrt(2) / 2,
        -1,
        -np.sqrt(2) / 2,
    ]  # Define weights for each attribute in attributes in computation of ISO Pleasantness
    ISOEv_weights = [
        0,
        1,
        np.sqrt(2) / 2,
        np.sqrt(2) / 2,
        -1,
        -np.sqrt(2) / 2,
        0,
        -np.sqrt(2) / 2,
    ]  # Define weights for each attribute in attributes in computation of ISO Eventfulness
    P = np.mean(
        ((data[attributes] * ISOPl_weights).sum(axis=1) / (4 + np.sqrt(32))).values
    )  # These are normalised ISO Pleasantness values (in [-1,1])
    E = np.mean(
        ((data[attributes] * ISOEv_weights).sum(axis=1) / (4 + np.sqrt(32))).values
    )  # These are normalised ISO Eventfulness values (in [-1,1])
    return P, E


def load(file, wav_calib=None, ch=None):
    """
    Function from Mosqito library.Full credit is given to the original authors.

    Adaptations have been made to suit our specific needs while retaining the core
    functionality of the original implementations.

    Extract the peak-pressure signal of chosen channel from .wav
    or .uff file and resample the signal to 48 kHz.

    Parameters
    ----------
    file : string
        Path to the signal file
    wav_calib : float, optional
        Wav file calibration factor [Pa/FS]. Level of the signal in Pa_peak
        corresponding to the full scale of the .wav file. If None, a
        calibration factor of 1 is considered. Default to None.
    ch : int, optional for mono files
        Channel chosen

    Outputs
    -------
    signal : numpy.array
        time signal values
    fs : integer
        sampling frequency
    """

    # Suppress WavFileWarning
    warnings.filterwarnings("ignore", category=WavFileWarning)

    # load the .wav file content
    if file[-3:] == "wav" or file[-3:] == "WAV":
        fs, signal = wavfile.read(file)

        # manage multichannel files
        if signal.ndim > 1:
            signal = signal[
                :, ch
            ]  # MODIFICATION: instead of taking channel-0 directly, choose

        # calibration factor for the signal to be in Pa
        if wav_calib is None:
            wav_calib = 1
            print("[Info] A calibration of 1 Pa/FS is considered")
        if isinstance(signal[0], np.int16):
            signal = wav_calib * signal / (2**15 - 1)
        elif isinstance(signal[0], np.int32):
            signal = wav_calib * signal / (2**31 - 1)
        elif isinstance(signal[0], np.float):
            signal = wav_calib * signal

    else:
        raise ValueError("""ERROR: only .wav .mat or .uff files are supported""")

    # resample to 48kHz to allow calculation
    if fs != 48000:
        signal = resample(signal, int(48000 * len(signal) / fs))
        fs = 48000

    return signal, fs


def save_wav(signal, fs, filepath):
    """
    Save the signal to a WAV file.

    Parameters:
    - signal: NumPy array representing the audio signal
    - fs: Sampling frequency of the signal
    - filepath: Path to save the WAV file
    """
    # Check if the signal needs to be converted to int16
    if signal.dtype != np.int16:
        # Scale the signal to the range [-32768, 32767] (16-bit signed integer)
        scaled_signal = np.int16(signal * 32767)
    else:
        scaled_signal = signal

    # Save the WAV file
    wavfile.write(filepath, fs, scaled_signal)


def delete_wav(filepath):
    """
    Delete a WAV file.

    Parameters:
    - filepath: Path to the WAV file to be deleted
    """
    try:
        os.remove(filepath)
    except FileNotFoundError:
        print(f"File '{filepath}' not found.")
    except Exception as e:
        print(f"Error deleting file '{filepath}': {e}")


# endregion

# region USM dataset - Sound Sources Predictions


def generate_features_USM(directory_path: str, model_clap, fold, sources_USM: list):
    """
    Reads a directory containing audio files from the USM dataset, processes each audio file to generate CLAP embeddings,
    and creates a JSON file for the new dataset, USM-extension. The JSON file contains metadata about each audio file,
    including the CLAP embeddings and binary multilabels indicating the presence of different sound sources.

    Parameters:
    - directory_path (str): The path to the directory containing the audio files from the USM dataset.
    - model_clap (CLAP): CLAP model that will be used to generate embeddings for the audio files.
    - fold (str): fold, or name of group of audios, being processed
    - sources_USM: sound sources labels that will form part of the dataset

    Returns:
    - json_entries (JSON): Output dataset in a JSON file.
    """

    # Initialize
    json_entries = []

    # Get all relevant files (audios) in the directory and sort them numerically
    all_files = [f for f in os.listdir(directory_path) if f.endswith("_mix.wav")]
    all_files.sort(key=lambda f: int(f.split("_")[0]))
    # Iterate over all files in the directory
    for file_name in all_files:
        if file_name.endswith("_mix.wav"):

            # Construct the full path to the wav file
            wav_file_path = os.path.join(directory_path, file_name)

            # Extract the index from the file name (e.g., "0" from "0_mix.wav")
            index = file_name.split("_")[0]

            print(index, ", ", fold)

            # Construct the corresponding target file name
            if len(sources_USM) == 26:  # If USM sound sources list has been inputted
                target_file_name = f"{index}_mix_target.npy"
            elif (
                len(sources_USM) == 8
            ):  # If USM simplified sound sources list has been inputted
                target_file_name = f"{index}_mix_target_simp.npy"
            target_file_path = os.path.join(directory_path, target_file_name)

            # Check if the corresponding target file exists
            if os.path.exists(target_file_path):

                # Extract features of audio
                # start_time_CLAP = time.time()
                embedding = model_clap.get_audio_embedding_from_filelist(
                    [wav_file_path], use_tensor=False
                )[0]
                # duration_time_CLAP = time.time() - start_time_CLAP
                # print("Embedding calculated in ", duration_time_CLAP, " seconds")

                # Create new entry
                entry = {
                    "file_name": file_name,
                    "target_file_name": target_file_name,
                    "fold": fold,
                    "index": index,
                }

                # Import multi-label array
                multiclass_vector = np.load(target_file_path)

                # Expand the embeddings into separate columns
                for i, name in enumerate(clap_features):
                    entry[name] = float(embedding[i])

                # Expand the binary multi-labels into separate columns
                for i, source in enumerate(sources_USM):
                    entry[source] = float(multiclass_vector[i])

        # Add new entry to JSON
        json_entries.append(entry)

    return json_entries


def generate_USM_extension_dataset(
    list_directory_path: list,
    clap_model_path: str,
    sources_USM: list,
    saving_folder: str,
):
    """
    Reads a list of directories containing audio files from the USM dataset (one for each fold or group of audios),
    processes each audio file to generate CLAP embeddings, and creates a complete and general JSON
    file for the new dataset, USM-extension. The JSON file contains metadata about each audio file, including the
    CLAP embeddings and binary multilabels indicating the presence of different sound sources.

    Parameters:
    - list_directory_path (list): List of paths to the directories containing the audio files from the USM dataset.
        Each directory consists of the audios corresponding to one fold of USM.
    - clap_model_path (CLAP):Path to the CLAP model that will be used to generate embeddings for the audio files.
    - sources_USM: sound sources labels that will form part of the dataset
    - saving_folder (str): Path to folder.

    Returns:
    - combined_data (JSON): Output dataset in a JSON file. Also, the function writes this JSON to a JSON file.
    """

    # Initialize
    combined_data = []

    # Load clap model
    print("------- code starts -----------")
    model_clap = CLAP_Module(enable_fusion=True)
    print("------- clap module -----------")
    model_clap.load_ckpt(clap_model_path)
    print("------- model loaded -----------")

    for fold_directory_path in list_directory_path:

        # Fold name
        fold = fold_directory_path.split("/")[2]
        print("Fold ", fold)

        # Generate features for fold
        fold_json = generate_features_USM(
            fold_directory_path, model_clap, fold, sources_USM
        )

        # Add to general json
        combined_data.extend(fold_json)

    # Check if the directory exists
    if not os.path.exists(saving_folder):
        # If it doesn't exist, create it
        os.makedirs(saving_folder)
        print(f"Directory {saving_folder} created.")

    # Save the combined data to a new JSON file
    with open(os.path.join(saving_folder, "USM_CLAP_dataset.json"), "w") as outfile:
        json.dump(combined_data, outfile, indent=4)

    # Save the combined data to a CSV file
    df_csv = pd.DataFrame(combined_data)
    df_csv.to_csv(os.path.join(saving_folder, "USM_CLAP_dataset.csv"), index=False)

    return combined_data


# endregion

# region US8K dataset - Sound Sources Predictions


def generate_features_US8k(dataset_path, data_path, saving_folder):
    # Load the model
    print("------- code starts -----------")
    model = CLAP_Module(enable_fusion=True)
    print("------- clap module -----------")
    model.load_ckpt("data/models/630k-fusion-best.pt")
    print("------- model loaded -----------")

    # Read dataframe
    df = pd.read_csv(dataset_path)
    # Initialize empty list to store JSON entries
    json_entries = []
    # Iterate over rows and calculate CLAP features for each audio
    for index, row in df.iterrows():
        print(index + 1, "/8732")
        audio_path = os.path.join(data_path, row["audio_path"])
        embedding = model.get_audio_embedding_from_filelist(
            [audio_path], use_tensor=False
        )[0]
        # print("Embedding calculated in ", duration_time_CLAP, " seconds")
        # Create new entry
        entry = row.to_dict()

        # Expand the embeddings into separate columns
        for i, name in enumerate(clap_features):
            entry[name] = float(embedding[i])

        print(entry)

        # Add new entry to JSON
        json_entries.append(entry)

    # Check if the directory exists
    if not os.path.exists(os.path.join(data_path, saving_folder)):
        # If it doesn't exist, create it
        os.makedirs(os.path.join(data_path, saving_folder))
        print(f"Directory {os.path.join(data_path,saving_folder)} created.")

    # Write the JSON entries to a file
    with open(
        os.path.join(data_path, saving_folder, "US8k_CLAP_dataset.json"), "w"
    ) as f:
        json.dump(json_entries, f, indent=4)

    # Save  data to a CSV file
    df_csv = pd.DataFrame(json_entries)
    df_csv.to_csv(
        os.path.join(data_path, saving_folder, "US8k_CLAP_dataset.csv"), index=False
    )


def load_wav_to_array(file_path):
    print("File ", file_path)

    # Read file using soundfile
    audio_array, fs = sf.read(
        file_path, dtype="float32"
    )  # Ensures values are in [-1.0, 1.0]

    # If multi-channel, take only the first channel (convert to mono)
    if len(audio_array.shape) > 1:
        audio_array = audio_array[:, 0]

    # Plot waveform (first channel if stereo)
    """plt.plot(audio_array)
    plt.title(f"Audio Waveform {file_path}")
    plt.show()"""

    return audio_array, fs


def generate_features_US8k_2s(dataset_path, data_path, saving_folder):
    # Load the model
    print("------- code starts -----------")
    model = CLAP_Module(enable_fusion=True)
    print("------- clap module -----------")
    model.load_ckpt("data/models/630k-fusion-best.pt")
    print("------- model loaded -----------")

    # Read dataframe
    df = pd.read_csv(dataset_path)
    # Initialize empty list to store JSON entries
    json_entries = []
    # Iterate over rows and calculate CLAP features for each audio
    for index, row in df.iterrows():
        print(index + 1, "/8732")
        audio_path = os.path.join(data_path, row["audio_path"])
        # Read audio
        audio_array, fs = load_wav_to_array(file_path=audio_path)
        audio_length = len(audio_array)
        # Cut audio to 2 seconds
        """ samples_cut = 2 * fs
        if audio_length > samples_cut:
            audio_array = audio_array[0:samples_cut] """

        # Get CLAP embedding
        embedding = model.get_audio_embedding_from_data(
            [audio_array], use_tensor=False
        )[0]

        # print("Embedding calculated in ", duration_time_CLAP, " seconds")
        # Create new entry
        entry = row.to_dict()

        # Expand the embeddings into separate columns
        for i, name in enumerate(clap_features):
            entry[name] = float(embedding[i])

        # print(entry)

        # Add new entry to JSON
        json_entries.append(entry)

    # Check if the directory exists
    if not os.path.exists(os.path.join(data_path, saving_folder)):
        # If it doesn't exist, create it
        os.makedirs(os.path.join(data_path, saving_folder))
        print(f"Directory {os.path.join(data_path,saving_folder)} created.")

    # Write the JSON entries to a file
    with open(
        os.path.join(data_path, saving_folder, "US8k_CLAP_dataset.json"), "w"
    ) as f:
        json.dump(json_entries, f, indent=4)

    # Save  data to a CSV file
    df_csv = pd.DataFrame(json_entries)
    df_csv.to_csv(
        os.path.join(data_path, saving_folder, "US8k_CLAP_dataset.csv"), index=False
    )


# endregion
