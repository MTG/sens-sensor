import time
import datetime
import pickle
import os
import glob
import numpy as np
import sys
import joblib
import json
import math
import socket
import zoneinfo
import RPi.GPIO as GPIO

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

from CLAP.src.laion_clap import CLAP_Module
import parameters as pm
from lib.towers import create_tower
from lib.functions_leds import turn_leds_on, turn_leds_off


def crossfade(audio1, audio2, duration, fs):
    # If audio1 contains something already (not first iteration)
    if audio1.shape[0] != 0:
        # Get samples
        samples_crossfade = duration * fs
        samples_fade_in = math.floor(samples_crossfade / 2)
        samples_fade_out = samples_fade_in
        # Check if crossfading slots are bigger than actual audios--> raise error
        if samples_fade_out > audio1.shape[0] or samples_fade_in > audio2.shape[0]:
            raise ValueError(
                "Crossfade duration cannot be greater than the length of the respective audio segment."
            )
        # Fade-in/out vectors values
        fade_out = np.linspace(1, 0, samples_fade_out)
        fade_in = np.linspace(0, 1, samples_fade_in)
        # Split audios into needed slots
        audio1_raw = audio1[:-samples_fade_out]
        audio1_cross = audio1[-samples_fade_out:]
        audio2_raw = audio2[samples_fade_in:]
        audio2_cross = audio2[:samples_fade_in]
        # Apply crossfade
        audio1_cross = audio1_cross * fade_out
        audio2_cross = audio2_cross * fade_in
        # Concatenate
        crossfaded = np.concatenate(
            (audio1_raw, audio1_cross + audio2_cross, audio2_raw)
        )

    # If it is first iteration
    else:
        crossfaded = audio2

    return crossfaded


def extract_timestamp(file_name):
    """file_name expected like ../temporary_audios/segment_20241120_141750.txt"""
    # Extract date and time from the file name
    print("file name to split ", file_name)
    date_part, time_part = file_name.split("/segment_")[1].split(".txt")[0].split("_")

    # Parse the date and time
    dt = datetime.datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")

    # Get the current timezone (e.g., local timezone)
    tzinfo = zoneinfo.ZoneInfo(time.tzname[0])

    # Replace the current time with the desired time in the specified timezone
    dt_tz = dt.replace(tzinfo=tzinfo)

    timestamp = dt_tz.replace(microsecond=0)

    return timestamp


def perform_prediction(
    file_path,
    files_path,
    model_CLAP,
    models_predictions,
    pca,
):
    start_pred_time = time.time()
    # Get parameters needed
    folder_path = pm.audios_folder_path
    saving_folder_path = pm.predictions_folder_path

    # SINGLE FILE ANALYSIS
    # Load data from .pkl file
    with open(file_path, "rb") as f:
        file_data = pickle.load(f)
    # Load data from corresponding txt file
    txt_file_name = file_path.split(".pkl")[0] + ".txt"
    txt_file_path = txt_file_name
    # txt_file_path = os.path.join(folder_path, txt_file_name)
    with open(txt_file_path, "r") as f:
        content = f.read().split(";")
        Leq = float(content[0])
        LAeq = float(content[1])
    # Extract features
    print("fila data ", file_data.shape)
    features_single = model_CLAP.get_audio_embedding_from_data(
        [file_data], use_tensor=False
    )
    # Apply PCA
    features_single = pca.transform(features_single)

    # GROUP OF FILES ANALYSIS (INTEGRATED)
    if "P" in models_predictions or "E" in models_predictions:
        # Iterate over each .pkl file
        joined_audio = np.empty((0,))
        for single_file_path in files_path:
            if os.path.exists(single_file_path):
                # Load data from .pkl file
                with open(single_file_path, "rb") as f:
                    single_file_data = pickle.load(f)
                # Append audio data to joined_audio
                audio_segment = single_file_data
                if isinstance(audio_segment, np.ndarray):
                    # To join audio apply crossfade, then apply microphone calibration
                    joined_audio = crossfade(joined_audio, audio_segment, 0.3, 48000)
                else:
                    print(f"Invalid audio data format in {single_file_path}")
        # Extract features
        features_group = model_CLAP.get_audio_embedding_from_data(
            [joined_audio], use_tensor=False
        )
        # Apply PCA
        features_group = pca.transform(features_group)

    # PREDICTIONS
    predictions = {}
    for model in models_predictions:
        if model == "P" or model == "E":
            # Model is P or E
            predictions[str(model + "_inst")] = models_predictions[model].predict(
                features_single
            )[0]
            predictions[str(model + "_intg")] = models_predictions[model].predict(
                features_group
            )[0]
        else:
            # Model is a source type
            # Check if "sources" exists in predictions; if not, initialize it as an empty dictionary
            if "sources" not in predictions:
                predictions["sources"] = {}
            predictions["sources"][model] = models_predictions[model].predict_proba(
                features_single
            )[0][1]

    # Complete dictionary with Leq, LAeq, datetime
    tzinfo = zoneinfo.ZoneInfo(time.tzname[0])
    current_timestamp = datetime.datetime.now(tzinfo).replace(microsecond=0)
    measure_timestamp = extract_timestamp(file_name=txt_file_name)
    predictions["leq"] = Leq
    predictions["LAeq"] = LAeq
    predictions["datetime"] = measure_timestamp.isoformat()

    # Save predictions to a JSON file
    file_name = "predictions_" + measure_timestamp.strftime("%Y%m%d_%H%M%S") + ".json"
    txt_file_path = os.path.join(saving_folder_path, file_name)
    with open(txt_file_path, "w") as file:
        json.dump(predictions, file, indent=4)
    end_pred_time = time.time()
    print(
        f"Predictions added to local json file, time diff {(current_timestamp-measure_timestamp).total_seconds()} seconds"
    )
    print(f"Prediction took {(end_pred_time-start_pred_time)} seconds")

    """ # PREDICTIONS
    all_predictions = [
        "birds",
        "construction",
        "dogs",
        "human",
        "music",
        "nature",
        "siren",
        "vehicles",
        "P",
        "E",
    ]
    predictions = []
    for model in all_predictions:
        if model in models_predictions:
            # This model is desired
            if model in sources:
                # Model is a source type
                prediction = models_predictions[model].predict_proba(features_single)[
                    0
                ][1]
                predictions.append(prediction)
            else:
                # Model is P or E
                prediction_inst = models_predictions[model].predict(features_single)[0]
                predictions.append(prediction_inst)
                prediction_intg = models_predictions[model].predict(features_group)[0]
                predictions.append(prediction_intg)
        else:
            # This model is not desired, write 0
            if model in sources:
                predictions.append(0)
            else:
                predictions.append(0)  # for inst
                predictions.append(0)  # for intg

    # Format the predictions into a string
    prediction_str = ";".join([f"{pred:.2f}" for pred in predictions])

    # Complete preditions line
    tzinfo = zoneinfo.ZoneInfo(time.tzname[0])
    current_timestamp = datetime.datetime.now(tzinfo).replace(microsecond=0)
    measure_timestamp = extract_timestamp(file_name=txt_file_name)
    output_line = f"{prediction_str};{Leq};{LAeq};{measure_timestamp.isoformat()}"

    # Save predictions vector in file
    file_name = "predictions_" + measure_timestamp.strftime("%Y%m%d_%H%M%S") + ".txt"
    txt_file_path = os.path.join(saving_folder_path, file_name)
    with open(txt_file_path, "w") as file:
        file.write(output_line)
    end_pred_time = time.time()
    print(
        f"Predictions added to local txt file, time diff {(current_timestamp-measure_timestamp).total_seconds()} seconds"
    )
    print(f"Prediction took {(end_pred_time-start_pred_time)} seconds") """


def initiate(model_CLAP_path, models_predictions_path, pca_path):
    # region MODEL LOADING #######################
    # Load the CLAP model to generate features
    code_starts = time.time()
    print("------- code starts -----------")
    """ model_CLAP = CLAP_Module(enable_fusion=True)
    print("CLAP MODULE LINE DONE. Start loading checkpoint")
    model_CLAP.load_ckpt(model_CLAP_path)
    print(
        "#############################################################################"
    ) """
    # manually reseed the random number generator as audio fusion relies on random chunks
    np.random.seed(0)
    model_CLAP, _ = create_tower(model_CLAP_path, enable_fusion=True)
    print("------- clap model loaded -----------")

    # Load models for predictions in a dictionary
    models_predictions = {}
    for model in models_predictions_path:
        print(f"...loading {model} model for predictions...")
        models_predictions[model] = joblib.load(models_predictions_path[model])
    print("------- prediction models loaded -----------")

    # Load PCA component
    pca = joblib.load("data/models/pca_model.pkl")

    loaded_end = time.time()
    print(
        "#############################################################################"
    )
    print(
        "Models loaded. It took ",
        loaded_end - code_starts,
        " seconds. ################",
    )
    print(
        "#############################################################################"
    )
    # endregion MODEL LOADING ####################

    return model_CLAP, models_predictions, pca


def sensor_work():

    # Get parameters needed
    led_pins = [pm.yellow]
    saving_path = pm.predictions_folder_path
    models_predictions_path = pm.models_predictions_path
    model_CLAP_path = pm.model_CLAP_path
    pca_path = pm.pca_path
    audios_folder_path = pm.audios_folder_path
    n_segments = pm.n_segments_intg

    # Configure LEDs
    GPIO.setmode(GPIO.BCM)  # Set up GPIO mode
    for pin in led_pins:  # Set up each LED pin as an output
        GPIO.setup(pin, GPIO.OUT)

    # Create folder to save predictions. Check if folder exists, and if not, create it
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
        print(f"Folder created: {saving_path}")

    # Load models
    model_CLAP, models_predictions, pca = initiate(
        model_CLAP_path, models_predictions_path, pca_path
    )

    # Announce that models are loaded
    i = 0
    for i in range(0, 3):
        turn_leds_on(GPIO, led_pins)  # Turn on LEDs
        time.sleep(1)  # Keep them on for 1 second
        turn_leds_off(GPIO, led_pins)  # Turn off LEDs
        time.sleep(1)  # Keep them off for 1 second
        i = i + 1

    # In loop, perform predictions
    prev_file = ""
    while True:
        # start_time = time.time()
        # print("Calculating ...")

        # Find all .pkl files in the folder
        file_pattern = "segment_*.pkl"
        files_path = glob.glob(os.path.join(audios_folder_path, file_pattern))

        # Sort files by timestamp in the filename
        files_path.sort()

        # Take most recent audio file for analysis
        single_file_path = files_path[-1]

        # Is it new?
        if single_file_path != prev_file:
            if os.path.getsize(single_file_path) >= 960163:
                print("New file!")

                # Leave only specified seconds of data (n_segments) for group analysis
                if "P" in models_predictions_path or "E" in models_predictions_path:
                    number_files = len(files_path)
                    if number_files >= n_segments:
                        files_path = files_path[
                            (number_files - n_segments) : number_files
                        ]

                turn_leds_on(GPIO, led_pins)  # Turn on LEDs
                # time.sleep(0.2)  # FIXME to make sure there is content !!!!!! PUT BACK???

                # Perform prediction
                perform_prediction(
                    file_path=single_file_path,
                    files_path=files_path,
                    model_CLAP=model_CLAP,
                    models_predictions=models_predictions,
                    pca=pca,
                )
                prev_file = single_file_path

        # print("Waiting...")
        turn_leds_off(GPIO, led_pins)  # Turn on LEDs
