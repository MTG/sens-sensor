"""
This script contains the function that simulates the workflow of a sensor connected to a microphone by progressively reading an audio file,
similar to how a real sensor would record data in real-time.

The script performs the following tasks:
- Simulates audio recording by reading an audio file in fragments, emulating how a sensor would capture and process audio data in chunks.
- In a separate thread, it reassembles these data fragments to reconstruct the complete audio signal.
- The reconstructed audio is then fed into a model to predict values of P (pleasantness), E (eventfulness) or sound sources (specified)
- The predicted values are saved into a text file, with each prediction on a new line.
- Simultaneously, in a separate thread, old data fragments are deleted.

"""

import wave
import datetime
import os
import numpy as np
import matplotlib
import sys
from maad.spl import pressure2leq
from maad.util import mean_dB
from scipy.signal import lfilter
from scipy.signal.filter_design import bilinear
from numpy import pi, convolve
import matplotlib.pyplot as plt
import pandas as pd
import json
from lib.functions_download_data import general_plots

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)


# Imports from this project
import parameters as pm
import lib.client as client
from lib.functions_predictions import initiate


matplotlib.use("Agg")  # Use the 'Agg' backend which does not require a GUI


def send_to_server(content, sensor_id, location):
    values = content.split(";")
    # Create dictionary for "sources" and the rest of the keys
    content = {
        "sources": dict(zip(pm.sources, values[:8])),
        "P_inst": values[8],
        "P_intg": values[9],
        "E_inst": values[10],
        "E_intg": values[11],
        "leq": values[12],
        "LAeq": values[13],
        "datetime": values[14],
    }
    response = client.post_sensor_data_simulation(
        data=content,
        sensor_timestamp=content["datetime"],
        save_to_disk=False,
        sensor_id=sensor_id,
        location=location,
    )

    if response != False:  # Connection is good
        if response.ok == True:  # File sent
            print(f"Prediction sent")
        else:
            print(f"File could not be sent. Server response: {response}")
    else:
        print("No connection.")

    return response.ok


def A_weighting(Fs):
    """Design of an A-weighting filter.

    B, A = A_weighting(Fs) designs a digital A-weighting filter for
    sampling frequency Fs. Usage: y = lfilter(B, A, x).
    Warning: Fs should normally be higher than 20 kHz. For example,
    Fs = 48000 yields a class 1-compliant filter.

    Originally a MATLAB script. Also included ASPEC, CDSGN, CSPEC.

    Author: Christophe Couvreur, Faculte Polytechnique de Mons (Belgium)
            couvreur@thor.fpms.ac.be
    Last modification: Aug. 20, 1997, 10:00am.

    http://www.mathworks.com/matlabcentral/fileexchange/69
    http://replaygain.hydrogenaudio.org/mfiles/adsgn.m
    Translated from adsgn.m to PyLab 2009-07-14 endolith@gmail.com

    References:
       [1] IEC/CD 1672: Electroacoustics-Sound Level Meters, Nov. 1996.

    """
    # Definition of analog A-weighting filter according to IEC/CD 1672.
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    A1000 = 1.9997

    NUMs = [(2 * pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    DENs = convolve(
        [1, +4 * pi * f4, (2 * pi * f4) ** 2],
        [1, +4 * pi * f1, (2 * pi * f1) ** 2],
        mode="full",
    )
    DENs = convolve(
        convolve(DENs, [1, 2 * pi * f3], mode="full"), [1, 2 * pi * f2], mode="full"
    )

    # Use the bilinear transformation to get the digital filter.
    # (Octave, MATLAB, and PyLab disagree about Fs vs 1/Fs)
    return bilinear(NUMs, DENs, Fs)


def calculate_LAeq(audio_samples, fs=48000):
    [B_A, A_A] = A_weighting(fs)
    audio_samples_A = lfilter(B_A, A_A, audio_samples)
    LAeq = mean_dB(pressure2leq(audio_samples_A, fs, 0.125))
    LAeq_str = "{:.2f}".format(LAeq)
    return LAeq_str


def extract_and_flatten_values(data):
    values = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "datetime":
                continue  # Skip the "datetime" key
            values.extend(extract_and_flatten_values(value))
    elif isinstance(data, list):
        for item in data:
            values.extend(extract_and_flatten_values(item))
    else:
        values.append(float(data))
    return values


def sensor_processing(
    audio_file_path: str,
    saving_folder_path: str,
    gain: float,
    timestamp,
    action: list,
    seconds_segment,
    n_segments,
    model_CLAP_path,
    models_predictions_path,
    pca_path,
):

    # Load models
    model_CLAP, models_predictions, pca = initiate(
        model_CLAP_path, models_predictions_path, pca_path
    )

    # Load audio
    wf = wave.open(audio_file_path, "rb")
    fs = wf.getframerate()
    ch = wf.getnchannels()
    sample_width = wf.getsampwidth()

    # Check sample width
    if sample_width == 1:
        dtype = np.uint8  # 8-bit unsigned
    elif sample_width == 2:
        dtype = np.int16  # 16-bit signed
    elif sample_width == 4:
        dtype = np.int32  # 32-bit signed
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # print(f"Fs {fs}, ch {ch}, sample width {sample_width}")

    segment_samples = seconds_segment * fs
    long_buffer_samples = n_segments * segment_samples

    # Buffers for accumulating audio data
    short_buffer = np.array([], dtype=np.float32)
    long_buffer = np.array([], dtype=np.float32)

    # Elapsed time
    elapsed_time = 0

    # Buffers for accumulating predictions
    if "save" in action:
        # Define the columns
        df_columns = [
            "birds",
            "construction",
            "dogs",
            "human",
            "music",
            "nature",
            "siren",
            "vehicles",
            "P_inst",
            "P_intg",
            "E_inst",
            "E_intg",
            "leq",
            "LAeq",
            "date_time",
            "elapsed_time",
        ]
        # Create an empty DataFrame
        df = pd.DataFrame(columns=df_columns)

        # Audio name
        audio_name = (audio_file_path.split("/")[-1]).split(".wav")[0]
        audio_folder = audio_file_path.split("/")[-2]

        # Declare folder where to save plots and predictions
        saving_folder_audio_path = os.path.join(saving_folder_path, audio_name)

    # Read and process the audio stream
    try:
        while True:
            # Read a chunk of audio data from the stream
            # Save in data first chunk of audio
            audio_samples = wf.readframes(segment_samples)
            if not audio_samples:
                break  # End of file reached
            # audio_data = stream.read(segment_samples)
            # audio_samples = np.frombuffer(audio_data, dtype=np.int16)

            # Convert audio_samples to a NumPy array
            audio_samples = np.frombuffer(audio_samples, dtype=dtype)
            audio_samples = audio_samples.reshape(-1, ch)  # Shape as [time, channels]
            audio_samples = audio_samples[:, 0]  # keep only one channel
            # Normalize to [-1, 1]
            if sample_width == 1:
                # 8-bit unsigned
                audio_samples = (audio_samples - 128) / 128
            else:
                max_value = float(2 ** (8 * sample_width - 1) - 1)
                audio_samples = audio_samples / max_value

            # Accumulate the audio samples in both buffers
            short_buffer = audio_samples  # * gain / 6.44  # apply gain
            long_buffer = np.concatenate((long_buffer, audio_samples))
            if len(long_buffer) > long_buffer_samples:
                long_buffer = long_buffer[
                    -long_buffer_samples:
                ]  # maintain only the most recent part

            # Extract features
            features_segment = model_CLAP.get_audio_embedding_from_data(
                [short_buffer], use_tensor=False
            )
            features_intg = model_CLAP.get_audio_embedding_from_data(
                [long_buffer], use_tensor=False
            )
            # Apply PCA
            features_segment = pca.transform(features_segment)
            features_intg = pca.transform(features_intg)

            # finish_time = time.time()

            # Calculate probabilities
            predictions = {}
            for model in models_predictions:
                if model == "P" or model == "E":
                    # Model is P or E
                    predictions[str(model + "_inst")] = models_predictions[
                        model
                    ].predict(features_segment)[0]
                    predictions[str(model + "_intg")] = models_predictions[
                        model
                    ].predict(features_intg)[0]
                else:
                    # Model is a source type
                    # Check if "sources" exists in predictions; if not, initialize it as an empty dictionary
                    if "sources" not in predictions:
                        predictions["sources"] = {}
                    predictions["sources"][model] = models_predictions[
                        model
                    ].predict_proba(features_segment)[0][1]

            # Calculate Leq
            short_buffer = short_buffer
            Leq_calc = mean_dB(pressure2leq(short_buffer * gain, 48000))
            predictions["leq"] = Leq_calc

            # Calculate LAeq
            LAeq = calculate_LAeq(short_buffer * gain, fs=48000)
            predictions["LAeq"] = LAeq

            # Add datetime
            predictions["datetime"] = timestamp.isoformat()

            # SAVE ALL predictions vector in file
            if "save" in action:
                # Create folder to save. Check if folder exists, and if not, create it
                if not os.path.exists(saving_folder_audio_path):
                    os.makedirs(saving_folder_audio_path)
                    print(f"Folder created: {saving_folder_audio_path}")

                ## Save predictions in single txt file
                file_name = (
                    "predictions_" + timestamp.strftime("%Y%m%d_%H%M%S") + ".json"
                )
                txt_file_path = os.path.join(saving_folder_audio_path, file_name)
                """ with open(txt_file_path, "w") as file:
                    file.write(output_line) """
                with open(txt_file_path, "w") as file:
                    json.dump(predictions, file, indent=4)
                print(f"Prediction saved to {txt_file_path}")

                ## Save data in new row in dataframe (dataframe is saved when completed)
                # values = output_line.split(";")
                # Convert values to float, except the last value (datetime string)
                # float_values = [float(value) for value in values[:-1]]  # Convert all but the last one to float
                # Combine converted values
                # Extract all values except the one in the "datetime" key
                float_values = extract_and_flatten_values(predictions)
                final_values = float_values + [predictions["datetime"]] + [elapsed_time]
                # Append the values as a new row to the DataFrame
                df.loc[len(df)] = final_values

            # SEND TO SERVER
            if action == "send":
                # response = send_to_server(output_line, sensor_id=sensor_id, location=location)
                response = client.post_sensor_data(
                    data=predictions,
                    sensor_timestamp=predictions["datetime"],
                    save_to_disk=False,
                )
                """ if response != True:
                    # Prediction was not sent - SAVE IT
                    # Create folder to save predictions. Check if folder exists, and if not, create it
                    if not os.path.exists(saving_folder_path):
                        os.makedirs(saving_folder_path)
                        print(f"Folder created: {saving_folder_path}")
                    file_name = (
                        "predictions_" + timestamp.strftime("%Y%m%d_%H%M%S") + ".txt"
                    )
                    txt_file_path = os.path.join(saving_folder_path, file_name)
                    with open(txt_file_path, "w") as file:
                        file.write(output_line) """

            # Prepare timestamp for next iteration
            timestamp = timestamp + datetime.timedelta(seconds=seconds_segment)
            elapsed_time = elapsed_time + seconds_segment

        # Save plots and dataframe
        if "save" in action:

            # Generate and save plots
            config = {
                "P_inst": {
                    "threshold": 0,
                    "color-line": "#111111",
                },
                "P_intg": {
                    "threshold": 0,
                    "color-line": "#111111",
                },
                "E_inst": {
                    "threshold": 0,
                    "color-line": "#111111",
                },
                "E_intg": {
                    "threshold": 0,
                    "color-line": "#111111",
                },
                "leq": {
                    "threshold": 90,
                    "color-line": "#111111",
                },
                "LAeq": {
                    "threshold": 65,
                    "color-line": "#111111",
                },
                "birds": {
                    "threshold": 0.5,
                    "color": "#8F7E8A",
                    "color-line": "#111111",
                },
                "construction": {
                    "threshold": 0.5,
                    "color": "#EE9E2E",
                    "color-line": "#111111",
                },
                "dogs": {
                    "threshold": 0.5,
                    "color": "#84B66F",
                    "color-line": "#111111",
                },
                "human": {
                    "threshold": 0.5,
                    "color": "#FABA32",
                    "color-line": "#111111",
                },
                "music": {
                    "threshold": 0.5,
                    "color": "#0DB2AC",
                    "color-line": "#111111",
                },
                "nature": {
                    "threshold": 0.5,
                    "color": "#A26294",
                    "color-line": "#111111",
                },
                "siren": {
                    "threshold": 0.5,
                    "color": "#FC694D",
                    "color-line": "#111111",
                },
                "vehicles": {
                    "threshold": 0.8,
                    "color": "#CF6671",
                    "color-line": "#111111",
                },
            }
            config_processed = {
                "P_inst": {
                    "threshold": 0,
                    "color-line": "#000000",
                },
                "P_intg": {
                    "threshold": 0,
                    "color-line": "#000000",
                },
                "E_inst": {
                    "threshold": 0,
                    "color-line": "#000000",
                },
                "E_intg": {
                    "threshold": 0,
                    "color-line": "#000000",
                },
                "leq": {
                    "threshold": 90,
                    "color-line": "#000000",
                },
                "LAeq": {
                    "threshold": 65,
                    "color-line": "#000000",
                },
                "birds": {
                    "threshold": 0.5,
                    "color": "#8F7E8A",
                    "color-line": "#8F7E8A",
                },
                "construction": {
                    "threshold": 0.5,
                    "color": "#EE9E2E",
                    "color-line": "#EE9E2E",
                },
                "dogs": {
                    "threshold": 0.5,
                    "color": "#84B66F",
                    "color-line": "#84B66F",
                },
                "human": {
                    "threshold": 0.5,
                    "color": "#FABA32",
                    "color-line": "#FABA32",
                },
                "music": {
                    "threshold": 0.5,
                    "color": "#0DB2AC",
                    "color-line": "#0DB2AC",
                },
                "nature": {
                    "threshold": 0.5,
                    "color": "#A26294",
                    "color-line": "#A26294",
                },
                "siren": {
                    "threshold": 0.5,
                    "color": "#FC694D",
                    "color-line": "#FC694D",
                },
                "vehicles": {
                    "threshold": 0.8,
                    "color": "#CF6671",
                    "color-line": "#CF6671",
                },
            }

            fontsizes = {
                "title": 20,
                "legend": 14,
                "label": 14,
                "axis": 12,
                "box": 14,
                "percentage": 20,
            }

            general_plots(
                df=df,
                saving_path=saving_folder_audio_path,
                start_str="",
                end_str="",
                config=config,
                fontsizes=fontsizes,
            )

            # Save dataframe with all information
            df.to_csv(os.path.join(saving_folder_audio_path, "data.csv"), index=False)
            print(f"DataFrame saved.")

    except KeyboardInterrupt:
        print("Processing interrupted.")

    finally:
        wf.close()
