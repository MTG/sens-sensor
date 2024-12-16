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
        "pleasantness_inst": values[8],
        "pleasantness_intg": values[9],
        "eventfulness_inst": values[10],
        "eventfulness_intg": values[11],
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


def sensor_processing(
    audio_file_path: str,
    saving_folder_path: str,
    gain: float,
    timestamp,
    action,
    sensor_id="",
    location="",
):
    seconds_segment = pm.segment_length
    n_segments = pm.n_segments_intg
    model_CLAP_path = pm.model_CLAP_path
    models_predictions_path = pm.models_predictions_path
    sources = pm.sources

    # Load models
    model_CLAP, models_predictions = initiate(model_CLAP_path, models_predictions_path)

    wf = wave.open(audio_file_path, "rb")
    fs = wf.getframerate()
    ch = wf.getnchannels()
    sample_width = wf.getsampwidth()

    print(f"Fs {fs}, ch {ch}, sample width {sample_width}")

    segment_samples = seconds_segment * fs
    long_buffer_samples = n_segments * segment_samples

    # Buffers for accumulating audio data
    short_buffer = np.array([], dtype=np.int16)
    long_buffer = np.array([], dtype=np.int16)

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
            audio_samples = np.frombuffer(audio_samples, dtype=np.int16)
            audio_samples = audio_samples.reshape(-1, ch)  # Shape as [time, channels]
            audio_samples = audio_samples[:, 0]  # keep only one channel
            audio_samples = audio_samples / (2**15 - 1)  # 16bit convert to wav [-1,1]

            # Accumulate the audio samples in both buffers
            short_buffer = audio_samples * gain / 6.44  # apply gain
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
            # finish_time = time.time()

            # Calculate probabilities for each source model INSTANTANEOUS
            predictions = []
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
                        prediction = models_predictions[model].predict_proba(
                            features_segment
                        )[0][1]
                        predictions.append(prediction)
                    else:
                        # Model is P or E
                        prediction_inst = models_predictions[model].predict(
                            features_segment
                        )[0]
                        predictions.append(prediction_inst)
                        prediction_intg = models_predictions[model].predict(
                            features_intg
                        )[0]
                        predictions.append(prediction_intg)
                else:
                    # This model is not desired, write 0
                    if model in sources:
                        predictions.append(0)
                    else:
                        predictions.append(0)  # for inst
                        predictions.append(0)  # for intg

            # Calculate Leq
            short_buffer = short_buffer
            Leq_calc = mean_dB(pressure2leq(short_buffer * 6.44, 48000))
            Leq_calc_str = "{:.2f}".format(Leq_calc)

            # Calculate LAeq
            LAeq = calculate_LAeq(short_buffer * 6.44, fs=48000)

            # Format the predictions into a string
            prediction_str = ";".join([f"{pred:.2f}" for pred in predictions])

            # Add Leq and the timestamp
            output_line = (
                f"{prediction_str};{Leq_calc_str};{LAeq};{timestamp.isoformat()}"
            )
            print(output_line)

            # SAVE ALL predictions vector in file
            if action == "save":
                # Create folder to save predictions. Check if folder exists, and if not, create it
                if not os.path.exists(saving_folder_path):
                    os.makedirs(saving_folder_path)
                    print(f"Folder created: {saving_folder_path}")
                file_name = (
                    "predictions_" + timestamp.strftime("%Y%m%d_%H%M%S") + ".txt"
                )
                txt_file_path = os.path.join(saving_folder_path, file_name)
                with open(txt_file_path, "w") as file:
                    file.write(output_line)
                print(f"Prediction saved to {txt_file_path}")

            # SEND TO SERVER
            if action == "send":
                response = send_to_server(
                    output_line, sensor_id=sensor_id, location=location
                )
                if response != True:
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
                        file.write(output_line)

            # Prepare timestamp for next iteration
            timestamp = timestamp + datetime.timedelta(seconds=seconds_segment)

    except KeyboardInterrupt:
        print("Processing interrupted.")

    finally:
        wf.close()
