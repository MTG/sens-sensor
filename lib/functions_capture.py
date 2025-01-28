import os
import sys
import numpy as np
import datetime
import wave
from maad.spl import pressure2leq
from maad.util import mean_dB
import pickle
import glob

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)


# Imports from this project
import parameters as pm
from lib.functions_simulation import calculate_LAeq


def sensor_capture(device):
    record_continuously(device, 48000, 1, pm.segment_length, pm.audios_folder_path)


def record_continuously(device, fs, channels, seconds_segment, saving_folder_path):
    print("Continuous recording started. Press Ctrl+C to stop.")
    try:
        # Buffer to store audio data
        global audio_buffer
        audio_buffer = []

        # Open the audio stream
        with sd.InputStream(
            device=device,
            samplerate=fs,
            channels=channels,
            callback=audio_callback,
            dtype="int16",
        ):
            while True:
                if len(audio_buffer) >= fs * seconds_segment:
                    # Extract 3 seconds of data
                    data_to_save = np.array(audio_buffer[: fs * seconds_segment])

                    # Take only mono audio
                    if (
                        len(data_to_save.shape) == 2
                    ):  # If the audio has more than one channel
                        print(
                            f"Audio has {data_to_save.shape[1]} channels. Keeping only the first channel."
                        )
                        data_to_save = data_to_save[:, 0]  # Keep only the first channel

                        print(data_to_save.shape)

                    # Leq and LAeq calculation
                    spl_str = calculate_SPL(
                        audio_data=data_to_save, fs=fs, gain=pm.mic_calib
                    )

                    # Generate a timestamped filename
                    date_time = datetime.datetime.now()
                    print("current date time ", date_time)
                    time_str = date_time.strftime("%Y%m%d_%H%M%S")
                    file_name = "segment_" + time_str

                    # Create saving folder
                    if not os.path.exists(saving_folder_path):
                        os.makedirs(saving_folder_path)

                    # Save SPL info in txt file
                    txt_file_path = os.path.join(saving_folder_path, file_name + ".txt")
                    with open(txt_file_path, "w") as f:
                        f.write(spl_str)

                    # Save audio in pickle file
                    pickle_file_path = os.path.join(
                        saving_folder_path, file_name + ".pkl"
                    )
                    with open(pickle_file_path, "wb") as f:
                        norm_gain = 1.5  # This is how AI models were trained with
                        audio_wav = (
                            data_to_save * 1 / norm_gain
                        )  # FIXME 1 can be changed for mic calibration
                        pickle.dump(audio_wav, f)

                    # Check if there are saved files older than specified maintain time (seconds)
                    file_pattern = "segment_*.pkl"
                    files = glob.glob(os.path.join(saving_folder_path, file_pattern))
                    files.sort()
                    for file in files:
                        # Get time stamp of the file
                        file_name = file.split("segment_")[1]
                        file_date_time = file_name.split(".")[0]
                        file_ts = datetime.datetime.strptime(
                            file_date_time, "%Y%m%d_%H%M%S"
                        )
                        time_difference = (date_time - file_ts).total_seconds()
                        if (
                            time_difference > 60
                        ):  # 60 seconds maintain time, older audios are removed
                            # Remove txt and audio file of old audio
                            os.remove(file)
                            os.remove(file.split(".pkl")[0] + ".txt")

                    # Remove the saved data from the buffer
                    audio_buffer = audio_buffer[fs * seconds_segment :]
    except KeyboardInterrupt:
        print("\nRecording stopped.")


def audio_callback(indata, frames, time, status):
    """Callback function to process audio data."""
    if status:
        print(f"Status: {status}")  # Log any errors or warnings
    audio_buffer.extend(indata.copy())


def save_wav_file(data, sample_rate, output_path, channels):
    """Save numpy array as a WAV file."""
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())

    ########################
    # Save dB in txt file
    txt_file_path = pm.foldersave + file_name + ".txt"
    with open(txt_file_path, "w") as f:
        f.write(f"{Leq_Z}")  # FIXME ADD ;{Leq_A} maybe?
    # Save audio in pickle file
    pickle_file_path = pm.foldersave + file_name + ".pkl"
    with open(pickle_file_path, "wb") as f:
        audio_wav = vect_truncated * calib / pm.norm_gain
        pickle.dump(audio_wav, f)


########################


def calculate_SPL(audio_data, fs, gain=1):

    # Calculate Leq
    Leq_calc = mean_dB(pressure2leq(audio_data * gain, fs))

    # Calculate LAeq
    LAeq = calculate_LAeq(audio_data * gain, fs=fs)

    return f"{Leq_calc};{LAeq}"
