import glob
import pickle
import os
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.functions_predictions import crossfade

folder_path="/home/admin/temporary_audios/"
mic_calib_path="/home/admin/noisedata_admin/calib.txt"
pref=20e-6

def Leq_funct(vect, pref, calibration):
    rms=np.sqrt(np.mean(vect**2))
    Leq=20*np.log(rms/pref)
    return Leq

# Read microphone calibration from file
with open(mic_calib_path, "r") as file:
    mic_calib_vect = file.read()
# Convert the read string to a float
mic_calib = float(mic_calib_vect)

# Find all .pkl files in the folder
file_pattern = "segment_*.pkl"
files = glob.glob(os.path.join(folder_path, file_pattern))

# Sort files by timestamp in the filename
files.sort()
files=files[-10:]
# Iterate over each .pkl file
joined_audio = np.empty((0,))
for single_file_path in files:
    # Load data from .pkl file
    with open(single_file_path, "rb") as f:
        single_file_data = pickle.load(f)
    # Append audio data to joined_audio
    audio_segment = (
        single_file_data
    )
    if isinstance(audio_segment, np.ndarray):
        # To join audio apply crossfade, then apply microphone calibration
        joined_audio = crossfade(joined_audio, audio_segment, 1, 48000)
    else:
        print(f"Invalid audio data format in {single_file_path}")

# Play the joined audio
sd.play(joined_audio, 48000)
sd.wait()#wait until the file is done playing

file_path=files[3]
print(file_path, " -------------")
# Load vect from .pkl file
with open(file_path, "rb") as f:
    vect = pickle.load(f)
    print("vect size ", vect.shape)

# Load vect from corresponding txt file
txt_file_path = os.path.join(
    folder_path, file_path.split(".pkl")[0] + ".txt"
)
with open(txt_file_path, "r") as f:
    Leq = float(f.read())
    print("Leq from file", Leq)


Leq_calc=Leq_funct(vect, pref, mic_calib)
print("Leq from calculation", Leq_calc)

# Plot (save plot)
plt.figure(figsize=(10, 4))
time_axis = np.arange(len(vect)) * (1 / 48000)
plt.plot(time_axis, vect)
#plt.ylim([-1, 1])
plt.title("Joined Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
title = "/home/admin/files_plots/plot.png"
plt.show()
plt.savefig(title)  # Save plot to file instead of showing
plt.close()

""" # Play the joined audio
sd.play(vect, 48000)
sd.wait()#wait until the file is done playing """