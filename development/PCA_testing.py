import os
import sys
import numpy as np
import joblib
import argparse
from scipy.io import wavfile


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
from lib.functions_predictions import initiate

folder_path = "data/listening_test_audios"
seconds_segment = 30
model_CLAP_path = "data/models/630k-fusion-best_audio_only.pt"
data_path = "data"
audio_file_path = "data/MACBA-Edwin/edge-case/2024-12-01_02.wav"

models_predictions_path = {
    "birds": "data/models/sources/birds.joblib",
    "construction": "data/models/sources/construction.joblib",
    "dogs": "data/models/sources/dogs.joblib",
    "human": "data/models/sources/human.joblib",
    "music": "data/models/sources/music.joblib",
    "nature": "data/models/sources/nature.joblib",
    "siren": "data/models/sources/siren.joblib",
    "vehicles": "data/models/sources/vehicles.joblib",
    "P": "data/models/model_pleasantness.joblib",
    "E": "data/models/model_eventfulness.joblib",
}


# Load CLAP model
from lib.towers import create_tower

np.random.seed(0)
model_CLAP, _ = create_tower(model_CLAP_path, enable_fusion=True)
print("------- clap model loaded -----------")

# Load PCA model
pca = joblib.load(os.path.join(data_path, "models/pca_model.pkl"))

# Load models
# model_CLAP, models_predictions = initiate(model_CLAP_path, models_predictions_path)

# Load the .wav file
sample_rate, audio_data = wavfile.read(audio_file_path)
print(sample_rate, audio_data)
