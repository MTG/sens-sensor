import os
import sys
import numpy as np
import joblib
import argparse
import datetime
import psutil
import warnings
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning

# Suppress the specific WavFileWarning
warnings.filterwarnings("ignore", category=WavFileWarning)

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
print("current dir ", current_dir)
src_dir = os.path.abspath(os.path.join(current_dir, "../"))
print("src dir ", src_dir)
sys.path.append(src_dir)


# Imports from this project
from lib.towers import create_tower

folder_path = "data/listening_test_audios"
seconds_segment = 30
model_CLAP_path = "data/models/630k-fusion-best_audio_only.pt"
data_path = "data"
audio_file_path = "data/audios/audio_simulation_1.wav"  # "data/MACBA-Edwin/edge-case/2024-12-01_02.wav"

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

models_predictions_path_pca = {
    "birds": "data/models/sources_USM_pca/birds.joblib",
    "construction": "data/models/sources_IDMT-US8k_pca/construction.joblib",
    "dogs": "data/models/sources_USM_pca/dogs.joblib",
    "human": "data/models/sources_USM_pca/human.joblib",
    "music": "data/models/sources_USM_pca/music.joblib",
    "nature": "data/models/sources_USM_pca/nature.joblib",
    "siren": "data/models/sources_USM_pca/siren.joblib",
    "vehicles": "data/models/sources_IDMT-US8k_pca/vehicles_IDMT.joblib",
    "E": "data/models/models_variations_PE/model_eventfulness_pca_30.joblib",
    "P": "data/models/models_variations_PE/model_pleasantness_pca_30.joblib",
}


# Function to get memory usage in MB
def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Resident Set Size (RSS) in MB


def main(models_predictions_path, do_pca=False):
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    # Load CLAP model
    np.random.seed(0)
    model_CLAP, _ = create_tower(model_CLAP_path, enable_fusion=True)
    # print("------- clap model loaded -----------")

    # Load PCA model
    if do_pca:
        pca = joblib.load(os.path.join(data_path, "models/pca_model.pkl"))

    # Load models for predictions in a dictionary
    models_predictions = {}
    before_memory = get_memory_usage()
    for model in models_predictions_path:
        # print(f"...loading {model} model for predictions...")
        models_predictions[model] = joblib.load(models_predictions_path[model])
        # Get the size of the model file
        """ model_size = os.path.getsize(models_predictions_path[model])
        # Convert to human-readable format (e.g., KB, MB)
        print(f"Model size: {model_size / (1024 * 1024):.2f} MB") """
    after_memory = get_memory_usage()
    print(f"Loading of models occupy in memory {after_memory - before_memory:.2f} MB.")
    # print("------- prediction models loaded -----------")

    # Load the .wav file
    sample_rate, audio_samples = wavfile.read(audio_file_path)
    audio_samples = audio_samples.reshape(-1, 2)  # Shape as [time, channels]
    audio_samples = audio_samples[:, 0]  # keep only one channel
    audio_samples = audio_samples / (2**15 - 1)  # 16bit convert to wav [-1,1]
    # print(sample_rate, audio_samples.shape)

    # Extract features
    start_time_feature = datetime.datetime.now()
    features = model_CLAP.get_audio_embedding_from_data(
        [audio_samples], use_tensor=False
    )
    if do_pca == True:
        # print("PCA selected")
        features = pca.transform(features)
    time_feature = (datetime.datetime.now() - start_time_feature).total_seconds()
    print(f"Time took to calculate features --> {time_feature} seconds.")

    # Perform predictions
    predictions = []
    start_time_predict_all = datetime.datetime.now()
    for model in models_predictions:

        start_time_predict = datetime.datetime.now()
        before_memory = get_memory_usage()
        current_model = models_predictions[model]

        if model != "P" and model != "E":
            # Model is a source type
            prediction = current_model.predict_proba(features)[0][1]
            predictions.append(prediction)
        else:
            # Model is P or E
            prediction_inst = current_model.predict(features)[0]
            predictions.append(prediction_inst)

        after_memory = get_memory_usage()
        time_predict = (datetime.datetime.now() - start_time_predict).total_seconds()
        print(
            f"Model {model} --> \n           time prediction {time_predict} seconds, memory used {after_memory - before_memory:.2f} MB"
        )
    time_predict_all = (
        datetime.datetime.now() - start_time_predict_all
    ).total_seconds()
    print("--------")
    print(f"Time ALL predictions{audio_file_path} --> {time_predict_all} seconds.")
    print("\n")
    print("\n")


print("- - - - - - - - - - - NO PCA - - - - - - - - - - - - - - -")
main(models_predictions_path, False)  # NO PCA
print("- - - - - - - - - - - YES PCA - - - - - - - - - - - - - - -")
main(models_predictions_path_pca, True)  # WITH PCA
