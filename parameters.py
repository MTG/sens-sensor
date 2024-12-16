# SENSOR DATA
sensor_id_path = "../sensor_id.txt"
sensor_location_path = "../location.txt"

# CONNECTION DATA
ip = "10.42.0.48"
port = 65432

# SOUND SOURCES AND PREDICTION MODELS
sources = [
    "birds",
    "construction",
    "dogs",
    "human",
    "music",
    "nature",
    "siren",
    "vehicles",
]
""" models_predictions_path = {
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
} """
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


# CLAP MODEL
model_CLAP_path = "data/models/630k-fusion-best_audio_only.pt"

# TEMPORARY AUDIOS FOLDER
audios_folder_path = "../temporary_audios"

# PREDICTIONS FOLDER
predictions_folder_path = "../predictions"
not_sent_predictions_folder_path = "../predictions_not_sent"

# MIC CALIBRATION
mic_calib_path = "../noisedata_admin/calib.txt"

# CONFIGURATIONS
segment_length = 3
n_segments_intg = 10

# PINS ASSIGNMENT
yellow = 20
red = 21
green = 16

# LIBRARY EXPERIMENT THRESHOLD
human_th = 0.25
