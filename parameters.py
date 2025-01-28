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

""" for PC
birds": "data/models/sources_USM_pca/birds.joblib",
    "construction": "data/models/sources_USM_pca/construction.joblib",
    "dogs": "data/models/sources_USM_pca/dogs.joblib",
    "human": "data/models/sources_USM_pca/human.joblib",
    "music": "data/models/sources_USM_pca/music.joblib",
    "nature": "data/models/sources_USM_pca/nature.joblib",
    "siren": "data/models/sources_USM_pca/siren.joblib",
    "vehicles": "data/models/sources_IDMT-US8k_pca/vehicles_IDMT.joblib",
    "P": "data/models/models_variations_KNN/model_pleasantness_pca_30.joblib",
    "E": "data/models/models_variations_KNN/model_eventfulness_pca_30.joblib",

"""

""" for sensor
"birds": "data/models/birds.joblib",
    "construction": "data/models/construction.joblib",
    "dogs": "data/models/dogs.joblib",
    "human": "data/models/human.joblib",
    "music": "data/models/music.joblib",
    "nature": "data/models/nature.joblib",
    "siren": "data/models/siren.joblib",
    "vehicles": "data/models/vehicles_IDMT.joblib",
    "P": "data/models/model_pleasantness.joblib",
    "E": "data/models/model_eventfulness.joblib",

"""

models_predictions_path = {
    "birds": "data/models/birds.joblib",
    "construction": "data/models/construction.joblib",
    "dogs": "data/models/dogs.joblib",
    "human": "data/models/human.joblib",
    "music": "data/models/music.joblib",
    "nature": "data/models/nature.joblib",
    "siren": "data/models/siren.joblib",
    "vehicles": "data/models/vehicles_IDMT.joblib",
    "P": "data/models/model_pleasantness.joblib",
    "E": "data/models/model_eventfulness.joblib",
}


# CLAP MODEL
model_CLAP_path = "data/models/630k-fusion-best_audio_only.pt"

# PCA
pca_path = "data/models/pca_model.pkl"

# TEMPORARY AUDIOS FOLDER
audios_folder_path = "../temporary_audios"

# PREDICTIONS FOLDER
predictions_folder_path = "../predictions"
not_sent_predictions_folder_path = "../predictions_not_sent"

# MIC CALIBRATION
mic_calib_path = "../noisedata_admin/calib.txt"
mic_calib = 1

# CONFIGURATIONS
segment_length = 3
n_segments_intg = 10

# PINS ASSIGNMENT
yellow = 20
red = 21
green = 16

# Check for sensor status every 10 messages
status_every = 10

# LIBRARY EXPERIMENT THRESHOLD
human_th = 0.25
