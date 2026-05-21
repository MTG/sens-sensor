# SENSOR DATA
sensor_id_path = "../sensor_id.txt"
sensor_location_path = "../location.txt"
sensor_dB_limit_path = "../dB_limit.txt"

# CONNECTION DATA
ip = "192.168.2.2"
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
    "birds": {"model": "data/models/birds.joblib", "pca": "data/models/pca_model.pkl"},  # This is an example, the rest of the sources do not have a pca model, but they could be added in the future
    "construction": {"model": "data/models/construction.joblib", "pca": None},
    "dogs": {"model": "data/models/dogs.joblib", "pca": None},
    "human": {"model": "data/models/human.joblib", "pca": None},
    "music": {"model": "data/models/music.joblib", "pca": None},
    "nature": {"model": "data/models/nature.joblib", "pca": None},
    "siren": {"model": "data/models/siren.joblib", "pca": None},
    "vehicles": {"model": "data/models/vehicles_IDMT.joblib", "pca": None},
    "P": {"model": "data/models/model_pleasantness.joblib", "pca": None},
    "E": {"model": "data/models/model_eventfulness.joblib", "pca": None},
}

# PCA
default_pca_path = "data/models/pca_model.pkl"

# CLAP MODEL
model_CLAP_path = "data/models/630k-fusion-best_audio_only.pt"

# TEMPORARY AUDIOS FOLDER
audios_folder_path = "../temporary_audios"

# PREDICTIONS FOLDER
predictions_folder_path = "../predictions"
not_sent_predictions_folder_path = "../predictions_not_sent"  # This was only used for the "library" version, not used anymore

# ERRORS REGISTER FILE PATH
errors_path = "../error_logs.txt"

# MIC CALIBRATION
mic_calib_path = "../noisedata_admin/calib.txt"
mic_calib = 1

# CONFIGURATIONS
segment_length = 3
n_segments_intg = 10
LAeq_limit = 40  # I think this is not used, instead, the limit is read from a file at sensor_dB_limit_path, but I will keep it here just in case 

# PINS ASSIGNMENT
yellow = 20
red = 21
green = 16
watchdog = 12

# Check for sensor status every 10 messages
status_every = 10

# Send batch of messages every
send_every_sec = 5  # seconds
max_per_batch = 10  # messages max per batch

# LIBRARY EXPERIMENT THRESHOLD
human_th = 0.25


# Set date using google
set_date = True