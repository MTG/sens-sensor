import os
import sys
import zoneinfo
import time
import datetime

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
import parameters as pm


from lib.functions_simulation import sensor_processing

tzinfo = zoneinfo.ZoneInfo(time.tzname[0])

sensor_processing(
    audio_file_path="data/MACBA-Edwin/edge-case/2024-12-01_02.wav",
    saving_folder_path="data/MACBA-Edwin-delete2",
    gain=1,
    timestamp=datetime.datetime.now(tzinfo).replace(microsecond=0),
    action="save",
    sensor_id="MACBA-Edwin_no-or-very-little-problem",
    location="MACBA-Edwin",
    seconds_segment=3,  # pm.segment_length,
    n_segments=10,  # pm.n_segments_intg,
    model_CLAP_path=pm.model_CLAP_path,
    pca_path=pm.pca_path,
    models_predictions_path=pm.models_predictions_path,
    sources=pm.sources,
)
