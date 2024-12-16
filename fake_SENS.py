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


from lib.functions_simulation import sensor_processing

tzinfo = zoneinfo.ZoneInfo(time.tzname[0])

sensor_processing(
    audio_file_path="data/audios/audio_simulation_1.wav",
    saving_folder_path="data/delete",
    gain=1,
    timestamp=datetime.datetime.now(tzinfo).replace(microsecond=0),
    action="send",
    sensor_id="fake_simulation",
    location="Amaia's Laptop",
)
