import os
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.functions_download_data import main_get_data_from_file, main_get_data_specific

# Get data from csv file with columns: Name;Sensor_Id;Location;Date;Start_time;End_time;Saving_Path
""" csv_file_path = "data/recording_points.csv"
main_get_data_from_file(csv_file=csv_file_path) """


# Get data from specific datetime
""" main_get_data_specific(
    sensor_id="sensor_01",
    start_year=2024,
    start_month=11,
    start_day=25,
    start_hour=17,
    start_minute=35,
    start_second=0,
    end_year=2024,
    end_month=11,
    end_day=25,
    end_hour=18,
    end_minute=20,
    end_second=0,
    saving_path="data/datos_experimento_bilioteca/entrada",
) """
