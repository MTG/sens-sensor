import time
import os
import sys
import socket
import glob
import datetime

import RPi.GPIO as GPIO

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.functions_predictions import initiate, perform_prediction
from lib.functions_connection import establish_connection
from lib.leds import turn_leds_on, turn_leds_off
import parameters as pm


def sensor_work():

    # Get parameters needed
    led_pins = [pm.yellow]
    saving_path = pm.predictions_folder_path
    models_predictions_path = pm.models_predictions_path
    model_CLAP_path = pm.model_CLAP_path
    audios_folder_path = pm.audios_folder_path
    n_segments = pm.n_segments_intg

    # Configure LEDs
    GPIO.setmode(GPIO.BCM)  # Set up GPIO mode
    for pin in led_pins:  # Set up each LED pin as an output
        GPIO.setup(pin, GPIO.OUT)

    # Create folder to save predictions. Check if folder exists, and if not, create it
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
        print(f"Folder created: {saving_path}")

    # Load models
    model_CLAP, models_predictions = initiate(model_CLAP_path, models_predictions_path)

    # Announce that models are loaded
    i = 0
    for i in range(0, 3):
        turn_leds_on(GPIO, led_pins)  # Turn on LEDs
        time.sleep(1)  # Keep them on for 1 second
        turn_leds_off(GPIO, led_pins)  # Turn off LEDs
        time.sleep(1)  # Keep them off for 1 second
        i = i + 1

    # In loop, perform predictions
    prev_file = ""
    while True:
        # start_time = time.time()
        # print("Calculating ...")

        # Find all .pkl files in the folder
        file_pattern = "segment_*.pkl"
        files_path = glob.glob(os.path.join(audios_folder_path, file_pattern))

        # Sort files by timestamp in the filename
        files_path.sort()

        # Take most recent audio file for analysis
        single_file_path = files_path[-1]

        # Is it new?
        if single_file_path != prev_file:
            if os.path.getsize(single_file_path) >= 960163:
                print("New file!")

                # Leave only specified seconds of data (n_segments) for group analysis
                if "P" in models_predictions_path or "E" in models_predictions_path:
                    number_files = len(files_path)
                    if number_files >= n_segments:
                        files_path = files_path[(number_files - n_segments) : number_files]

                turn_leds_on(GPIO, led_pins)  # Turn on LEDs
                # time.sleep(0.2)  # FIXME to make sure there is content !!!!!! PUT BACK???

                # Perform prediction 
                perform_prediction(
                    file_path=single_file_path,
                    files_path=files_path,
                    model_CLAP=model_CLAP,
                    models_predictions=models_predictions,
                )
                prev_file = single_file_path

        # print("Waiting...")
        turn_leds_off(GPIO, led_pins)  # Turn on LEDs
