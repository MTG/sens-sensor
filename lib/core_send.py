import socket
import time
import os
import glob
import datetime
import RPi.GPIO as GPIO
import sys
import shutil
import json


# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
import lib.client as client
from lib.leds import turn_leds_on, turn_leds_off
import parameters as pm


# OUTDATED FUNCTION --> NOT NEEDED
def get_latest_folder(folder_path):
    # Get the txt files in the folder
    folders = [
        f
        for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f)) and f.startswith("predictions_")
    ]

    if not folders:
        return None

    # Find the most recent folder based on the folder name
    latest_folder = max(folders, key=lambda f: f.split("_")[1:])
    latest_folder_path = os.path.join(folder_path, latest_folder)

    return latest_folder_path


# OUTDATED FUNCTION --> NOT GONNA WORK
def send(folder_path, ip, port):
    sock = connect_to_server(ip, port)
    if not sock:
        print("Could not establish connection to the server. Exiting.")
        return

    # Get folder
    latest_folder_path = get_latest_folder(folder_path)
    print(f"Latest folder {latest_folder_path}")

    # Read predictions, send them and delete them once sent

    last_sent_file = None  # To track last sent file to avoid resending

    try:
        while True:
            # Get latest file
            file_pattern = "*.txt"
            files = glob.glob(os.path.join(latest_folder_path, file_pattern))
            if len(files) != 0:
                # There are files!
                files.sort()
                latest_file = files[-1]
                print(f"Latest file {latest_file}")

                # Check if file was not sent already
                if last_sent_file != latest_file:
                    # New file
                    # Check if file is not empty
                    if os.path.getsize(latest_file) > 0:
                        # File has content
                        # Read and send content
                        with open(latest_file, "r") as file:
                            content = file.read()

                        if sock is not None:
                            # Send predictions
                            try:
                                sock.sendall(content.encode("utf-8"))
                                print(f"Content from file {latest_file} sent.")
                                # Update
                                last_sent_file = latest_file

                                # Proceed to delete sent file
                                os.remove(latest_file)
                                print(f"File {latest_file} deleted.")

                            except (socket.error, BrokenPipeError):
                                print("Connection is lost!")
                                # Try to reconnect
                                connect_to_server(ip, port)

                time.sleep(1)  # FIXME delete

    finally:
        print("Adios")


def connect_to_server(ip, port):
    print(f"Initiating connection to {ip}:{port}")
    attempt = 0
    while True:
        print(f"Attempt {attempt}")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            print("socket created")
            sock.connect((ip, port))
            print(f"Connected to server {ip}:{port}")
            return sock
        except (socket.error, socket.timeout) as e:
            print(f"Connection failed: {e}. Retrying ({attempt+1})...")
            attempt += 1
            time.sleep(1)
    print("Failed to connect after multiple attempts.")
    return None


def send_server():

    # Get parameters needed
    led_pins = [pm.red]  # Define GPIO pins for each LED
    sources = pm.sources
    folder_path = pm.predictions_folder_path

    # Configure LEDs
    GPIO.setmode(GPIO.BCM)  # Set up GPIO mode
    # 20--> Yellow
    # 21--> Red
    # 16--> Green
    for pin in led_pins:  # Set up each LED pin as an output
        GPIO.setup(pin, GPIO.OUT)

    # Read predictions, send them and delete them once sent
    try:
        while True:
            # Get latest file
            file_pattern = "*.json"  # file_pattern = "*.txt"
            files = glob.glob(os.path.join(folder_path, file_pattern))
            if len(files) != 0:
                # There are files!
                files.sort()
                for single_file in files:
                    print("file ", single_file)
                    # Check if file is not empty
                    if os.path.getsize(single_file) > 0:
                        # File has content
                        # Read and send content
                        with open(single_file, "r") as file:
                            content = json.load(single_file)  # content = file.read()

                        """ # Split data string into individual values
                        values = content.split(";")

                        # Create dictionary for "sources" and the rest of the keys
                        content = {
                            "sources": dict(zip(pm.sources, values[:8])),
                            "pleasantness_inst": values[8],
                            "pleasantness_intg": values[9],
                            "eventfulness_inst": values[10],
                            "eventfulness_intg": values[11],
                            "leq": values[12],
                            "LAeq": values[13],
                            "datetime": values[14],
                        }
                        # content["datetime"]=content["datetime"].split("\n")[0]
                        print("content json ", content) """
                        response = client.post_sensor_data(
                            data=content,
                            sensor_timestamp=content["datetime"],
                            save_to_disk=False,
                        )

                        if response != False:  # Connection is good
                            if response.ok == True:  # File sent
                                print(f"Prediction sent - {single_file}")
                                # Proceed to delete sent file
                                os.remove(single_file)
                                print(f"Deleted.")
                                turn_leds_on(GPIO, led_pins)  # Turn on LEDs
                            else:
                                print(
                                    f"File {single_file} could not be sent. Server response: {response}"
                                )
                                turn_leds_off(GPIO, led_pins)
                        else:
                            print("No connection.")

            time.sleep(0.2)

            # If nothing to send, turn off
            # print("waiting...")
            turn_leds_off(GPIO, led_pins)

    finally:
        print("Adios")


def send_library():

    # Get parameters needed
    led_pins = [pm.red]  # Define GPIO pins for each LED
    predictions_folder_path = pm.predictions_folder_path
    not_sent_predictions_folder_path = pm.not_sent_predictions_folder_path
    ip = pm.ip
    port = pm.port

    # First connect to server
    sock = connect_to_server(ip, port)

    # Configure LEDs
    GPIO.setmode(GPIO.BCM)  # Set up GPIO mode
    # 20--> Yellow
    # 21--> Red
    # 16--> Green
    for pin in led_pins:  # Set up each LED pin as an output
        GPIO.setup(pin, GPIO.OUT)

    # Read predictions, send them and delete them once sent
    try:
        while True:
            # Get latest file
            file_pattern = "*.json"
            files = glob.glob(os.path.join(predictions_folder_path, file_pattern))
            if len(files) != 0:
                # There are files!
                files.sort()
                single_file = files[-1]
                # Check if file is not empty
                if os.path.getsize(single_file) > 0:
                    # File has content
                    # Read and send content
                    with open(single_file, "r") as file:
                        content = file.read()

                    # content_screen = content + ";" + str(pm.human_th)
                    content_screen = content
                    content_screen["human_threshold"] = pm.human_th

                    # Check connection
                    turn_leds_on(GPIO, led_pins)  # Turn on LEDs
                    try:
                        sock.sendall(content_screen.encode("utf-8"))
                    except BrokenPipeError:
                        print("Broken pipe error: Socket is no longer available.")
                        sock = connect_to_server(ip, port)
                        sock.sendall(content_screen.encode("utf-8"))
                        print(f"Sent to screen {single_file}")
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")

                    # Move to another folder
                    """ destination_path = (
                        not_sent_predictions_folder_path
                        + single_file.split(predictions_folder_path)[-1]
                    )

                    # Move file to not sent anyways
                    try:
                        if not os.path.exists(not_sent_predictions_folder_path):
                            os.makedirs(not_sent_predictions_folder_path)
                            print(f"Folder created: {not_sent_predictions_folder_path}")

                        shutil.move(single_file, destination_path)
                        print(f"File moved from {single_file} to {destination_path}")
                    except Exception as e:
                        print(
                            f"An error occurred in moving file to another folder: {e}"
                        ) """

                    # Send to server
                    """ # Split data string into individual values
                    values = content.split(";")

                    # Create dictionary for "sources" and the rest of the keys
                    content = {
                        "sources": dict(zip(pm.sources, values[:8])),
                        "pleasantness_inst": values[8],
                        "pleasantness_intg": values[9],
                        "eventfulness_inst": values[10],
                        "eventfulness_intg": values[11],
                        "leq": values[12],
                        "LAeq": values[13],
                        "datetime": values[14],
                    } """
                    # content["datetime"]=content["datetime"].split("\n")[0]
                    response = client.post_sensor_data(
                        data=content,
                        sensor_timestamp=content["datetime"],
                        save_to_disk=False,
                    )

                    if response != False:  # Connection is good
                        if response.ok == True:  # File sent
                            print(f"Prediction sent - {single_file}")
                            # Proceed to delete sent file
                            os.remove(single_file)
                            print(f"Deleted.")
                            turn_leds_on(GPIO, led_pins)  # Turn on LEDs
                        else:  # File NOT sent
                            print(
                                f"File {single_file} could not be sent. Server response: {response}"
                            )

                            turn_leds_off(GPIO, led_pins)

                            # Move file to not sent
                            destination_path = (
                                not_sent_predictions_folder_path
                                + single_file.split(predictions_folder_path)[-1]
                            )
                            try:
                                if not os.path.exists(not_sent_predictions_folder_path):
                                    os.makedirs(not_sent_predictions_folder_path)
                                    print(
                                        f"Folder created: {not_sent_predictions_folder_path}"
                                    )

                                shutil.move(single_file, destination_path)
                                print(
                                    f"File moved from {single_file} to {destination_path}"
                                )
                            except Exception as e:
                                print(
                                    f"An error occurred in moving file to another folder: {e}"
                                )
                    else:
                        print("No connection.")
                        # Move file to not sent
                        destination_path = (
                            not_sent_predictions_folder_path
                            + single_file.split(predictions_folder_path)[-1]
                        )
                        try:
                            if not os.path.exists(not_sent_predictions_folder_path):
                                os.makedirs(not_sent_predictions_folder_path)
                                print(
                                    f"Folder created: {not_sent_predictions_folder_path}"
                                )

                            shutil.move(single_file, destination_path)
                            print(
                                f"File moved from {single_file} to {destination_path}"
                            )
                        except Exception as e:
                            print(
                                f"An error occurred in moving file to another folder: {e}"
                            )

            time.sleep(0.5)

            # If nothing to send, turn off
            # print("waiting...")
            turn_leds_off(GPIO, led_pins)

    finally:
        print("Adios")
