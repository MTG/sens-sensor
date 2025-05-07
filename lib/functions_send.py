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
from lib.functions_leds import turn_leds_on, turn_leds_off
import parameters as pm
from lib.functions_status import gather_raspberry_pi_info


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
    folder_path = pm.predictions_folder_path
    status_every = pm.status_every
    errors_path = pm.errors_path

    # Configure LEDs
    GPIO.setmode(GPIO.BCM)  # Set up GPIO mode
    # 20--> Yellow
    # 21--> Red
    # 16--> Green
    for pin in led_pins:  # Set up each LED pin as an output
        GPIO.setup(pin, GPIO.OUT)

    #### WATCHDOG code ###
    watchdog_pin = pm.watchdog
    GPIO.setup(watchdog_pin, GPIO.OUT)
    ####################

    # Counter for sending sensor status updates
    counter_status = 0

    # Read predictions, send them and delete them once sent
    try:
        while True:
            # Get latest file
            file_pattern = "*.json"  # file_pattern = "*.txt"
            files = glob.glob(os.path.join(folder_path, file_pattern))
            if len(files) != 0:
                # There are files!
                files.sort()
                most_recent = files[-1]
                # Check if most recent file is not empty
                if os.path.getsize(most_recent) > 0:
                    for single_file in files:
                        print("file ", single_file)
                        # Check if file is not empty
                        if os.path.getsize(single_file) > 0:
                            # File has content
                            # Read and send content
                            with open(single_file, "r") as file:
                                content = json.load(file)

                            # Sensor status
                            print(counter_status)
                            if counter_status == 0:
                                sensor_info = gather_raspberry_pi_info()
                                content["sensor_info"] = sensor_info

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
                                    #### WATCHDOG code ###
                                    GPIO.output(watchdog_pin, GPIO.HIGH)  # Send pulse
                                    ####################
                                    # Update counter status
                                    counter_status = counter_status + 1
                                    if counter_status >= status_every:
                                        counter_status = 0
                                else:
                                    print(
                                        f"File {single_file} could not be sent. Server response: {response}"
                                    )
                                    turn_leds_off(GPIO, led_pins)
                                    #### WATCHDOG code ###
                                    GPIO.output(watchdog_pin, GPIO.LOW)  # Stop pulse
                                    ####################
                            else:
                                print("No connection.")
                        else:
                            # File is old and empty! Delete to not accumulate!
                            os.remove(single_file)
                            log_text = (
                                f"Send process: Deleted because empty --> {single_file}"
                            )
                            update_logs_file(errors_path, log_text)

            time.sleep(0.2)

            # If nothing to send, turn off
            print("waiting...")
            turn_leds_off(GPIO, led_pins)
            #### WATCHDOG code ###
            GPIO.output(watchdog_pin, GPIO.LOW)  # Stop pulse
            ####################

    finally:
        print("Adios")


def send_server_batch():

    # Get parameters needed
    led_pins = [pm.red]  # Define GPIO pins for each LED
    folder_path = pm.predictions_folder_path
    status_every = pm.status_every
    errors_path = pm.errors_path
    send_every_sec = pm.send_every_sec
    max_per_batch = pm.max_per_batch

    # Configure LEDs
    GPIO.setmode(GPIO.BCM)  # Set up GPIO mode
    # 20--> Yellow
    # 21--> Red
    # 16--> Green
    for pin in led_pins:  # Set up each LED pin as an output
        GPIO.setup(pin, GPIO.OUT)

    #### WATCHDOG code ###
    watchdog_pin = pm.watchdog
    GPIO.setup(watchdog_pin, GPIO.OUT)
    ####################

    # Counter for sending sensor status updates
    counter_status = 0

    # Read predictions, send them and delete them once sent
    try:
        while True:
            # Get JSON files
            file_pattern = "*.json"  # file_pattern = "*.txt"
            files = glob.glob(os.path.join(folder_path, file_pattern))
            if len(files) >= max_per_batch:
                # There are enough files!
                files.sort()
                most_recent = files[-1]
                # check
                print("most recent file ", most_recent)
                print("least recent file ", files[0])
                data_list = []  # to accumulate jsons of data
                files_list = []

                batch_counter = 0
                for single_file in files:

                    print("file ", single_file)
                    # Check if file is not empty
                    if os.path.getsize(single_file) > 0:
                        # File has content
                        # Read and send content
                        """with open(single_file, "r") as file:
                            content = json.load(file)"""

                        try:
                            with open(single_file, "r") as file:
                                content = json.load(file)
                        except json.JSONDecodeError:
                            # File is old and empty! Delete to not accumulate!
                            os.remove(single_file)
                            log_text = f"Send process: Deleted because error in JSON --> {single_file}"
                            update_logs_file(errors_path, log_text)
                            continue

                        # increase counter of messages in batch
                        batch_counter = batch_counter + 1

                        if max_per_batch == batch_counter:
                            # Add sensor status to last message
                            sensor_info = gather_raspberry_pi_info()
                            content["sensor_info"] = sensor_info

                        # Create complete content entry
                        content = client.post_sensor_data_nosend(
                            content,
                            sensor_timestamp=content["datetime"],
                            save_to_disk=False,
                        )
                        # add to list of messages
                        data_list.append(content)
                        files_list.append(single_file)

                        if max_per_batch == batch_counter:
                            # Completed message, send it!
                            response = client.post_sensor_data_send(
                                data=data_list,
                            )

                            if response != False:  # Connection is good
                                if response.ok == True:  # File sent
                                    for single_file in files_list:
                                        print(f"Prediction sent - {single_file}")
                                        # Proceed to delete sent file
                                        os.remove(single_file)
                                        print(f"Deleted.")
                                        # OK --> activate LEDs and watchdog
                                        turn_leds_on(GPIO, led_pins)  # ON
                                        GPIO.output(watchdog_pin, GPIO.HIGH)
                                        time.sleep(0.1)  # Make sure watchdog receives
                                        GPIO.output(watchdog_pin, GPIO.LOW)
                                        turn_leds_off(GPIO, led_pins)  # OFF
                                        ####################

                                        # Reset for next iterations
                                        batch_counter = 0
                                        data_list = []
                                        files_list = []

                                else:
                                    print(
                                        f"Files could not be sent. Server response: {response}"
                                    )
                                    turn_leds_off(GPIO, led_pins)
                                    #### WATCHDOG code ###
                                    GPIO.output(watchdog_pin, GPIO.LOW)  # Stop pulse
                                    ####################
                            else:
                                print("No connection.")

                    else:
                        # If it is the most recent (meaning that it is still being written)
                        if single_file == most_recent:
                            continue
                        # File is old and empty! Delete to not accumulate!
                        os.remove(single_file)
                        log_text = (
                            f"Send process: Deleted because empty --> {single_file}"
                        )
                        update_logs_file(errors_path, log_text)

            # If nothing to send, turn off
            print("waiting...")
            turn_leds_off(GPIO, led_pins)
            #### WATCHDOG code ###
            GPIO.output(watchdog_pin, GPIO.LOW)  # Stop pulse
            ####################

            time.sleep(send_every_sec)  # wait to accumulate messages for a minute

    finally:
        print("Adios")


def update_logs_file(file_path, new_content):
    # Check if the file exists
    if not os.path.exists(file_path):
        # Create the file and write the initial content
        with open(file_path, "w") as file:
            file.write(new_content + "\n")
        print(f"File created and content written: {new_content}")
    else:
        # Read the current content
        with open(file_path, "r") as file:
            current_content = file.read()
        print("Current content of the file:")
        print(current_content)

        # Append new content
        with open(file_path, "a") as file:
            file.write(new_content + "\n")
        print(f"New content appended: {new_content}")


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
                        content = json.load(file)

                    # content_screen = content + ";" + str(pm.human_th)
                    content_screen = content
                    content_screen["human_threshold"] = pm.human_th

                    # Check connection
                    turn_leds_on(GPIO, led_pins)  # Turn on LEDs
                    try:
                        sock.sendall(json.dumps(content_screen).encode("utf-8"))
                    except BrokenPipeError:
                        print("Broken pipe error: Socket is no longer available.")
                        sock = connect_to_server(ip, port)
                        sock.sendall(json.dumps(content_screen).encode("utf-8"))
                        print(f"Sent to screen {single_file}")
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")

                    # Send to server
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
