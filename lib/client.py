import requests
import json
import uuid
import os
import datetime
import sys
import msgpack

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
import parameters as pm

API_BASE_URL = "https://sens.upf.edu/sens/api"  # "https://labs.freesound.org/sens/api"
LOCAL_COPY_DATA_PATH = "posted"

# Get SENSOR_ID from a file named sensor_id.txt which should be in the same directory
SENSOR_ID = "unknown"
try:
    with open(pm.sensor_id_path, "r") as f:
        SENSOR_ID = f.read().strip()
except FileNotFoundError:
    print(
        "sensor_id.txt not found. Please create a file named sensor_id.txt in the same directory and write the sensor ID in it."
    )

# Get LOCATION from a file named location.txt which should be in the same directory
LOCATION = "unknown"
try:
    with open(pm.sensor_location_path, "r") as f:
        LOCATION = f.read().strip()
except FileNotFoundError:
    print(
        "location.txt not found. Please create a file named location.txt in the same directory and write the location in it."
    )


def save_posted_data_to_disk(data):
    # Save the data to a JSON file, include timestamp and UUID in the filename so files are sorted easily
    # Use human-readable timestamp in the filename including year, month, day, hour, minute, second
    # If path LOCAL_COPY_DATA_PATH does not exist, create it
    if not os.path.exists(LOCAL_COPY_DATA_PATH):
        os.makedirs(LOCAL_COPY_DATA_PATH)
    filename = (
        f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{data['uuid']}.json"
    )
    with open(os.path.join(LOCAL_COPY_DATA_PATH, filename), "w") as f:
        json.dump(data, f)


def post_sensor_data_nosend(data, sensor_timestamp=None, save_to_disk=True):

    if sensor_timestamp is None:
        sensor_timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    data = {
        "uuid": str(uuid.uuid4()),  # Generate unique uuid for data point
        "sensor_timestamp": sensor_timestamp,  # .isoformat(), # DELETED BECAUSE ALREADY DONE
        "sensor_id": SENSOR_ID,
        "location": LOCATION,
        "data": data,
    }
    if save_to_disk:
        save_posted_data_to_disk(data)

    return data


def post_sensor_data_send(data):

    url = API_BASE_URL + "/sensor-data/"
    # headers = {"Content-Type": "application/json"}
    headers = {"Content-Type": "application/msgpack"}

    data = msgpack.packb(data)  # Pack, binary serializer

    # Send to server
    try:
        # response = requests.post(url, headers=headers, json=data, timeout=10)
        response = requests.post(
            url, headers=headers, data=data, timeout=10, verify=False
        )
    except requests.exceptions.ConnectionError:
        return False
    return response


def post_sensor_data(data, sensor_timestamp=None, save_to_disk=True):
    # Example usage:
    # post_sensor_data({"pleasantness": 0.85, "eventfulness": 0.3, "sources": {"bird": 0.5, "car": 0.2}})
    # post_sensor_data({"pleasantness": 0.85, "eventfulness": 0.3, "sources": {"bird": 0.5, "car": 0.2}}, sensor_timestamp=datetime.datetime(2024, 1, 1, 12, 23))
    url = API_BASE_URL + "/sensor-data/"
    headers = {"Content-Type": "application/json"}
    if sensor_timestamp is None:
        sensor_timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    data = {
        "uuid": str(uuid.uuid4()),  # Generate unique uuid for data point
        "sensor_timestamp": sensor_timestamp,  # .isoformat(), # DELETED BECAUSE ALREADY DONE
        "sensor_id": SENSOR_ID,
        "location": LOCATION,
        "data": data,
    }
    if save_to_disk:
        save_posted_data_to_disk(data)
    # Send to server
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
    except requests.exceptions.ConnectionError:
        return False
    return response


def post_sensor_data_simulation(
    data, sensor_id, location, sensor_timestamp=None, save_to_disk=True
):
    # Example usage:
    # post_sensor_data({"pleasantness": 0.85, "eventfulness": 0.3, "sources": {"bird": 0.5, "car": 0.2}})
    # post_sensor_data({"pleasantness": 0.85, "eventfulness": 0.3, "sources": {"bird": 0.5, "car": 0.2}}, sensor_timestamp=datetime.datetime(2024, 1, 1, 12, 23))
    url = API_BASE_URL + "/sensor-data/"
    headers = {"Content-Type": "application/json"}
    if sensor_timestamp is None:
        sensor_timestamp = datetime.datetime.now().replace(microsecond=0).isoformat()
    data = {
        "uuid": str(uuid.uuid4()),  # Generate unique uuid for data point
        "sensor_timestamp": sensor_timestamp,  # .isoformat(), # DELETED BECAUSE ALREADY DONE
        "sensor_id": sensor_id,
        "location": location,
        "data": data,
    }
    if save_to_disk:
        save_posted_data_to_disk(data)
    # Send to server
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
    except requests.exceptions.ConnectionError:
        return False
    return response


def get_sensor_ids():
    # Example usage:
    # get_sensor_ids()
    url = API_BASE_URL + "/sensor-data/sensor-ids/"
    response = requests.get(url, timeout=10)
    return response.json()


def get_sensors_data(start_date, end_date):
    # Example usage:
    # get_sensors_data(start_date=datetime.datetime(2023,1,1), end_date=datetime.datetime(2025,3,5,22,13))
    url = (
        API_BASE_URL
        + f"/sensor-data/time-range/?start_date={start_date}&end_date={end_date}"
    )
    response = requests.get(url, timeout=10)
    return response.json()


def get_data_for_sensor_id(sensor_id, start_date=None, end_date=None):
    # Example usage:
    # get_data_for_sensor_id(sensor_id='sensor1')
    # get_data_for_sensor_id(sensor_id='sensor1', start_date=datetime.datetime(2023,1,1), end_date=datetime.datetime(2025,3,5,22,13))
    url = API_BASE_URL + f"/download-sensor-data/{sensor_id}/"
    if start_date and end_date:
        url += f"time-range/?start_date={start_date}&end_date={end_date}"
    print("URL OF REQUEST ", url)
    response = requests.get(url, timeout=10)
    return response.json()
