import os
import sys
import sounddevice as sd

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.functions_capture import sensor_capture


def main():

    # Print available audio devices
    print("Available audio devices:")
    print(sd.query_devices())

    # Check if the user provided exactly 2 arguments (IP and Port)
    if len(sys.argv) == 2:
        try:
            device_number = int(sys.argv[1])
            # Call function with the specified device
            sensor_capture(device=device_number)
        except ValueError:
            print("Invalid device number. Please provide a valid integer.")
            sys.exit(1)
    else:
        print("Usage: << python main_capture.py [device_number] >> ")
        sys.exit(1)


if __name__ == "__main__":
    main()
