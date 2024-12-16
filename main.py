import os
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
from lib.core import sensor_work
import parameters as pm


def main():
    # Check if the user provided exactly 2 arguments (IP and Port)
    if len(sys.argv) == 1:
        # Call function
        sensor_work()
    else:
        print("Usage: << python main.py >> ")
        sys.exit(1)


if __name__ == "__main__":
    main()
