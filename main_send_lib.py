import os
import sys

# Path importing
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(src_dir)

# Imports from this project
import parameters as pm
from lib.functions_send import send_library


def main():
    if len(sys.argv) == 1:
        # Call function
        send_library()


if __name__ == "__main__":
    main()
