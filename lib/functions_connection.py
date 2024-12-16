import socket


def establish_connection(RPI_IP, RPI_PORT):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((RPI_IP, RPI_PORT))
        print(f"Connection stablished {RPI_IP,RPI_PORT}")
        return s
    except (socket.error, socket.timeout) as e:
        print("Error in establish connection ", e)
