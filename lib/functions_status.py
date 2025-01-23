import psutil
import socket
from datetime import datetime, timedelta


def get_memory_info():
    memory = psutil.virtual_memory()
    return {
        # "total_memory": memory.total,
        # "used_memory": memory.used,
        # "available_memory": memory.available,
        "percent_used": memory.percent
    }


def get_cpu_info():
    return {
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "cpu_count": psutil.cpu_count(logical=True),
    }


def get_temperature():
    try:
        # Read from the thermal zone file (common for Raspberry Pi)
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read()) / 1000.0  # Convert millidegrees to Celsius
        return {"temperature_celsius": temp}
    except FileNotFoundError:
        return {"temperature_celsius": None}


def get_network_info():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    interfaces = psutil.net_if_addrs()
    interface_info = {
        iface: [addr.address for addr in addrs if addr.family == socket.AF_INET]
        for iface, addrs in interfaces.items()
    }
    return {
        "hostname": hostname,
        "ip_address": ip_address,
        "interfaces": interface_info,
    }


# Additional useful information
def get_uptime():
    boot_time = psutil.boot_time()  # Boot time as a timestamp
    current_time = datetime.now().timestamp()  # Current time as a timestamp
    uptime_seconds = int(current_time - boot_time)  # Calculate uptime in seconds
    uptime = str(
        timedelta(seconds=uptime_seconds)
    )  # Convert seconds to hh:mm:ss format
    return {
        # "uptime_seconds": uptime_seconds,
        "uptime_readable": uptime
    }


def get_disk_usage():
    disk = psutil.disk_usage("/")
    return {
        # "total_disk_space": disk.total,
        # "used_disk_space": disk.used,
        # "free_disk_space": disk.free,
        "percent_used": disk.percent
    }


# Gather all info
def gather_raspberry_pi_info():
    return {
        "memory_info": get_memory_info(),
        "cpu_info": get_cpu_info(),
        "temperature": get_temperature(),
        "network_info": get_network_info(),
        "uptime": get_uptime(),
        "disk_usage": get_disk_usage(),
    }
