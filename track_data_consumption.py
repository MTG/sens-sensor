import time
import matplotlib.pyplot as plt
import signal
import sys

INTERFACE="wlx3c52a1860ac4"

# Global data stores
timestamps = []
downloaded_data = []
uploaded_data = []
total_data = []

def get_network_data(interface=INTERFACE):
    with open('/proc/net/dev', 'r') as f:
        data = f.readlines()

    for line in data:
        if interface in line:
            parts = line.split()
            received_bytes = int(parts[1])
            transmitted_bytes = int(parts[9])
            return received_bytes, transmitted_bytes
    return 0, 0

def create_graphs():
    total_downloaded = sum(downloaded_data)
    total_uploaded = sum(uploaded_data)
    total_consumed = total_downloaded + total_uploaded

    # Convert to MB
    downloaded_mb = [d / (1024**2) for d in downloaded_data]
    uploaded_mb = [u / (1024**2) for u in uploaded_data]
    total_mb = [t / (1024**2) for t in total_data]

    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, downloaded_mb, label='Downloaded (MB)', color='blue')
    plt.plot(timestamps, uploaded_mb, label='Uploaded (MB)', color='green')
    plt.plot(timestamps, total_mb, label='Total (MB)', color='orange')
    plt.xlabel('Time')
    plt.ylabel('Data (MB)')
    plt.title('Internet Data Usage Over Time')
    plt.legend()

    # Add summary textbox
    summary_text = (
        f"Total Downloaded: {total_downloaded / (1024**2):.2f} MB\n"
        f"Total Uploaded: {total_uploaded / (1024**2):.2f} MB\n"
        f"Total: {total_consumed / (1024**2):.2f} MB"
    )
    plt.gcf().text(0.72, 0.55, summary_text, fontsize=10,
                   bbox=dict(facecolor='lightgrey', alpha=0.5))

    # Save graph
    plt.tight_layout()
    plt.savefig("data_usage_summary.png")
    print("📊 Graph saved as 'data_usage_summary.png'")
    plt.show()

def signal_handler(sig, frame):
    print("\nStopping data monitor and generating graph...")
    create_graphs()
    sys.exit(0)

def monitor_data(interface=INTERFACE, interval=10):
    print(f"Monitoring interface: {interface}")
    prev_received, prev_transmitted = get_network_data(interface)
    
    signal.signal(signal.SIGINT, signal_handler)

    while True:
        time.sleep(interval)
        new_received, new_transmitted = get_network_data(interface)

        delta_received = new_received - prev_received
        delta_transmitted = new_transmitted - prev_transmitted
        total = delta_received + delta_transmitted

        timestamp = time.strftime('%H:%M:%S')
        print(f"[{timestamp}] Downloaded: {delta_received / 1024:.2f} KB, "
              f"Uploaded: {delta_transmitted / 1024:.2f} KB, "
              f"Total: {total / 1024:.2f} KB")

        timestamps.append(timestamp)
        downloaded_data.append(delta_received)
        uploaded_data.append(delta_transmitted)
        total_data.append(total)

        prev_received, prev_transmitted = new_received, new_transmitted

# 👇 Start monitoring
monitor_data(interface=INTERFACE, interval=10)

