import os
import psutil
from datetime import datetime, timedelta
from collections import defaultdict

default_file_data = lambda: {"upload_count": 0, "copy_count": 0, "delete_count": 0}
file_activity = defaultdict(default_file_data)

device_usage = {
    "external_devices": 0,
    "printers": 0,
    "audio_video_usage": 0
}

network_activity = {
    "traffic_volume": 0,
    "active_connections": 0,
    "unauthorized_networks": []
}

# Placeholder for productivity metrics
productivity_metrics = {
    "task_duration": timedelta(),
    "breaks": 0,
    "activity_coefficient": 0.0
}

def monitor_file_activity():
    """Simulate monitoring file activities."""
    print("Monitoring file activities...")
    # Simulated file actions
    simulated_files = [
        {"action": "upload", "file_name": "example.txt"},
        {"action": "delete", "file_name": "data.csv"},
        {"action": "copy", "file_name": "report.docx"}
    ]
    for file in simulated_files:
        action = file["action"]
        file_name = file["file_name"]
        file_activity[file_name][f"{action}_count"] += 1
        print(f"File {action}: {file_name}")

def monitor_device_usage():
    """Simulate monitoring external devices and printers."""
    print("Monitoring device usage...")
    # Simulate device usage
    device_usage["external_devices"] += 1  # Example: USB device connected
    device_usage["printers"] += 1          # Example: Printer job
    device_usage["audio_video_usage"] += 1 # Example: Microphone or camera usage

def monitor_network_activity():
    """Simulate monitoring network activities."""
    print("Monitoring network activities...")
    # Simulate network activity
    network_activity["traffic_volume"] += 500
    network_activity["active_connections"] += 2
    network_activity["unauthorized_networks"].append("Unknown_WiFi")  # Example unauthorized network

def monitor_productivity():
    """Simulate monitoring productivity metrics."""
    print("Monitoring productivity metrics...")
    # Simulated productivity data
    productivity_metrics["task_duration"] += timedelta(minutes=30)
    productivity_metrics["breaks"] += 1
    # Simulate activity coefficient calculation
    total_minutes = 480  # Example: 8-hour workday
    active_minutes = 400 # Simulated active time
    productivity_metrics["activity_coefficient"] = active_minutes / total_minutes

def display_report():
    """Display a report of monitored activities."""
    print("\nFile Activity Report:")
    print(f"{'File Name':<20} {'Uploads':<10} {'Copies':<10} {'Deletes':<10}")
    print("-" * 50)
    for file_name, data in file_activity.items():
        print(f"{file_name:<20} {data['upload_count']:<10} {data['copy_count']:<10} {data['delete_count']:<10}")

    print("\nDevice Usage Report:")
    print(f"External Devices: {device_usage['external_devices']}")
    print(f"Printers Used: {device_usage['printers']}")
    print(f"Audio/Video Usage: {device_usage['audio_video_usage']}")

    print("\nNetwork Activity Report:")
    print(f"Traffic Volume: {network_activity['traffic_volume']} MB")
    print(f"Active Connections: {network_activity['active_connections']}")
    print(f"Unauthorized Networks: {', '.join(network_activity['unauthorized_networks'])}")

    print("\nProductivity Report:")
    print(f"Task Duration: {productivity_metrics['task_duration']}")
    print(f"Breaks Taken: {productivity_metrics['breaks']}")
    print(f"Activity Coefficient: {productivity_metrics['activity_coefficient']:.2f}")

if __name__ == "__main__":
    print("Starting monitoring...")
    monitor_file_activity()
    monitor_device_usage()
    monitor_network_activity()
    monitor_productivity()
    display_report()
