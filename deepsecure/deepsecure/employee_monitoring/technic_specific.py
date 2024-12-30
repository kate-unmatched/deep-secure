import psutil
import time
from datetime import datetime, timedelta
from collections import defaultdict
import requests
from bs4 import BeautifulSoup

# Variables to store application usage data
app_usage = defaultdict(lambda: {"start_time": None, "total_time": timedelta(), "launch_count": 0})

# Placeholder for corporate system interaction logs
corporate_system_logs = []

# Placeholder for visited websites
web_activity = defaultdict(lambda: {"visit_count": 0, "total_time": timedelta(), "last_visited": None, "text_snippets": []})

def track_applications(interval=5):
    """Track active applications and their usage."""
    print("Tracking application usage. Press Ctrl+C to stop.\n")
    try:
        while True:
            # Get all running processes
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    active_name = proc.info['name']

                    # If the app is not being tracked, initialize its data
                    if app_usage[active_name]["start_time"] is None:
                        app_usage[active_name]["start_time"] = datetime.now()
                        app_usage[active_name]["launch_count"] += 1

                    # Update total time spent on the application
                    app_usage[active_name]["total_time"] += timedelta(seconds=interval)

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass

            # Simulate corporate system interaction logging
            log_corporate_system_interaction()

            # Simulate website tracking
            simulate_web_activity()

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopping tracking...")
        display_report()

def log_corporate_system_interaction():
    """Simulate and log interaction with corporate systems."""
    # Simulating log of a CRM system interaction
    now = datetime.now()
    corporate_system_logs.append({
        "system": "CRM",
        "action": "Login",
        "timestamp": now
    })
    print(f"Logged interaction with CRM system at {now}")

def simulate_web_activity():
    """Simulate tracking of visited websites."""
    visited_sites = [
        {"url": "https://example.com", "duration": 10},
        {"url": "https://example.org", "duration": 15}
    ]

    for site in visited_sites:
        url = site["url"]
        duration = site["duration"]

        if url not in web_activity:
            web_activity[url]["visit_count"] = 0
            web_activity[url]["total_time"] = timedelta()

        web_activity[url]["visit_count"] += 1
        web_activity[url]["total_time"] += timedelta(seconds=duration)
        web_activity[url]["last_visited"] = datetime.now()

        # Extract a snippet from the website and save it
        snippet = fetch_website_snippet(url)
        if snippet:
            web_activity[url]["text_snippets"].append(snippet)

def fetch_website_snippet(url):
    """Fetch and return a snippet of text from a website."""
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all('p')
        if paragraphs:
            return paragraphs[0].text.strip()
    except Exception as e:
        print(f"Error fetching snippet from {url}: {e}")
    return None

def save_snippets_to_file():
    """Save website snippets to files."""
    for url, data in web_activity.items():
        file_name = url.replace("https://", "").replace("/", "_") + ".txt"
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(f"Website: {url}\n")
            file.write(f"Visit Count: {data['visit_count']}\n")
            file.write(f"Total Time: {data['total_time']}\n")
            file.write(f"Last Visited: {data['last_visited']}\n\n")
            file.write("Snippets:\n")
            file.writelines(f"- {snippet}\n" for snippet in data["text_snippets"])

def display_report():
    """Display a report of application usage, system interactions, and web activity."""
    print("\nApplication Usage Report:")
    print(f"{'Application':<20} {'Total Time':<20} {'Launch Count':<15}")
    print("-" * 60)
    for app, data in app_usage.items():
        total_time = str(data['total_time'])
        print(f"{app:<20} {total_time:<20} {data['launch_count']:<15}")

    print("\nCorporate System Interaction Logs:")
    print(f"{'System':<15} {'Action':<15} {'Timestamp':<20}")
    print("-" * 50)
    for log in corporate_system_logs:
        print(f"{log['system']:<15} {log['action']:<15} {log['timestamp']:<20}")

    print("\nWebsite Activity Report:")
    print(f"{'URL':<40} {'Visit Count':<15} {'Total Time':<15} {'Last Visited':<20}")
    print("-" * 90)
    for url, data in web_activity.items():
        print(f"{url:<40} {data['visit_count']:<15} {data['total_time']:<15} {data['last_visited']:<20}")

    # Save snippets to files
    save_snippets_to_file()

if __name__ == "__main__":
    track_applications(interval=5)