"""
Helper script to start the Streamlit UI with the correct configuration.
This ensures the fileWatcherType setting is applied properly.
"""
import os
import subprocess

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(project_dir, ".streamlit")
web_ui_path = os.path.join(project_dir, "web_ui.py")

# Verify the config file exists
config_path = os.path.join(config_dir, "config.toml")
if not os.path.exists(config_path):
    print(f"Config file not found at {config_path}")
    print("Creating config directory and file...")
    os.makedirs(config_dir, exist_ok=True)
    with open(config_path, "w") as f:
        f.write("[server]\nfileWatcherType = \"none\"\n")
    print("Config file created.")

# Print information
print(f"Project directory: {project_dir}")
print(f"Config directory: {config_dir}")
print(f"Config file: {config_path}")
print(f"Web UI path: {web_ui_path}")

# Start Streamlit with explicit config path
print("\nStarting Streamlit with custom configuration...")
cmd = ["streamlit", "run", web_ui_path, "--server.fileWatcherType=none"]
print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd)
