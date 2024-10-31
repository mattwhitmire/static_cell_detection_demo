#!/bin/bash

# Start Xvfb
Xvfb :99 -screen 0 1280x720x24 &

# Export DISPLAY variable
export DISPLAY=:99

# Start JupyterLab
jupyter lab --ip 0.0.0.0 --no-browser --allow-root

# # Wait for JupyterLab to start
# sleep 5

# # Retrieve the token
# TOKEN=$(jupyter server list | grep -oP 'token=\K[^&]+')

# # Construct the URL
# URL="http://localhost:8888/lab?token=$TOKEN"

# # Print the URL for user
# echo "JupyterLab token URL: $URL"