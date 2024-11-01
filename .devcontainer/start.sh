#!/bin/bash

# Start Xvfb
Xvfb :99 -screen 0 1280x720x24 &

# Export DISPLAY variable
export DISPLAY=:99

# Start JupyterLab without token authentication
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''