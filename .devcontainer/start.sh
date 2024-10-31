#!/bin/bash

# Start Xvfb
Xvfb :99 -screen 0 1280x720x24 &

# Export DISPLAY variable
export DISPLAY=:99

# Start JupyterLab
jupyter --ip 0.0.0.0 --no-browser --allow-root