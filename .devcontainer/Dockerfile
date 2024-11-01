# Use a base image with Python and other dependencies
FROM python:3.12-slim

# Install necessary packages
RUN apt-get update && apt-get install -y \ 
    xvfb \
    libxi6 \
    libgconf-2-4 \
    libnss3 \
    libxrender1 \
    libxrandr2 \
    libxtst6 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon-x11-0 \
    fonts-liberation \
    libappindicator3-1 \
    libu2f-udev \
    libv4l-dev \
    xdg-utils \
    unzip \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

# Disable Jupyter token and password authentication
RUN mkdir -p ~/.jupyter && \
    echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> ~/.jupyter/jupyter_notebook_config.py

# Copy the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set the working directory
WORKDIR /app

# Expose the port JupyterLab will run on
EXPOSE 8888

# Command to run the startup script
CMD ["/app/start.sh"]
