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
    # Install Google Chrome
    gnupg \
    curl \
    && curl -fsSL https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | tee /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update && apt-get install -y google-chrome-stable git vim \
    && rm -rf /var/lib/apt/lists/*

# Download and extract ChromeDriver
RUN wget https://storage.googleapis.com/chrome-for-testing-public/127.0.6533.88/linux64/chromedriver-linux64.zip -O /tmp/chromedriver.zip \
    && unzip /tmp/chromedriver.zip -d /tmp/chromedriver \
    && mv /tmp/chromedriver/chromedriver-linux64/chromedriver /usr/local/bin/ \
    && chmod +x /usr/local/bin/chromedriver \
    && rm /tmp/chromedriver.zip \
    && rm -rf /tmp/chromedriver

# Verify ChromeDriver installation
RUN ls -l /usr/local/bin/chromedriver

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt


# Copy the startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Set the working directory
WORKDIR /app

# Expose the port JupyterLab will run on
EXPOSE 8888

# Command to run the startup script
CMD ["/app/start.sh"]

# ENV DISPLAY=:99

# CMD ["sh", "-c", "Xvfb :99 -screen 0 1280x720x24 & jupyter lab --ip 0.0.0.0 --no-browser --allow-root"]
