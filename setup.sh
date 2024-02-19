#!/bin/bash

# Detect the operating system
os=$(uname)

if [ "$os" = "Linux" ]; then
    echo "Updating package lists..."
    apt-get update -y
    echo "Installing dependencies..."
    apt-get install -y xvfb
    apt-get install libxrender1
    apt-get install libxi6 libgconf-2-4
    apt-get install libxkbcommon-x11-0
    apt-get install -y libgl1-mesa-glx

    echo "Installing Blender-4.0.2..."
    wget https://ftp.nluug.nl/pub/graphics/blender//release/Blender4.0/blender-4.0.2-linux-x64.tar.xz && tar -xf blender-4.0.2-linux-x64.tar.xz && rm blender-4.0.2-linux-x64.tar.xz
else
    echo "Not a Linux system. Skipping Blender installation. Please download and install blender manually ONLY for data rendering purpose"
fi

# Check if requirements.txt exists and then install dependencies
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found."
fi
