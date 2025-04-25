#!/bin/bash

echo "Checking environment setup..."

# Check Python versions
echo -n "Python 3.8: "
with_py38 python3 --version

echo -n "Python 3.9: "
with_py39 python3 --version

# Check ROS installation
echo -n "ROS Noetic: "
if [ -d "/opt/ros/noetic" ]; then
  echo "Installed"
else
  echo "Not found"
fi

# Check if virtual environments exist
echo -n "Python 3.8 venv: "
if [ -d "/opt/venv_py38" ]; then
  echo "Found"
else
  echo "Not found"
fi

echo -n "Python 3.9 venv: "
if [ -d "/opt/venv_py39" ]; then
  echo "Found"
else
  echo "Not found"
fi

# Check for wrapper scripts
echo -n "Python wrapper scripts: "
if [ -f "/usr/local/bin/with_py38" ] && [ -f "/usr/local/bin/with_py39" ]; then
  echo "Found"
else
  echo "Not found"
fi

# Check GPU availability
echo "Checking GPU availability..."
/detect_gpu.sh

echo "Setup check complete."