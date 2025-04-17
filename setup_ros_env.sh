#!/bin/bash

echo "Setting up ROS environment..."

# Source ROS setup
source /opt/ros/noetic/setup.bash

echo "Rebuilding catkin workspace..."
cd /workspace
catkin clean -y
catkin init
catkin config --extend /opt/ros/noetic

# Build the workspace
echo "Building catkin workspace..."
catkin build

# Source the workspace setup file
source /workspace/devel/setup.bash

# Verify ROS package path includes the workspace
echo "ROS_PACKAGE_PATH: $ROS_PACKAGE_PATH"

# Fix Windows line endings in Python files
echo "Fixing line endings in Python files..."
find /workspace/src -name "*.py" -exec sed -i 's/\r$//' {} \;

# Make all Python scripts executable
echo "Making Python scripts executable..."
find /workspace/src -name "*.py" -exec chmod +x {} \;

# Make sure scripts are executable
chmod +x /workspace/src/gaze_detection/scripts/run_gaze_detection_wrapper.sh
chmod +x /workspace/src/gaze_detection/scripts/gaze_detection_input_output.py

echo "ROS environment setup complete" 