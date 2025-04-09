#!/bin/bash

# Check if the workspace has already been built
if [ ! -d "/workspace/devel" ] || [ ! -d "/workspace/build" ]; then
    echo "Catkin workspace not built yet. Building now..."
    source /opt/ros/noetic/setup.bash
    cd /workspace
    catkin build
    echo "Workspace built successfully!"
else
    echo "Catkin workspace already built. Skipping build step."
fi

# Make all Python scripts executable
echo "Making Python scripts executable..."
find /workspace/src -name "*.py" -exec chmod +x {} \;

# Make sure the input and output directories exist
echo "Ensuring input and output directories exist..."
mkdir -p /workspace/src/gaze_detection/input
mkdir -p /workspace/src/gaze_detection/output

echo "Setup complete!"

