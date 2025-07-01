#!/bin/bash
set -e

# Source GPU detection script
source /detect_gpu.sh

# Source ROS setup
source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash

export PYTHONPATH=$PYTHONPATH:/workspace/src/3DGazeNet/demo:/workspace/src/gaze_detection/scripts

# Print GPU/CPU configuration message
if [ "$USE_GPU" -eq 1 ]; then
    echo " Running with NVIDIA CUDA GPU acceleration"
    echo " ROS environment ready with GPU support"
else
    echo " Running on CPU only"
    echo " ROS environment ready with CPU support"
fi

# Display ROS environment information
echo "ROS_DISTRO: $ROS_DISTRO"
echo "ROS_ROOT: $ROS_ROOT"
echo "ROS_PACKAGE_PATH: $ROS_PACKAGE_PATH"

# Execute the command passed to docker run
exec "$@"