#!/bin/bash
# Save this as src/gaze_detection/scripts/run_gaze_detection_wrapper.sh

# Source the ROS environment
source /opt/ros/noetic/setup.bash
if [ -f /workspace/devel/setup.bash ]; then
  source /workspace/devel/setup.bash
fi

# Run the Python script with Python 3.9
/usr/local/bin/with_py39 python3 /workspace/src/gaze_detection/scripts/gaze_detection_input_output.py "$@"