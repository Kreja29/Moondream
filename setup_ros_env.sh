#!/bin/bash

echo "Setting up ROS environment..."

# Source ROS setup
source /opt/ros/noetic/setup.bash

# Print Python version
echo "Python version:"
python3 --version

echo "Rebuilding catkin workspace..."
cd /workspace
catkin clean -y
catkin init
catkin config --extend /opt/ros/noetic

# Make sure CMakeLists.txt exists for the package
if [ ! -f "/workspace/src/gaze_detection/CMakeLists.txt" ]; then
    echo "Creating CMakeLists.txt for gaze_detection package..."
    cat > /workspace/src/gaze_detection/CMakeLists.txt << 'EOL'
cmake_minimum_required(VERSION 3.0.2)
project(gaze_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/scripts
)
EOL
fi

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

echo "ROS environment setup complete" 