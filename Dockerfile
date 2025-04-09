FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 AS base

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set locale
RUN apt-get update && apt-get install -y locales
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install Python 3.9
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.9 the default Python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 && \
    update-alternatives --set python3 /usr/bin/python3.9

# Install pip for Python 3.9
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Install ROS Noetic
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    python3-osrf-pycommon \
    ros-noetic-cv-bridge \
    && rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && apt-get install -y --no-install-recommends \  
    python3-netifaces \
    && rm -rf /var/lib/apt/lists/*


# Initialize rosdep
RUN rosdep init && rosdep update

# Configure environment
SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-opencv \
    git \
    wget \
    vim \
    tmux \
    nano \
    libvips42 \
    libvips-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN pip3 install --upgrade pip

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install --upgrade importlib_metadata

RUN pip3 install netifaces

# Install CPU dependencies first
RUN pip3 install \
    numpy>=1.24.0 \
    torchvision==0.19.1 \
    opencv-python>=4.8.0 \
    transformers==4.44.0 \
    safetensors \
    accelerate==0.32.1 \
    pillow==10.4.0 \
    matplotlib>=3.7.0 \
    einops \
    tqdm>=4.65.0 \
    pyvips==2.2.3 \
    pyvips-binary==8.16.0 \
    huggingface-hub==0.24.0 \
    gradio==4.38.1

# Install PyTorch with CUDA support (will be used if GPU is available)
RUN pip3 install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install datasets==3.1.0 editdistance==0.8.1

# Create and set up the catkin workspace
WORKDIR /workspace
RUN mkdir -p /workspace/src

# Copy your gaze_detection package to the workspace
RUN mkdir -p /workspace/src/gaze_detection/scripts
RUN mkdir -p /workspace/src/gaze_detection/launch
RUN mkdir -p /workspace/src/gaze_detection/input
RUN mkdir -p /workspace/src/gaze_detection/output

# Set up the workspace
WORKDIR /workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    catkin init && \
    catkin config --extend /opt/ros/noetic && \
    catkin build"

# Add setup files to bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo "source /workspace/devel/setup.bash" >> ~/.bashrc

# Create GPU detection script
COPY ./detect_gpu.sh /detect_gpu.sh
RUN chmod +x /detect_gpu.sh

# Set up entry point
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the working directory
WORKDIR /workspace

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]