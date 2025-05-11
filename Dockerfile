FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04 AS base

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set locale
RUN apt-get update && apt-get install -y locales
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install system dependencies first
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    gnupg2 \
    lsb-release \
    git \
    wget \
    vim \
    tmux \
    nano \
    libvips42 \
    libvips-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.8 (default from Ubuntu 20.04) and Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install pip for both Python versions
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Create virtual environments for both Python versions
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8-venv \
    python3.9-venv \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environments
RUN python3.8 -m venv /opt/venv_py38 && \
    python3.9 -m venv /opt/venv_py39

# Make Python 3.8 the default Python for ROS
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Install ROS Noetic (which will use Python 3.8)
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    build-essential \
    python3-catkin-tools \
    python3-osrf-pycommon \
    ros-noetic-cv-bridge \
    ros-noetic-rgbd-launch \
    ros-noetic-tf2-sensor-msgs \
    python3-netifaces \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

# Configure environment 
SHELL ["/bin/bash", "-c"]

# Install Python 3.8 dependencies for ROS
RUN . /opt/venv_py38/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install netifaces opencv-python && \
    deactivate

# 1) Install OpenCV runtime libs so the manylinux wheel works at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 2) Activate the venv, bump pip & friends, install everything else in one go…
RUN . /opt/venv_py39/bin/activate && \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install \
    numpy>=1.24.0 \
    torchvision==0.16.2 \
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
    gradio==4.38.1 && \
    deactivate

# 3) …then install torch via the PyTorch index…
RUN . /opt/venv_py39/bin/activate && \
    pip install torch==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 && \
    deactivate

# 4) Install small extras
RUN . /opt/venv_py39/bin/activate && \
    pip install datasets==3.1.0 editdistance==0.8.1 rospkg catkin_pkg netifaces && \
    deactivate

# Create and set up the catkin workspace structure
WORKDIR /workspace
RUN mkdir -p /workspace/src

# Install libfreenect
RUN mkdir -p ~/external_repos && \
    cd ~/external_repos && \
    git clone https://github.com/OpenKinect/libfreenect.git && \
    cd libfreenect && \
    mkdir build && \
    cd build && \
    cmake -L .. && \
    make -j 4 && \
    make install


# Clone the freenect ROS packages into the workspace
WORKDIR /workspace/src
RUN git clone https://github.com/ros-drivers/freenect_stack.git

WORKDIR /workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    catkin init && \
    catkin config --extend /opt/ros/noetic && \
    catkin build"

# Create basic wrapper scripts for Python environments
RUN echo '#!/bin/bash\nsource /opt/venv_py38/bin/activate\nexec "$@"\n' > /usr/local/bin/with_py38 && \
    echo '#!/bin/bash\nsource /opt/venv_py39/bin/activate\nexec "$@"\n' > /usr/local/bin/with_py39 && \
    chmod +x /usr/local/bin/with_py38 /usr/local/bin/with_py39

# Add aliases to bashrc for convenience
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    echo 'alias py38="with_py38 python3"' >> ~/.bashrc && \
    echo 'alias py39="with_py39 python3"' >> ~/.bashrc && \
    echo 'if [ -f /workspace/devel/setup.bash ]; then source /workspace/devel/setup.bash; fi' >> ~/.bashrc

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