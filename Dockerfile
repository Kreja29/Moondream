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
    cmake \
    libusb-1.0-0-dev \
    freeglut3-dev \
    libxmu-dev \
    libxi-dev \
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
RUN curl -s https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip-3.8.py && \
    curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip-3.9.py && \
    python3.8 get-pip-3.8.py && \
    python3.9 get-pip-3.9.py && \
    rm get-pip-3.8.py get-pip-3.9.py

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
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list


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
    ros-noetic-camera-info-manager \
    ros-noetic-image-transport \
    ros-noetic-dynamic-reconfigure \
    python3-netifaces \
    python3-opencv \
    python3-rospy \
    && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

# Configure environment 
SHELL ["/bin/bash", "-c"]

# Install Python 3.8 dependencies for ROS
RUN . /opt/venv_py38/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install netifaces opencv-python open3d && \
    deactivate

# Install OpenCV runtime libs so the manylinux wheel works at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install PyKDL dependency
RUN apt-get update && \
    apt-get install -y python3-pykdl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python 3.9 dependencies
RUN source /opt/venv_py39/bin/activate && \
    pip install --upgrade pip setuptools wheel && \
    pip install \
        numpy>=1.24.0 \
        torchvision==0.16.2 \
        opencv-python>=4.8.0 \
        transformers==4.44.0 \
        safetensors \
        accelerate==0.32.1 \
        pillow==10.0.0 \
        matplotlib>=3.7.0 \
        einops \
        tqdm>=4.65.0 \
        pyvips==2.2.3 \
        pyvips-binary==8.16.0 \
        huggingface-hub==0.24.0 \
        gradio==4.38.1 \
        open3d \
        scipy \
        albumentations==1.3.1 \
        certifi==2023.7.22 \
        charset-normalizer==3.3.1 \
        coloredlogs==15.0.1 \
        cmake==3.25.0 \
        cython==3.0.4 \
        easydict==1.11 \
        flatbuffers==23.5.26 \
        humanfriendly==10.0 \
        idna==3.4 \
        imageio==2.31.6 \
        insightface==0.7.3 \
        joblib==1.3.2 \
        lazy-loader==0.3 \
        lit==15.0.7 \
        mpmath==1.3.0 \
        networkx==3.1 \
        onnx==1.14.1 \
        onnxruntime-gpu==1.15.0 \
        opencv-python-headless==4.8.1.78 \
        prettytable==3.9.0 \
        pywavelets==1.4.1 \
        pyyaml==6.0.1 \
        qudida==0.0.4 \
        scikit-image==0.21.0 \
        scikit-learn==1.3.2 \
        sympy==1.12 \
        threadpoolctl==3.2.0 \
        tifffile==2023.7.10 \
        urllib3==2.0.7 \
        wcwidth==0.2.8 && \
    pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install datasets==3.1.0 editdistance==0.8.1 rospkg catkin_pkg netifaces && \
    deactivate
   
RUN apt-get update && \
    apt-get install -y usbutils && \
    rm -rf /var/lib/apt/lists/*

# Install libfreenect
RUN mkdir -p external_repos && \
    cd external_repos && \
    git clone https://github.com/OpenKinect/libfreenect.git && \
    cd libfreenect && \
    mkdir build && cd build && \
    cmake -L .. && \
    make -j4 && \
    make install

# Create and set up the catkin workspace structure
WORKDIR /workspace
RUN mkdir -p src logs

# Clone the freenect ROS packages and rosnumpy into the workspace
WORKDIR /workspace/src
RUN git clone https://github.com/Kreja29/freenect_stack.git
RUN git clone https://github.com/eric-wieser/ros_numpy.git


WORKDIR /workspace/src
RUN git clone https://github.com/eververas/3DGazeNet.git


# First catkin config and build
WORKDIR /workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    catkin init && \
    catkin config --extend /opt/ros/noetic && \
    catkin config --merge-devel && \
    catkin config --cmake-args \
        -DCMAKE_BUILD_TYPE=Release && \
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