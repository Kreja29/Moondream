#!/bin/bash

# Check if nvidia-smi is available and working
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected - using CUDA"
    export CUDA_VISIBLE_DEVICES=0
    # Add any other GPU-specific environment variables here
    export USE_GPU=1
    
    # Set ROS GPU-specific environment variables
    export ROS_PARALLEL_JOBS="-l6 -j4"  # Limit parallel jobs for better GPU memory management
else
    echo "No NVIDIA GPU detected or NVIDIA drivers not properly installed - using CPU only"
    export CUDA_VISIBLE_DEVICES=""
    # Force PyTorch to use CPU
    export USE_GPU=0
    
    # Set ROS CPU-specific environment variables
    export ROS_PARALLEL_JOBS="-l4 -j2"  # Limit parallel jobs for CPU
fi

# Echo the configuration for verification
if [ "$USE_GPU" -eq 1 ]; then
    echo "Configuration: Using GPU with CUDA"
    # Get GPU information
    nvidia-smi
    
    # Configure PyTorch to use smaller batches on GPU
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
else
    echo "Configuration: Using CPU only"
    
    # Configure PyTorch for CPU
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
fi