#!/bin/bash
# GPU Library Path Düzeltme Scripti

echo "======================================"
echo "FIXING GPU LIBRARY PATH"
echo "======================================"
echo ""

# Conda environment path
CONDA_ENV_PATH="/home/melih/miniconda3/envs/tf_gpu"

# CUDA libraries path
CUDA_LIB_PATH="/usr/local/cuda/lib64"

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CONDA_ENV_PATH}/lib:${CUDA_LIB_PATH}:$LD_LIBRARY_PATH

echo "[1] LD_LIBRARY_PATH set:"
echo "  $LD_LIBRARY_PATH"
echo ""

# Test GPU
echo "[2] Testing GPU detection..."
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU Count: {len(gpus)}'); [print(f'  - {gpu}') for gpu in gpus]"

echo ""
echo "======================================"
echo "If GPU: 1 appears above, problem solved!"
echo "======================================"


