#!/bin/bash
# Bone Disease Training - Improved Version
# Activate tf_gpu environment and run training

echo "======================================"
echo "Bone Disease Training - Improved"
echo "======================================"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate tf_gpu environment
echo "[1/3] Activating tf_gpu environment..."
conda activate tf_gpu

# Check GPU
echo ""
echo "[2/3] Checking GPU and packages..."
python3 -c "
import tensorflow as tf
import numpy as np
print(f'TensorFlow: {tf.__version__}')
print(f'NumPy: {np.__version__}')
print(f'GPU Available: {len(tf.config.list_physical_devices(\"GPU\"))}')
"

# Change to project directory
cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier

# Run training
echo ""
echo "[3/3] Starting improved training..."
echo "======================================"
echo ""
python3 train_bone_4class_improved.py

echo ""
echo "======================================"
echo "Training completed!"
echo "======================================"

