#!/bin/bash
# WSL2 Ubuntu - IMPROVED 5 Class Eye Disease Training Script
# RTX 5070 GPU ile iyileştirilmiş eğitim başlatma

echo "======================================"
echo "IMPROVED 5 CLASS EYE DISEASE TRAINING"
echo "GPU: RTX 5070"
echo "======================================"
echo ""

# Set GPU library path (KRİTİK!)
export LD_LIBRARY_PATH=/home/melih/miniconda3/envs/tf_gpu/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Conda environment'ı aktive et
echo "[1/5] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

# GPU kontrolü
echo ""
echo "[2/5] Checking GPU..."
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU Count: {len(gpus)}'); [print(f'  - {gpu}') for gpu in gpus]" || {
    echo "ERROR: GPU not detected!"
    exit 1
}

# Dizine git
echo ""
echo "[3/5] Navigating to project directory..."
cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier || {
    echo "ERROR: Directory not found!"
    exit 1
}

# Python version kontrolü
echo ""
echo "[4/5] Checking Python environment..."
python3 --version
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Eğitimi başlat
echo ""
echo "[5/5] Starting improved training..."
echo "======================================"
echo "Training will begin shortly..."
echo "Press Ctrl+C to stop (not recommended during training)"
echo "======================================"
echo ""

python3 train_mendeley_eye_5class_improved.py 2>&1 | tee training_improved_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "======================================"
echo "Training completed!"
echo "======================================"


