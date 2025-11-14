#!/bin/bash
# WSL2 Ubuntu - 5 Class Eye Disease Training Script
# RTX 5070 GPU ile eğitim başlatma

echo "======================================"
echo "5 CLASS EYE DISEASE TRAINING"
echo "GPU: RTX 5070"
echo "======================================"
echo ""

# Set GPU library path (ÖNEMLİ!)
export LD_LIBRARY_PATH=/home/melih/miniconda3/envs/tf_gpu/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Conda environment'ı aktive et
echo "[1/4] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

# GPU kontrolü
echo ""
echo "[2/4] Checking GPU..."
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU Count: {len(gpus)}'); [print(f'  - {gpu}') for gpu in gpus]"

# Dizine git
echo ""
echo "[3/4] Navigating to project directory..."
cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier

# Eğitimi başlat
echo ""
echo "[4/4] Starting training..."
echo "Training will be saved to: training_5class_live.log"
echo ""
echo "======================================"
echo "TRAINING STARTED"
echo "To monitor progress: tail -f training_5class_live.log"
echo "To stop: Ctrl+C"
echo "======================================"
echo ""

# Eğitimi başlat ve log'a kaydet
python3 train_mendeley_eye_5class.py 2>&1 | tee training_5class_live.log

echo ""
echo "Training completed!"

