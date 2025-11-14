#!/bin/bash
# WSL2 Ubuntu - Bone Disease 4 Class Training Script
# GPU: Windows'ta kurulu NVIDIA GPU kullanılır

echo "======================================"
echo "BONE DISEASE DETECTION - 4 CLASS TRAINING"
echo "WSL2 GPU Training"
echo "======================================"
echo ""

# Set GPU library path (WSL için KRİTİK!)
export LD_LIBRARY_PATH=/home/melih/miniconda3/envs/tf_gpu/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Conda environment'ı aktive et
echo "[1/6] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

# GPU kontrolü
echo ""
echo "[2/6] Checking GPU..."
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU Count: {len(gpus)}'); [print(f'  - {gpu}') for gpu in gpus]" || {
    echo "ERROR: GPU not detected!"
    echo "Check:"
    echo "  1. NVIDIA driver installed on Windows"
    echo "  2. WSL2 CUDA support enabled"
    echo "  3. TensorFlow GPU version installed"
    exit 1
}

# Dataset kontrolü
echo ""
echo "[3/6] Checking dataset..."
DATASET_PATH="/mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier/datasets/bone/Bone_4Class_Final"
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found at: $DATASET_PATH"
    echo "Please run organize_bone_4class_final.py first!"
    exit 1
fi

for split in train val test; do
    if [ ! -d "$DATASET_PATH/$split" ]; then
        echo "ERROR: Missing $split directory"
        exit 1
    fi
done

TRAIN_COUNT=$(find "$DATASET_PATH/train" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
VAL_COUNT=$(find "$DATASET_PATH/val" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
TEST_COUNT=$(find "$DATASET_PATH/test" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)

echo "  Train images: $TRAIN_COUNT"
echo "  Val images: $VAL_COUNT"
echo "  Test images: $TEST_COUNT"

# Dizine git
echo ""
echo "[4/6] Navigating to project directory..."
cd /mnt/c/Users/melih/dev/disease_detection/Skin-Disease-Classifier || {
    echo "ERROR: Directory not found!"
    exit 1
}

# Python environment kontrolü
echo ""
echo "[5/6] Checking Python environment..."
python3 --version
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python3 -c "import numpy, matplotlib, sklearn, seaborn, pandas; print('All packages OK')" || {
    echo "WARNING: Some packages missing, installing..."
    pip install numpy matplotlib scikit-learn seaborn pandas openpyxl
}

# Models klasörü oluştur
mkdir -p models
mkdir -p training_logs

# Eğitimi başlat
echo ""
echo "[6/6] Starting bone disease training (IMPROVED VERSION)..."
echo "======================================"
echo "Training Configuration:"
echo "  - Model: EfficientNetB2"
echo "  - Image Size: 512x512"
echo "  - Batch Size: 16"
echo "  - Loss: Focal Loss (for class imbalance)"
echo "  - Classes: Normal, Fracture, Benign_Tumor, Malignant_Tumor"
echo "  - Phases: Initial (150 epochs) + Fine-tuning (80 epochs)"
echo "  - Improvements: Aggressive class weights, Lower LR, Higher patience"
echo ""
echo "Training will begin shortly..."
echo "Press Ctrl+C to stop (not recommended during training)"
echo "======================================"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_logs/bone_4class_improved_training_${TIMESTAMP}.log"

python3 train_bone_4class_improved.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "======================================"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Log saved to: $LOG_FILE"
else
    echo "Training failed! Check log: $LOG_FILE"
fi
echo "======================================"

