#!/bin/bash
# Ubuntu GPU Training Script for Bone Disease Detection
# DenseNet-121 + Soft Macro F1 + Grayscale

echo "======================================"
echo "🦴 BONE DISEASE DETECTION - GPU TRAINING"
echo "======================================"
echo ""

# 1. GPU Kontrolü
echo "[1/5] Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA driver found"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "❌ ERROR: nvidia-smi not found!"
    echo "Please install NVIDIA driver first:"
    echo "  sudo apt update"
    echo "  sudo apt install nvidia-driver-xxx"
    exit 1
fi

# 2. TensorFlow GPU Kontrolü
echo ""
echo "[2/5] Checking TensorFlow GPU support..."
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU Count: {len(gpus)}'); [print(f'  - {gpu}') for gpu in gpus]" 2>/dev/null || {
    echo "❌ ERROR: TensorFlow GPU not available!"
    echo "Install with: pip install tensorflow[and-cuda]"
    exit 1
}

# 3. Dataset Kontrolü
echo ""
echo "[3/5] Checking dataset..."
DATASET_PATH="datasets/bone/Bone_4Class_Final"
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ ERROR: Dataset not found at: $DATASET_PATH"
    exit 1
fi

for split in train val test; do
    if [ ! -d "$DATASET_PATH/$split" ]; then
        echo "❌ ERROR: Missing $split directory"
        exit 1
    fi
done

echo "✅ Dataset found"

# 4. Python Paketleri Kontrolü
echo ""
echo "[4/5] Checking Python packages..."
python3 -c "import tensorflow, numpy, matplotlib, sklearn, seaborn, pandas; print('✅ All packages OK')" 2>/dev/null || {
    echo "⚠️  WARNING: Some packages missing"
    echo "Installing required packages..."
    pip install -q tensorflow numpy matplotlib scikit-learn seaborn pandas
}

# 5. Models klasörü oluştur
mkdir -p models
mkdir -p training_logs

# 6. Eğitimi başlat
echo ""
echo "[5/5] Starting training..."
echo "======================================"
echo "Training Configuration:"
echo "  - Model: DenseNet-121"
echo "  - Loss: Soft Macro F1 Loss"
echo "  - Image Size: 384×384"
echo "  - Color Mode: Grayscale"
echo "  - Batch Size: 16"
echo "  - Classes: Normal, Fracture, Benign_Tumor, Malignant_Tumor"
echo "======================================"
echo ""
echo "💡 Tip: Open another terminal and run 'watch -n 1 nvidia-smi' to monitor GPU usage"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_logs/bone_4class_densenet121_macro_f1_${TIMESTAMP}.log"

# Script'i çalıştır ve log'a kaydet
python3 train_bone_4class_macro_f1.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "======================================"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "📝 Log saved to: $LOG_FILE"
else
    echo "❌ Training failed! Check log: $LOG_FILE"
fi
echo "======================================"

