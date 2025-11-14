#!/bin/bash
# Ubuntu GPU Training Script for Bone Disease Detection
# This script sets up the environment and starts training on GPU

set -e  # Exit on error

echo "================================================================================"
echo "BONE DISEASE DETECTION - UBUNTU GPU TRAINING"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_info "Working directory: $SCRIPT_DIR"
echo ""

# Check if CUDA is available
print_info "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    print_info "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    print_error "nvidia-smi not found. Is NVIDIA driver installed?"
    exit 1
fi

# Check if CUDA toolkit is installed
print_info "Checking CUDA toolkit..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | awk -F',' '{print $1}')
    print_info "CUDA version: $CUDA_VERSION"
else
    print_warn "nvcc not found in PATH. CUDA toolkit may not be in PATH."
    print_warn "TensorFlow will use the CUDA version it was compiled with."
fi
echo ""

# Set CUDA environment variables (adjust paths if needed)
print_info "Setting up CUDA environment variables..."

# Common CUDA paths (adjust based on your installation)
CUDA_PATHS=(
    "/usr/local/cuda"
    "/usr/local/cuda-11.8"
    "/usr/local/cuda-12.0"
    "/usr/local/cuda-12.1"
    "/usr/local/cuda-12.2"
    "/opt/cuda"
)

CUDA_FOUND=false
for CUDA_PATH in "${CUDA_PATHS[@]}"; do
    if [ -d "$CUDA_PATH" ]; then
        export CUDA_HOME="$CUDA_PATH"
        export PATH="$CUDA_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
        print_info "Using CUDA from: $CUDA_PATH"
        CUDA_FOUND=true
        break
    fi
done

if [ "$CUDA_FOUND" = false ]; then
    print_warn "CUDA installation directory not found in common paths."
    print_warn "If CUDA is installed elsewhere, set CUDA_HOME manually."
fi
echo ""

# Check Python version
print_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_info "$PYTHON_VERSION"
    
    # Check if required packages are installed
    print_info "Checking Python packages..."
    python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" 2>/dev/null || {
        print_error "TensorFlow not found. Installing requirements..."
        pip3 install -r requirements.txt
    }
    
    python3 -c "import numpy; import matplotlib; import sklearn; import seaborn" 2>/dev/null || {
        print_warn "Some packages missing. Installing..."
        pip3 install numpy matplotlib scikit-learn seaborn pandas openpyxl
    }
else
    print_error "Python3 not found. Please install Python 3.8+"
    exit 1
fi
echo ""

# Verify TensorFlow GPU support
print_info "Verifying TensorFlow GPU support..."
python3 - << EOF
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA available: {tf.test.is_built_with_cuda()}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detected: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu}")
if len(gpus) == 0:
    print("WARNING: No GPU detected by TensorFlow!")
    print("This could mean:")
    print("  1. CUDA/cuDNN not properly installed")
    print("  2. TensorFlow not compiled with GPU support")
    print("  3. GPU not accessible")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    print_error "TensorFlow GPU setup failed. Please check your CUDA/cuDNN installation."
    exit 1
fi
echo ""

# Check if dataset exists
print_info "Checking dataset..."
DATASET_DIR="datasets/bone/Bone_4Class_Final"
if [ ! -d "$DATASET_DIR" ]; then
    print_error "Dataset directory not found: $DATASET_DIR"
    print_error "Please run the dataset organization script first."
    exit 1
fi

# Check train/val/test directories
for split in train val test; do
    if [ ! -d "$DATASET_DIR/$split" ]; then
        print_error "Missing $split directory in dataset"
        exit 1
    fi
done
print_info "Dataset structure verified."
echo ""

# Create models directory if it doesn't exist
mkdir -p models

# Create log directory
LOG_DIR="training_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/bone_4class_training_$TIMESTAMP.log"

print_info "Training log will be saved to: $LOG_FILE"
echo ""

# Start training
print_info "Starting training..."
print_info "Training script: train_bone_4class_optimized.py"
echo ""
echo "================================================================================"
echo "TRAINING STARTED - Output will be logged to: $LOG_FILE"
echo "================================================================================"
echo ""

# Run training with logging
python3 train_bone_4class_optimized.py 2>&1 | tee "$LOG_FILE"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================================"
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    print_info "Training completed successfully!"
    print_info "Check the log file for details: $LOG_FILE"
else
    print_error "Training failed with exit code: $TRAINING_EXIT_CODE"
    print_error "Check the log file for errors: $LOG_FILE"
fi
echo "================================================================================"

exit $TRAINING_EXIT_CODE

