#!/bin/bash
# Ubuntu GPU Training Script - Bone Disease Detection
# DenseNet121 + Macro F1 + Grayscale

set -e  # Exit on error

echo "======================================"
echo "🦴 BONE DISEASE DETECTION - GPU TRAINING"
echo "DenseNet121 + Soft Macro F1 + Grayscale"
echo "======================================"
echo ""

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================================
# 1. GPU KONTROLÜ
# ============================================================================
echo -e "${YELLOW}[1/7] Checking GPU...${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA Driver:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo -e "${RED}ERROR: nvidia-smi not found!${NC}"
    echo "Please install NVIDIA drivers first."
    exit 1
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c1-4)
    echo "CUDA Version: $CUDA_VERSION"
else
    echo -e "${YELLOW}WARNING: nvcc not found (CUDA toolkit may not be installed)${NC}"
    echo "TensorFlow will still work if GPU drivers are installed."
fi

echo ""

# ============================================================================
# 2. TENSORFLOW GPU KONTROLÜ
# ============================================================================
echo -e "${YELLOW}[2/7] Checking TensorFlow GPU support...${NC}"

python3 -c "
import tensorflow as tf
print(f'TensorFlow Version: {tf.__version__}')
print(f'Built with CUDA: {tf.test.is_built_with_cuda()}')

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'\n✅ {len(gpus)} GPU(s) detected:')
    for i, gpu in enumerate(gpus):
        print(f'  GPU {i}: {gpu.name}')
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f'    Details: {details}')
        except:
            pass
else:
    print('\n❌ No GPU detected!')
    print('Available devices:', tf.config.list_physical_devices())
    exit(1)
" || {
    echo -e "${RED}ERROR: TensorFlow GPU check failed!${NC}"
    echo ""
    echo "Solutions:"
    echo "  1. Install TensorFlow GPU: pip install tensorflow[and-cuda]"
    echo "  2. Or: pip install tensorflow-gpu (if available for your TF version)"
    echo "  3. Check CUDA compatibility with TensorFlow version"
    exit 1
}

echo ""

# ============================================================================
# 3. PYTHON ENVIRONMENT KONTROLÜ
# ============================================================================
echo -e "${YELLOW}[3/7] Checking Python environment...${NC}"

PYTHON_CMD="python3"

# Check if conda environment exists
if [ -d "$HOME/miniconda3/envs/tf_gpu" ] || [ -d "$HOME/anaconda3/envs/tf_gpu" ]; then
    echo "Found conda environment: tf_gpu"
    read -p "Activate conda environment? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        fi
        conda activate tf_gpu
        echo "✅ Conda environment activated"
    fi
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "Python: $PYTHON_VERSION"

# Check required packages
echo "Checking required packages..."
$PYTHON_CMD -c "
import sys
packages = {
    'tensorflow': 'TensorFlow',
    'keras': 'Keras',
    'numpy': 'NumPy',
    'matplotlib': 'Matplotlib',
    'sklearn': 'Scikit-learn',
    'seaborn': 'Seaborn',
    'PIL': 'Pillow'
}

missing = []
for package, name in packages.items():
    try:
        __import__(package)
        print(f'  ✅ {name}')
    except ImportError:
        print(f'  ❌ {name} - MISSING')
        missing.append(name)

if missing:
    print(f'\n⚠️  Missing packages: {', '.join(missing)}')
    print('Install with: pip install tensorflow matplotlib scikit-learn seaborn pillow')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo ""
    read -p "Install missing packages? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install tensorflow matplotlib scikit-learn seaborn pillow
    else
        exit 1
    fi
fi

echo ""

# ============================================================================
# 4. DATASET KONTROLÜ
# ============================================================================
echo -e "${YELLOW}[4/7] Checking dataset...${NC}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATASET_PATH="$SCRIPT_DIR/datasets/bone/Bone_4Class_Final"

if [ ! -d "$DATASET_PATH" ]; then
    echo -e "${RED}ERROR: Dataset not found at: $DATASET_PATH${NC}"
    exit 1
fi

for split in train val test; do
    if [ ! -d "$DATASET_PATH/$split" ]; then
        echo -e "${RED}ERROR: Missing $split directory${NC}"
        exit 1
    fi
done

# Count images
echo "Dataset structure:"
for split in train val test; do
    COUNT=$(find "$DATASET_PATH/$split" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
    echo "  $split: $COUNT images"
    
    # Check classes
    CLASSES=$(find "$DATASET_PATH/$split" -mindepth 1 -maxdepth 1 -type d | wc -l)
    echo "    Classes: $CLASSES"
done

echo ""

# ============================================================================
# 5. WORKING DIRECTORY
# ============================================================================
echo -e "${YELLOW}[5/7] Setting up working directory...${NC}"

cd "$SCRIPT_DIR" || {
    echo -e "${RED}ERROR: Cannot navigate to script directory${NC}"
    exit 1
}

echo "Working directory: $(pwd)"

# Create directories
mkdir -p models
mkdir -p training_logs

echo "✅ Directories ready"
echo ""

# ============================================================================
# 6. GPU MEMORY CONFIGURATION (Optional)
# ============================================================================
echo -e "${YELLOW}[6/7] GPU Memory Configuration...${NC}"

# Set GPU memory growth (prevents OOM)
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "Environment variables set:"
echo "  TF_FORCE_GPU_ALLOW_GROWTH=true"
echo "  TF_GPU_ALLOCATOR=cuda_malloc_async"
echo ""

# ============================================================================
# 7. START TRAINING
# ============================================================================
echo -e "${YELLOW}[7/7] Starting training...${NC}"
echo "======================================"
echo -e "${GREEN}Training Configuration:${NC}"
echo "  Model: DenseNet121"
echo "  Input: 384×384 Grayscale"
echo "  Loss: Soft Macro F1 Loss"
echo "  Metric: Macro F1"
echo "  Batch Size: 16"
echo "  Initial Epochs: 150"
echo "  Fine-tune Epochs: 80"
echo "  Classes: Normal, Fracture, Benign_Tumor, Malignant_Tumor"
echo ""
echo "GPU will be used automatically if available."
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop training${NC}"
echo "======================================"
echo ""

# Create log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="training_logs/bone_4class_densenet121_macro_f1_${TIMESTAMP}.log"

echo "Log file: $LOG_FILE"
echo ""

# Run training
$PYTHON_CMD train_bone_4class_macro_f1.py 2>&1 | tee "$LOG_FILE"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "======================================"
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo "Log saved to: $LOG_FILE"
    echo ""
    echo "Models saved to: models/"
    echo "  - bone_4class_densenet121_macro_f1_initial.keras"
    echo "  - bone_4class_densenet121_macro_f1_finetuned.keras"
    echo "  - bone_disease_model_4class_densenet121_macro_f1.keras"
else
    echo -e "${RED}❌ Training failed!${NC}"
    echo "Check log file: $LOG_FILE"
    exit 1
fi
echo "======================================"

