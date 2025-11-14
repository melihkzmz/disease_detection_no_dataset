#!/bin/bash
# GPU Kontrol Scripti

echo "======================================"
echo "GPU DIAGNOSTIC CHECK"
echo "======================================"
echo ""

echo "[1] Conda Environment:"
conda info --envs | grep "*"
echo ""

echo "[2] Python Version:"
python3 --version
echo ""

echo "[3] TensorFlow Version:"
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
echo ""

echo "[4] GPU Devices:"
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPU Count: {len(gpus)}'); [print(f'  - {gpu}') for gpu in gpus]"
echo ""

echo "[5] CUDA Availability:"
python3 -c "import tensorflow as tf; print(f'CUDA Built: {tf.test.is_built_with_cuda()}')"
python3 -c "import tensorflow as tf; print(f'GPU Available: {tf.test.is_gpu_available()}')"
echo ""

echo "[6] LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH
echo ""

echo "[7] CUDA Version (nvcc):"
nvcc --version 2>/dev/null || echo "nvcc not found in PATH"
echo ""

echo "[8] NVIDIA Driver:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
echo ""


