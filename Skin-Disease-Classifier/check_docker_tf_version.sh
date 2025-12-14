#!/bin/bash
# Docker container'daki TensorFlow versiyonunu kontrol et

echo "Docker container'daki TensorFlow versiyonunu kontrol ediliyor..."

docker run --rm nvcr.io/nvidia/tensorflow:25.01-tf2-py3 python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Keras version:', tf.keras.__version__ if hasattr(tf.keras, '__version__') else 'N/A')"


