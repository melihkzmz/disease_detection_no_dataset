import tensorflow as tf

print("\n" + "="*60)
print("GPU DETECTION TEST")
print("="*60)

print(f"\nTensorFlow Version: {tf.__version__}")
print(f"CUDA Built: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU Count: {len(gpus)}")

if len(gpus) > 0:
    print("\n[SUCCESS] GPU BULUNDU!")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"    Details: {details}")
        except:
            pass
else:
    print("\n[FAIL] GPU BULUNAMADI!")
    print("\nOlasi Sebepler:")
    print("  1. TensorFlow GPU versiyonu kurulu degil")
    print("  2. CUDA/cuDNN uyumsuz")
    print("  3. NVIDIA driver guncel degil")

print("="*60 + "\n")

