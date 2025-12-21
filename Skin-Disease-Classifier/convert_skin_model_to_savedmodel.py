#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert skin disease model from .keras to SavedModel format
This fixes Keras 3.x compatibility issues
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print("="*70)
print("SKIN DISEASE MODEL CONVERSION: .keras -> SavedModel")
print("="*70)

# Model paths
KERAS_MODEL_PATH = 'models/skin_disease_model_5class_efficientnetb3_macro_f1.keras'
SAVEDMODEL_PATH = 'models/skin_disease_model_5class_efficientnetb3_macro_f1_savedmodel'

# Custom metric class (same as in training script)
class StreamingMacroF1(keras.metrics.Metric):
    """Streaming Macro F1 Metric - needed for model loading"""
    def __init__(self, num_classes=5, name='macro_f1_metric', **kwargs):
        super(StreamingMacroF1, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros', dtype=tf.float32)
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros', dtype=tf.float32)
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros', dtype=tf.float32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_classes = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
        y_pred_classes = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
        y_true_one_hot = tf.one_hot(y_true_classes, depth=self.num_classes, dtype=tf.float32)
        y_pred_one_hot = tf.one_hot(y_pred_classes, depth=self.num_classes, dtype=tf.float32)
        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fp = tf.reduce_sum((1.0 - y_true_one_hot) * y_pred_one_hot, axis=0)
        fn = tf.reduce_sum(y_true_one_hot * (1.0 - y_pred_one_hot), axis=0)
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1_scores = 2.0 * precision * recall / (precision + recall + 1e-8)
        return tf.reduce_mean(f1_scores)
    
    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

# Check if Keras model exists
if not os.path.exists(KERAS_MODEL_PATH):
    print(f"[HATA] Keras model bulunamadi: {KERAS_MODEL_PATH}")
    sys.exit(1)

print(f"\n[1/3] Keras model yukleniyor: {KERAS_MODEL_PATH}")

try:
    # Keras 3.x compatibility issue - model was saved with Keras 2.x
    # Try multiple methods to load
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        # Method 1: Try with TensorFlow 2.x compatibility (if available)
        try:
            # Force TensorFlow 2.x behavior
            import os
            os.environ['TF_KERAS'] = '1'  # Use tf.keras
            
            # Try loading with tf.keras (TensorFlow's Keras wrapper)
            # This might work if TensorFlow 2.x is available
            model = tf.keras.models.load_model(
                KERAS_MODEL_PATH,
                custom_objects={'StreamingMacroF1': StreamingMacroF1},
                compile=False
            )
            print("[OK] Model yuklendi (tf.keras ile)")
        except Exception as e1:
            print(f"[UYARI] tf.keras ile yukleme basarisiz: {str(e1)[:200]}")
            
            # Method 2: Try with Keras 2.x compatibility mode
            try:
                # Set Keras backend to use TensorFlow 2.x
                import keras
                # Try loading with safe_mode=False
                model = keras.models.load_model(
                    KERAS_MODEL_PATH,
                    custom_objects={'StreamingMacroF1': StreamingMacroF1},
                    compile=False,
                    safe_mode=False
                )
                print("[OK] Model yuklendi (keras ile safe_mode=False)")
            except Exception as e2:
                print(f"[UYARI] keras ile yukleme basarisiz: {str(e2)[:200]}")
                
                # Method 3: Last resort - try to load weights only and rebuild
                print("\n[COZUM] Keras 3.x uyumluluk sorunu tespit edildi.")
                print("[COZUM] Model .keras formatindan SavedModel'e donusturulemiyor.")
                print("[COZUM] LUTFEN ASAGIDAKI YONTEMLERDEN BIRINI DENEYIN:")
                print("\n  1. TensorFlow 2.x kullanarak model'i yeniden kaydedin:")
                print("     python -c \"import tensorflow as tf; tf.keras.models.load_model('models/skin_disease_model_5class_efficientnetb3_macro_f1.keras', compile=False).save('models/skin_disease_model_5class_efficientnetb3_macro_f1_savedmodel', save_format='tf')\"")
                print("\n  2. VEYA training script'i calistirarak model'i yeniden egitin (SavedModel otomatik kaydedilecek)")
                print("\n  3. VEYA TensorFlow 2.x kurulumu yapin ve bu script'i tekrar calistirin")
                raise Exception("Keras 3.x uyumluluk sorunu - TensorFlow 2.x gerekli")
    
    print(f"[OK] Model yuklendi!")
    print(f"[BILGI] Input shape: {model.input_shape}")
    print(f"[BILGI] Output shape: {model.output_shape}")
    
except Exception as e:
    print(f"\n[HATA] Model yuklenemedi: {str(e)[:500]}")
    print("\n[COZUM] Model dosyasini kontrol edin veya yeniden egitin")
    sys.exit(1)

# Convert to SavedModel
print(f"\n[2/3] SavedModel formatina donusturuluyor: {SAVEDMODEL_PATH}")

try:
    # Remove old SavedModel if exists
    if os.path.exists(SAVEDMODEL_PATH):
        import shutil
        shutil.rmtree(SAVEDMODEL_PATH)
        print(f"[INFO] Eski SavedModel silindi")
    
    # Save as SavedModel
    model.save(SAVEDMODEL_PATH, save_format='tf')
    print(f"[OK] SavedModel kaydedildi!")
    
    # Verify SavedModel
    if os.path.exists(SAVEDMODEL_PATH):
        saved_model_size = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, dirnames, filenames in os.walk(SAVEDMODEL_PATH)
            for filename in filenames
        ) / (1024 * 1024)  # MB
        print(f"[OK] SavedModel dogrulandi: {saved_model_size:.2f} MB")
    
except Exception as e:
    print(f"\n[HATA] SavedModel donusumu basarisiz: {str(e)[:500]}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test SavedModel loading
print(f"\n[3/3] SavedModel test ediliyor...")

try:
    saved_model = tf.saved_model.load(SAVEDMODEL_PATH)
    
    # Test prediction
    test_input = np.random.random((1, 300, 300, 3)).astype(np.float32)
    
    if hasattr(saved_model, 'signatures') and 'serving_default' in saved_model.signatures:
        # Use signature
        result = saved_model.signatures['serving_default'](tf.constant(test_input))
        output_key = list(result.keys())[0]
        test_output = result[output_key].numpy()
    elif callable(saved_model):
        # Direct callable
        test_output = saved_model(tf.constant(test_input)).numpy()
    else:
        raise ValueError("SavedModel format desteklenmiyor")
    
    print(f"[OK] SavedModel test basarili!")
    print(f"[OK] Output shape: {test_output.shape}")
    print(f"[OK] Output sum: {test_output.sum():.4f} (should be ~1.0)")
    
except Exception as e:
    print(f"[UYARI] SavedModel test basarisiz: {str(e)[:200]}")
    print("[UYARI] Ancak SavedModel kaydedildi, API'de test edilebilir")

print("\n" + "="*70)
print("DONUSUM TAMAMLANDI!")
print("="*70)
print(f"[OK] SavedModel: {SAVEDMODEL_PATH}")
print(f"[OK] Artik skin_disease_api.py SavedModel formatini kullanabilir")
print("="*70)

