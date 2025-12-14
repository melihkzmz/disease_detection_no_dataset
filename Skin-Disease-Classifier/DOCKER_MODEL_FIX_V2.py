#!/usr/bin/env python3
# Docker container içinde çalıştırılacak script
# Modeli Windows'ta çalışacak şekilde yeniden kaydeder

from tensorflow import keras
import tensorflow as tf
import os

# Custom classes
class GrayscaleToRGB(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GrayscaleToRGB, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.repeat(inputs, 3, axis=-1)
    def get_config(self):
        config = super(GrayscaleToRGB, self).get_config()
        return config

class StreamingMacroF1(keras.metrics.Metric):
    def __init__(self, num_classes=4, name='macro_f1_metric', **kwargs):
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

print("="*70)
print("MODEL YENIDEN KAYDETME (V2)")
print("="*70)

custom_objects = {
    'GrayscaleToRGB': GrayscaleToRGB,
    'StreamingMacroF1': StreamingMacroF1
}

# Modeli yükle
print("\n[1/3] Model yükleniyor...")
try:
    model = keras.models.load_model(
        'models/bone_disease_model_4class_densenet121_macro_f1.keras',
        custom_objects=custom_objects,
        compile=False
    )
    print("✅ Model yüklendi!")
except Exception as e:
    print(f"❌ Model yüklenemedi: {e}")
    exit(1)

# Mixed precision kapat
print("\n[2/3] Mixed precision kapatılıyor...")
tf.keras.mixed_precision.set_global_policy('float32')

# Test prediction - Model input shape'ini kontrol et
print("\n[3/3] Model test ediliyor...")
import numpy as np
# Model input shape'ini al (genellikle 384x384 veya 224x224)
input_shape = model.input_shape[1:3]  # (height, width) al
print(f"   Model input shape: {input_shape}")
test_input = np.random.random((1, input_shape[0], input_shape[1], 3)).astype(np.float32)
test_output = model.predict(test_input, verbose=0)
print(f"✅ Test prediction başarılı! Output shape: {test_output.shape}")

# Modeli yeniden kaydet - SavedModel formatında dene
print("\n[4/4] Model kaydediliyor...")

# Yöntem 1: .keras formatında (safe_mode=False ile)
try:
    new_path = 'models/bone_disease_model_4class_densenet121_macro_f1_fixed.keras'
    model.save(new_path, save_format='keras')
    print(f"✅ Model kaydedildi (keras format): {new_path}")
    print(f"   Dosya boyutu: {os.path.getsize(new_path) / (1024*1024):.2f} MB")
except Exception as e:
    print(f"⚠️ Keras formatında kayıt başarısız: {e}")

# Yöntem 2: SavedModel formatında (alternatif)
try:
    savedmodel_path = 'models/bone_disease_model_4class_densenet121_macro_f1_fixed_savedmodel'
    model.save(savedmodel_path, save_format='tf')
    print(f"✅ Model kaydedildi (SavedModel format): {savedmodel_path}")
except Exception as e:
    print(f"⚠️ SavedModel formatında kayıt başarısız: {e}")

print("\n" + "="*70)
print("TAMAMLANDI!")
print("="*70)

