#!/usr/bin/env python3
# Docker container içinde çalıştırılacak script
# En basit yöntem: Modeli yükle, bir dummy prediction yap, sonra kaydet

from tensorflow import keras
import tensorflow as tf
import numpy as np
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
print("BASIT MODEL YENIDEN KAYDETME")
print("="*70)

custom_objects = {
    'GrayscaleToRGB': GrayscaleToRGB,
    'StreamingMacroF1': StreamingMacroF1
}

# Modeli yükle
print("\n[1/3] Model yükleniyor...")
model = keras.models.load_model(
    'models/bone_disease_model_4class_densenet121_macro_f1.keras',
    custom_objects=custom_objects,
    compile=False
)
print("✅ Model yüklendi!")

# Input shape'i al
input_shape = model.input_shape[1:3] if model.input_shape else (384, 384)
print(f"   Model input shape: {input_shape}")

# Mixed precision kapat
print("\n[2/3] Mixed precision kapatılıyor...")
tf.keras.mixed_precision.set_global_policy('float32')

# Modeli bir dummy input ile çalıştır (quantization_mode hatası için)
print("   Model bir dummy input ile test ediliyor...")
try:
    dummy_input = np.random.random((1, input_shape[0], input_shape[1], 3)).astype(np.float32)
    _ = model.predict(dummy_input, verbose=0)
    print("   ✅ Model çalışıyor!")
except Exception as e:
    print(f"   ⚠️ Dummy prediction hatası (normal olabilir): {str(e)[:100]}")

# Modeli yeniden kaydet - .keras formatında
print("\n[3/3] Model kaydediliyor...")
new_path = 'models/bone_disease_model_4class_densenet121_macro_f1_fixed.keras'

# Önce eski dosyayı sil
if os.path.exists(new_path):
    os.remove(new_path)

# Modeli kaydet
try:
    model.save(new_path)
    print(f"✅ Model kaydedildi: {new_path}")
    print(f"   Dosya boyutu: {os.path.getsize(new_path) / (1024*1024):.2f} MB")
except Exception as e:
    print(f"❌ Kayıt hatası: {e}")
    print("\nAlternatif: Weights-only kaydetmeyi deneyin:")
    weights_path = 'models/bone_model_weights.h5'
    model.save_weights(weights_path)
    print(f"✅ Weights kaydedildi: {weights_path}")

print("\n" + "="*70)
print("TAMAMLANDI!")
print("="*70)

