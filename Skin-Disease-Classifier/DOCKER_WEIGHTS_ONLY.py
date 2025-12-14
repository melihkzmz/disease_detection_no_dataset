#!/usr/bin/env python3
# Docker container içinde çalıştırılacak script
# Modeli weights-only olarak kaydeder, sonra yapıyı yeniden oluşturup yükleriz

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
print("MODEL WEIGHTS-ONLY KAYDETME")
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

# Mixed precision kapat
print("\n[2/3] Mixed precision kapatılıyor...")
tf.keras.mixed_precision.set_global_policy('float32')

# Model yapısını JSON olarak kaydet
print("\n[3/3] Model yapısı ve weights kaydediliyor...")

# Yöntem 1: Model config + weights
model_config = model.get_config()
import json
config_path = 'models/bone_model_config.json'
with open(config_path, 'w') as f:
    json.dump(model_config, f, indent=2)
print(f"✅ Model config kaydedildi: {config_path}")

# Weights'leri kaydet
weights_path = 'models/bone_model_weights.h5'
model.save_weights(weights_path)
print(f"✅ Model weights kaydedildi: {weights_path}")
print(f"   Weights boyutu: {os.path.getsize(weights_path) / (1024*1024):.2f} MB")

# Alternatif: Model'i JSON + weights olarak
model_json = model.to_json()
json_path = 'models/bone_model_architecture.json'
with open(json_path, 'w') as f:
    f.write(model_json)
print(f"✅ Model architecture (JSON) kaydedildi: {json_path}")

print("\n" + "="*70)
print("TAMAMLANDI!")
print("="*70)
print("\nWindows'ta modeli yüklemek için:")
print("1. Model config'i yükle: model = keras.models.model_from_config(config, custom_objects)")
print("2. Weights'i yükle: model.load_weights(weights_path)")

