#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Docker'da kaydedilmiş modeli Windows TensorFlow versiyonu ile düzelt
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Custom objects
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
print("DOCKER MODEL DOSYASI DÜZELTME")
print("="*70)

# Windows TensorFlow versiyonu
print(f"\nWindows TensorFlow versiyonu: {tf.__version__}")

# Model dosyaları
model_files = [
    'models/bone_4class_densenet121_macro_f1_finetuned.keras',
    'models/bone_4class_densenet121_macro_f1_initial.keras',
    'models/bone_disease_model_4class_densenet121_macro_f1.keras'
]

custom_objects = {
    'GrayscaleToRGB': GrayscaleToRGB,
    'StreamingMacroF1': StreamingMacroF1
}

for model_path in model_files:
    if not os.path.exists(model_path):
        print(f"\n[DOSYA YOK] {model_path}")
        continue
    
    print(f"\n[YUKLENIYOR] {model_path}")
    try:
        # Docker'da kaydedilmiş modeli yükle
        model = keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        print(f"[BASARILI] Model yuklendi!")
        
        # Test prediction yap
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        test_pred = model.predict(dummy_input, verbose=0)
        print(f"[TEST] Model tahmin yapabiliyor: {test_pred.shape}")
        
        # Windows TensorFlow versiyonu ile yeniden kaydet
        fixed_path = model_path.replace('.keras', '_fixed.keras')
        print(f"[KAYDEDIYOR] {fixed_path}")
        model.save(fixed_path)
        print(f"[BASARILI] Model Windows formatina donusturuldu!")
        
    except Exception as e:
        print(f"[HATA] {str(e)}")

print("\n" + "="*70)
print("ISLEM TAMAMLANDI")
print("="*70)


