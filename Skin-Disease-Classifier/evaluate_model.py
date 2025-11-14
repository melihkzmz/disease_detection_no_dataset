#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick evaluation of trained eye disease model
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

print("\n" + "="*60)
print("EYE DISEASE MODEL - EVALUATION")
print("="*60)

# Load model
MODEL_PATH = 'models/eye_disease_model.keras'
TEST_DIR = 'datasets/Eye_Mendeley/test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    'Diabetic_Retinopathy',
    'Disc_Edema',
    'Glaucoma',
    'Macular_Scar',
    'Myopia',
    'Normal',
    'Pterygium',
    'Retinal_Detachment',
    'Retinitis_Pigmentosa'
]

print(f"\n[LOADING] Model: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("[SUCCESS] Model loaded!")

print(f"\n[LOADING] Test dataset: {TEST_DIR}")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    shuffle=False,
    batch_size=BATCH_SIZE
)

print("\n[EVALUATING] Running evaluation on test set...")
results = model.evaluate(test_ds, verbose=1)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"\nTest Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]*100:.2f}%")
print(f"Top-3 Accuracy: {results[2]*100:.2f}%")

# Per-class accuracy
print("\n[COMPUTING] Per-class predictions...")
y_pred = model.predict(test_ds, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_true_classes = np.argmax(y_true, axis=1)

from sklearn.metrics import classification_report, confusion_matrix

print("\n" + "="*60)
print("PER-CLASS PERFORMANCE")
print("="*60)
print(classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES, digits=3))

print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)
cm = confusion_matrix(y_true_classes, y_pred_classes)
print(cm)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Model: EfficientNetB3")
print(f"Dataset: Mendeley Eye Disease")
print(f"Classes: {len(CLASS_NAMES)}")
print(f"Test Samples: {len(y_true_classes)}")
print(f"\nTest Accuracy: {results[1]*100:.2f}%")
print(f"Top-3 Accuracy: {results[2]*100:.2f}%")
print("="*60 + "\n")

