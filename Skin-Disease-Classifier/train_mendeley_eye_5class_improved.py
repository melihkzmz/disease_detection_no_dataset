#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mendeley Eye Disease Detection - IMPROVED 5 CLASS TRAINING
Fixed: Early Stopping, Loss Function, Learning Rate, Class Weights
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print("\n" + "="*70)
print("MENDELEY EYE DISEASE DETECTION - IMPROVED 5 CLASS TRAINING")
print("Fixes: Better Loss, Higher Patience, Label Smoothing, Better LR")
print("="*70)

# Check GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"\n[GPU] {len(physical_devices)} GPU(s) available")
    for device in physical_devices:
        print(f"  - {device}")
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("[GPU] Memory growth enabled")
    except:
        pass
else:
    print("\n[CPU] No GPU found, training on CPU")

# Hyperparameters - IMPROVED
TRAIN_DIR = 'datasets/Eye_Mendeley/train'
VAL_DIR = 'datasets/Eye_Mendeley/val'
TEST_DIR = 'datasets/Eye_Mendeley/test'

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
INITIAL_EPOCHS = 100  # Increased
FINE_TUNE_EPOCHS = 50  # Increased
LEARNING_RATE = 0.0005  # Lower initial LR
FINE_TUNE_LR = 0.00005  # Lower fine-tune LR
LABEL_SMOOTHING = 0.1  # Add label smoothing

# 5 Selected Classes
CLASS_NAMES = [
    'Diabetic_Retinopathy',
    'Glaucoma',
    'Macular_Scar',
    'Myopia',
    'Normal'
]

NUM_CLASSES = len(CLASS_NAMES)

print(f"\n[CONFIG]")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Initial Epochs: {INITIAL_EPOCHS}")
print(f"  Fine-tune Epochs: {FINE_TUNE_EPOCHS}")
print(f"  Initial LR: {LEARNING_RATE}")
print(f"  Fine-tune LR: {FINE_TUNE_LR}")
print(f"  Label Smoothing: {LABEL_SMOOTHING}")
print(f"  Number of Classes: {NUM_CLASSES}")
print(f"  Classes: {', '.join(CLASS_NAMES)}")

# Label smoothing loss (better than focal loss for this case)
def categorical_crossentropy_smooth(smoothing=0.1):
    """
    Categorical crossentropy with label smoothing
    Better than focal loss when class imbalance is moderate
    """
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        # Apply label smoothing
        num_classes = tf.cast(tf.shape(y_true)[1], tf.float32)
        y_true_smooth = y_true * (1.0 - smoothing) + smoothing / num_classes
        loss = -tf.reduce_sum(y_true_smooth * tf.math.log(y_pred), axis=-1)
        return tf.reduce_mean(loss)
    return loss

# Moderate data augmentation (not too aggressive)
print("\n[DATA] Creating data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Reduced
    width_shift_range=0.15,  # Reduced
    height_shift_range=0.15,  # Reduced
    shear_range=0.1,  # Reduced
    zoom_range=0.2,  # Reduced
    horizontal_flip=True,
    vertical_flip=False,  # Removed
    fill_mode='reflect',
    brightness_range=[0.8, 1.2],  # Less aggressive
    channel_shift_range=10  # Reduced
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

print(f"\n[DATA] Generators created:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Test samples: {test_generator.samples}")

# Calculate STRONG class weights (inverse frequency)
print("\n[DATA] Calculating class weights...")
class_counts = np.bincount(train_generator.classes)
total_samples = len(train_generator.classes)

# Inverse frequency weighting (more aggressive)
class_weights = total_samples / (NUM_CLASSES * class_counts)
class_weights_dict = dict(zip(range(NUM_CLASSES), class_weights))

print("\n[INFO] Class weights (inverse frequency):")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name}: {class_weights_dict[i]:.3f} (samples: {class_counts[i]})")

# Build model
print("\n[MODEL] Building EfficientNetB3 model...")

base_model = EfficientNetB3(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# Simpler architecture (less overfitting risk)
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Reduced
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),  # Less regularization
    layers.BatchNormalization(),
    layers.Dropout(0.4),  # Reduced
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile with label smoothing loss
smooth_loss = categorical_crossentropy_smooth(LABEL_SMOOTHING)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999),
    loss=smooth_loss,
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]
)

print("\n[MODEL] Model architecture:")
model.summary()
print(f"\n[MODEL] Total parameters: {model.count_params():,}")

# Callbacks with BETTER settings
os.makedirs('models', exist_ok=True)

checkpoint_initial = ModelCheckpoint(
    'models/mendeley_eye_5class_improved_initial.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1,
    save_weights_only=False
)

# CRITICAL FIX: Much higher patience
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=50,  # CRITICAL: Was 20, now 50 - let model learn!
    restore_best_weights=True,
    verbose=1,
    mode='max',
    min_delta=0.001  # Minimum improvement required
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,  # More aggressive reduction
    patience=10,  # Wait longer before reducing
    min_lr=1e-7,
    verbose=1,
    cooldown=5  # Wait before resuming after LR reduction
)

# Phase 1: Initial Training
print("\n" + "="*70)
print("PHASE 1: INITIAL TRAINING (Base Model Frozen)")
print("Using: Label Smoothing + Strong Class Weights")
print("="*70)

history_initial = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights_dict,  # Use class weights
    callbacks=[checkpoint_initial, early_stopping, reduce_lr],
    verbose=1
)

# Load best model from phase 1
print("\n[INFO] Loading best model from Phase 1...")
model = keras.models.load_model(
    'models/mendeley_eye_5class_improved_initial.keras',
    custom_objects={'loss': smooth_loss}
)

# Check intermediate results
print("\n[EVAL] Phase 1 Results:")
loss, accuracy, top_3_acc = model.evaluate(test_generator, verbose=0)
print(f"  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Top-3 Accuracy: {top_3_acc*100:.2f}%")

# Check if model is learning all classes
test_generator.reset()
y_pred_phase1 = model.predict(test_generator, verbose=0)
y_pred_classes_phase1 = np.argmax(y_pred_phase1, axis=1)
y_true = test_generator.classes

unique_preds = np.unique(y_pred_classes_phase1)
print(f"\n[INFO] Phase 1: Model predicts {len(unique_preds)} unique classes out of {NUM_CLASSES}")
print(f"  Predicted classes: {unique_preds}")
if len(unique_preds) == 1:
    print("  WARNING: Model only predicts one class! Training may have issues.")
else:
    print("  GOOD: Model is learning multiple classes!")

# Phase 2: Fine-tuning
print("\n" + "="*70)
print("PHASE 2: FINE-TUNING (Unfreezing Top Layers)")
print("="*70)

base_model.trainable = True

# Unfreeze last 120 layers (more layers)
freeze_until = len(base_model.layers) - 120
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

print(f"\n[MODEL] Unfrozen layers: {len([l for l in base_model.layers if l.trainable])}")
print(f"[MODEL] Frozen layers: {len([l for l in base_model.layers if not l.trainable])}")

# Recompile with lower learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR, beta_1=0.9, beta_2=0.999),
    loss=smooth_loss,
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]
)

checkpoint_finetune = ModelCheckpoint(
    'models/mendeley_eye_5class_improved_finetuned.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1,
    save_weights_only=False
)

# Reset early stopping for fine-tuning phase
early_stopping_finetune = EarlyStopping(
    monitor='val_accuracy',
    patience=30,  # Still high but less than initial phase
    restore_best_weights=True,
    verbose=1,
    mode='max',
    min_delta=0.001
)

# Fine-tune
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights_dict,
    callbacks=[checkpoint_finetune, early_stopping_finetune, reduce_lr],
    verbose=1
)

# Load best fine-tuned model
print("\n[INFO] Loading best fine-tuned model...")
model = keras.models.load_model(
    'models/mendeley_eye_5class_improved_finetuned.keras',
    custom_objects={'loss': smooth_loss}
)

# Final evaluation
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

loss, accuracy, top_3_acc = model.evaluate(test_generator, verbose=1)

print(f"\n[RESULTS] Final Test Metrics:")
print(f"  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Top-3 Accuracy: {top_3_acc*100:.2f}%")

# Per-class accuracy
print("\n[RESULTS] Per-class Accuracy:")
test_generator.reset()
y_pred = model.predict(test_generator, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

print("\n" + classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES, zero_division=0))

cm = confusion_matrix(y_true, y_pred_classes)
print("\n[RESULTS] Confusion Matrix:")
print(cm)

# Check prediction distribution
unique_preds_final = np.unique(y_pred_classes)
print(f"\n[INFO] Final: Model predicts {len(unique_preds_final)} unique classes out of {NUM_CLASSES}")
print(f"  Predicted classes: {unique_preds_final}")
if len(unique_preds_final) < NUM_CLASSES:
    print(f"  WARNING: Model does not predict all classes!")
    print(f"  Missing classes: {set(range(NUM_CLASSES)) - set(unique_preds_final)}")
else:
    print("  EXCELLENT: Model predicts all classes!")

# Save final model
final_model_path = 'models/eye_disease_model_5class_improved.keras'
model.save(final_model_path)
print(f"\n[SUCCESS] Final model saved to: {final_model_path}")

# Plot training history
print("\n[INFO] Generating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

epochs_initial = range(1, len(history_initial.history['accuracy']) + 1)
epochs_finetune = range(len(history_initial.history['accuracy']) + 1, 
                        len(history_initial.history['accuracy']) + len(history_finetune.history['accuracy']) + 1)

# Accuracy
axes[0, 0].plot(epochs_initial, history_initial.history['accuracy'], 'b-', label='Train (Phase 1)', linewidth=2)
axes[0, 0].plot(epochs_initial, history_initial.history['val_accuracy'], 'b--', label='Val (Phase 1)', linewidth=2)
axes[0, 0].plot(epochs_finetune, history_finetune.history['accuracy'], 'r-', label='Train (Phase 2)', linewidth=2)
axes[0, 0].plot(epochs_finetune, history_finetune.history['val_accuracy'], 'r--', label='Val (Phase 2)', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(epochs_initial, history_initial.history['loss'], 'b-', label='Train (Phase 1)', linewidth=2)
axes[0, 1].plot(epochs_initial, history_initial.history['val_loss'], 'b--', label='Val (Phase 1)', linewidth=2)
axes[0, 1].plot(epochs_finetune, history_finetune.history['loss'], 'r-', label='Train (Phase 2)', linewidth=2)
axes[0, 1].plot(epochs_finetune, history_finetune.history['val_loss'], 'r--', label='Val (Phase 2)', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Loss', fontsize=12)
axes[0, 1].set_title('Model Loss (Label Smoothing)', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Top-3 Accuracy
axes[1, 0].plot(epochs_initial, history_initial.history['top_3_accuracy'], 'b-', label='Train (Phase 1)', linewidth=2)
axes[1, 0].plot(epochs_initial, history_initial.history['val_top_3_accuracy'], 'b--', label='Val (Phase 1)', linewidth=2)
axes[1, 0].plot(epochs_finetune, history_finetune.history['top_3_accuracy'], 'r-', label='Train (Phase 2)', linewidth=2)
axes[1, 0].plot(epochs_finetune, history_finetune.history['val_top_3_accuracy'], 'r--', label='Val (Phase 2)', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Top-3 Accuracy', fontsize=12)
axes[1, 0].set_title('Model Top-3 Accuracy', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1, 1],
            cbar_kws={'label': 'Count'})
axes[1, 1].set_xlabel('Predicted', fontsize=12)
axes[1, 1].set_ylabel('True', fontsize=12)
axes[1, 1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right')
plt.setp(axes[1, 1].get_yticklabels(), rotation=0)

plt.tight_layout()
plot_path = 'models/training_history_mendeley_eye_5class_improved.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"[SUCCESS] Training plots saved to: {plot_path}")

# Summary
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\n[FILES]")
print(f"  Final Model: {final_model_path}")
print(f"  Training Plot: {plot_path}")
print(f"\n[METRICS]")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Top-3 Accuracy: {top_3_acc*100:.2f}%")
print(f"  Total Classes: {NUM_CLASSES}")
print(f"  Total Parameters: {model.count_params():,}")
print(f"\n[CLASSES]")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {i+1}. {class_name}")
print(f"\n[IMPROVEMENTS]")
print(f"  ✓ Label Smoothing Loss (instead of Focal Loss)")
print(f"  ✓ Higher Early Stopping Patience (50 epochs)")
print(f"  ✓ Lower Learning Rates")
print(f"  ✓ Stronger Class Weights")
print(f"  ✓ More Epochs (100+50)")
print(f"  ✓ Better Architecture")
print(f"\n[IMPORTANT]")
if len(unique_preds_final) == NUM_CLASSES:
    print(f"  ✓ Model predicts all classes - Good!")
else:
    print(f"  ⚠ Model does not predict all classes - May need more training")
print("="*70 + "\n")

