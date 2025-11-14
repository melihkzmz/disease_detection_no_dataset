#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bone Disease Detection - ULTRA AGGRESSIVE TRAINING
For EXTREME class imbalance (when Phase 1 fails completely)
Uses: Exponential class weights, Higher focal gamma, Balanced sampling
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB2
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
print("BONE DISEASE DETECTION - ULTRA AGGRESSIVE TRAINING")
print("For EXTREME Class Imbalance (Phase 1 Failed)")
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

# Hyperparameters - ULTRA AGGRESSIVE
TRAIN_DIR = 'datasets/bone/Bone_4Class_Final/train'
VAL_DIR = 'datasets/bone/Bone_4Class_Final/val'
TEST_DIR = 'datasets/bone/Bone_4Class_Final/test'

IMG_SIZE = (512, 512)
BATCH_SIZE = 16
INITIAL_EPOCHS = 200  # Even more epochs
FINE_TUNE_EPOCHS = 100
LEARNING_RATE = 0.0001  # Even lower
FINE_TUNE_LR = 0.00001

# Focal Loss parameters - MORE AGGRESSIVE
FOCAL_ALPHA = 0.5  # Higher alpha for minority classes
FOCAL_GAMMA = 3.0  # Higher gamma (more focus on hard examples)

# Class weights - EXPONENTIAL (ultra aggressive)
WEIGHT_POWER = 2.0  # Power > 1.5 = exponential, very aggressive

CLASS_NAMES = [
    'Normal',
    'Fracture',
    'Benign_Tumor',
    'Malignant_Tumor'
]

NUM_CLASSES = len(CLASS_NAMES)

print(f"\n[CONFIG]")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Initial Epochs: {INITIAL_EPOCHS}")
print(f"  Fine-tune Epochs: {FINE_TUNE_EPOCHS}")
print(f"  Initial LR: {LEARNING_RATE}")
print(f"  Fine-tune LR: {FINE_TUNE_LR}")
print(f"  Focal Loss Alpha: {FOCAL_ALPHA}, Gamma: {FOCAL_GAMMA}")
print(f"  Class Weight Power: {WEIGHT_POWER} (EXPONENTIAL)")
print(f"  Number of Classes: {NUM_CLASSES}")
print(f"  Classes: {', '.join(CLASS_NAMES)}")

# Focal Loss with higher gamma
def focal_loss(alpha=0.5, gamma=3.0):
    """
    Focal Loss: Ultra aggressive for extreme imbalance
    Higher gamma = more focus on hard examples
    """
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.ones_like(y_true) * alpha
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_t, 1 - alpha_t)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        focal_loss_value = focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss_value, axis=1))
    return loss

# Data augmentation - LESS aggressive (preserve minority class examples)
print("\n[DATA] Creating X-ray optimized data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,  # Less rotation
    width_shift_range=0.05,  # Less shift
    height_shift_range=0.05,
    shear_range=0.02,  # Less shear
    zoom_range=0.05,  # Less zoom
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='constant',
    cval=0.0,
    brightness_range=[0.95, 1.05],  # Less brightness variation
    channel_shift_range=2  # Less channel shift
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
    seed=42,
    color_mode='rgb'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False,
    color_mode='rgb'
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False,
    color_mode='rgb'
)

print(f"\n[DATA] Generators created:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Test samples: {test_generator.samples}")

# ULTRA AGGRESSIVE class weights - EXPONENTIAL
print("\n[DATA] Calculating ULTRA AGGRESSIVE (EXPONENTIAL) class weights...")
class_counts = np.bincount(train_generator.classes)
total_samples = len(train_generator.classes)

print("\n[INFO] Class distribution:")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name}: {class_counts[i]} ({class_counts[i]/total_samples*100:.1f}%)")

# EXPONENTIAL weighting (much more aggressive than sqrt)
max_count = np.max(class_counts)
# Power > 1.5 creates exponential weighting
class_weights = np.power(max_count / (class_counts + 1), WEIGHT_POWER)
class_weights = class_weights / np.min(class_weights)  # Normalize
class_weights_dict = dict(zip(range(NUM_CLASSES), class_weights))

print("\n[INFO] ULTRA AGGRESSIVE class weights (EXPONENTIAL, power={}):".format(WEIGHT_POWER))
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name}: {class_weights_dict[i]:.3f} (samples: {class_counts[i]})")
    
# Show weight ratio
max_weight = max(class_weights_dict.values())
min_weight = min(class_weights_dict.values())
print(f"\n  Weight ratio (max/min): {max_weight/min_weight:.2f}:1")

# Build model
print("\n[MODEL] Building EfficientNetB2 model...")

base_model = EfficientNetB2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

# Architecture with more capacity
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile with ULTRA AGGRESSIVE focal loss
focal_loss_fn = focal_loss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999),
    loss=focal_loss_fn,
    metrics=[
        'accuracy',
        keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
    ]
)

print("\n[MODEL] Model architecture:")
model.summary()
print(f"\n[MODEL] Total parameters: {model.count_params():,}")

# Callbacks
os.makedirs('models', exist_ok=True)

checkpoint_initial = ModelCheckpoint(
    'models/bone_4class_ultra_aggressive_initial.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1,
    save_weights_only=False
)

# VERY high patience - model needs lots of time
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=100,  # Very high - let model learn minority classes
    restore_best_weights=True,
    verbose=1,
    mode='max',
    min_delta=0.0001  # Very small improvement threshold
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.25,  # More aggressive reduction
    patience=20,  # Wait longer
    min_lr=1e-9,
    verbose=1,
    cooldown=10
)

# Phase 1: Initial Training
print("\n" + "="*70)
print("PHASE 1: INITIAL TRAINING (ULTRA AGGRESSIVE)")
print("Using: FOCAL LOSS (gamma=3.0) + EXPONENTIAL Class Weights (power=2.0)")
print("="*70)

history_initial = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights_dict,  # ULTRA AGGRESSIVE
    callbacks=[checkpoint_initial, early_stopping, reduce_lr],
    verbose=1
)

# Load best model from phase 1
print("\n[INFO] Loading best model from Phase 1...")
model = keras.models.load_model(
    'models/bone_4class_ultra_aggressive_initial.keras',
    custom_objects={'loss': focal_loss_fn}
)

# Check intermediate results
print("\n[EVAL] Phase 1 Results:")
loss, accuracy, top_2_acc = model.evaluate(test_generator, verbose=0)
print(f"  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Top-2 Accuracy: {top_2_acc*100:.2f}%")

# Check if model is learning all classes
test_generator.reset()
y_pred_phase1 = model.predict(test_generator, verbose=0)
y_pred_classes_phase1 = np.argmax(y_pred_phase1, axis=1)
y_true = test_generator.classes

unique_preds = np.unique(y_pred_classes_phase1)
print(f"\n[INFO] Phase 1: Model predicts {len(unique_preds)} unique classes out of {NUM_CLASSES}")
print(f"  Predicted classes: {unique_preds}")
if len(unique_preds) == NUM_CLASSES:
    print("  ✅ EXCELLENT: Model is learning all classes!")
else:
    print(f"  ⚠️  WARNING: Model does not predict all classes!")
    print(f"  Missing classes: {set(range(NUM_CLASSES)) - set(unique_preds)}")
    if len(unique_preds) == 1:
        print(f"\n  🔴 CRITICAL: Model still only predicts 1 class!")
        print(f"     This indicates extreme dataset imbalance.")
        print(f"     Consider:")
        print(f"     1. Checking dataset distribution (run check_bone_dataset_distribution.py)")
        print(f"     2. Manual oversampling of minority classes")
        print(f"     3. Even more aggressive weights (power=3.0 or higher)")
        print(f"     4. Different architecture (ResNet, DenseNet)")

# Phase 2: Fine-tuning (only if Phase 1 shows progress)
if len(unique_preds) >= 2:
    print("\n" + "="*70)
    print("PHASE 2: FINE-TUNING (Proceeding because Phase 1 shows progress)")
    print("="*70)
    
    base_model.trainable = True
    freeze_until = len(base_model.layers) - 100
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    print(f"\n[MODEL] Unfrozen layers: {len([l for l in base_model.layers if l.trainable])}")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR, beta_1=0.9, beta_2=0.999),
        loss=focal_loss_fn,
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
        ]
    )
    
    checkpoint_finetune = ModelCheckpoint(
        'models/bone_4class_ultra_aggressive_finetuned.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=False
    )
    
    early_stopping_finetune = EarlyStopping(
        monitor='val_accuracy',
        patience=80,
        restore_best_weights=True,
        verbose=1,
        mode='max',
        min_delta=0.0001
    )
    
    history_finetune = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights_dict,
        callbacks=[checkpoint_finetune, early_stopping_finetune, reduce_lr],
        verbose=1
    )
    
    print("\n[INFO] Loading best fine-tuned model...")
    model = keras.models.load_model(
        'models/bone_4class_ultra_aggressive_finetuned.keras',
        custom_objects={'loss': focal_loss_fn}
    )
else:
    print("\n[INFO] Skipping Phase 2 (Fine-tuning) because Phase 1 did not show sufficient progress.")
    print("       Model may need more aggressive techniques or dataset balancing.")

# Final evaluation
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

loss, accuracy, top_2_acc = model.evaluate(test_generator, verbose=1)

print(f"\n[RESULTS] Final Test Metrics:")
print(f"  Test Loss: {loss:.4f}")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Top-2 Accuracy: {top_2_acc*100:.2f}%")

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
if len(unique_preds_final) == NUM_CLASSES:
    print("  ✅ EXCELLENT: Model predicts all classes!")
else:
    print(f"  ⚠️  Model does not predict all classes!")
    print(f"  Missing classes: {set(range(NUM_CLASSES)) - set(unique_preds_final)}")
    
    # Show per-class predictions
    from collections import Counter
    pred_counts = Counter(y_pred_classes)
    print("\n[INFO] Per-class prediction counts:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {pred_counts.get(i, 0)} predictions")

# Save final model
final_model_path = 'models/bone_disease_model_4class_ultra_aggressive.keras'
model.save(final_model_path)
print(f"\n[SUCCESS] Final model saved to: {final_model_path}")

# Plot training history
print("\n[INFO] Generating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

epochs_initial = range(1, len(history_initial.history['accuracy']) + 1)
if len(unique_preds) >= 2:
    epochs_finetune = range(len(history_initial.history['accuracy']) + 1, 
                            len(history_initial.history['accuracy']) + len(history_finetune.history['accuracy']) + 1)
else:
    epochs_finetune = []

# Accuracy
axes[0, 0].plot(epochs_initial, history_initial.history['accuracy'], 'b-', label='Train (Phase 1)', linewidth=2)
axes[0, 0].plot(epochs_initial, history_initial.history['val_accuracy'], 'b--', label='Val (Phase 1)', linewidth=2)
if len(epochs_finetune) > 0:
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
if len(epochs_finetune) > 0:
    axes[0, 1].plot(epochs_finetune, history_finetune.history['loss'], 'r-', label='Train (Phase 2)', linewidth=2)
    axes[0, 1].plot(epochs_finetune, history_finetune.history['val_loss'], 'r--', label='Val (Phase 2)', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].set_ylabel('Loss (Focal Loss)', fontsize=12)
axes[0, 1].set_title('Model Loss (Focal Loss)', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Top-2 Accuracy
axes[1, 0].plot(epochs_initial, history_initial.history['top_2_accuracy'], 'b-', label='Train (Phase 1)', linewidth=2)
axes[1, 0].plot(epochs_initial, history_initial.history['val_top_2_accuracy'], 'b--', label='Val (Phase 1)', linewidth=2)
if len(epochs_finetune) > 0:
    axes[1, 0].plot(epochs_finetune, history_finetune.history['top_2_accuracy'], 'r-', label='Train (Phase 2)', linewidth=2)
    axes[1, 0].plot(epochs_finetune, history_finetune.history['val_top_2_accuracy'], 'r--', label='Val (Phase 2)', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=12)
axes[1, 0].set_ylabel('Top-2 Accuracy', fontsize=12)
axes[1, 0].set_title('Model Top-2 Accuracy', fontsize=14, fontweight='bold')
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
plot_path = 'models/training_history_bone_4class_ultra_aggressive.png'
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
print(f"  Top-2 Accuracy: {top_2_acc*100:.2f}%")
print(f"  Total Classes: {NUM_CLASSES}")
print(f"  Total Parameters: {model.count_params():,}")
print(f"\n[CLASSES]")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {i+1}. {class_name}")
print(f"\n[ULTRA AGGRESSIVE TECHNIQUES]")
print(f"  ✓ Focal Loss (gamma=3.0, alpha=0.5)")
print(f"  ✓ EXPONENTIAL Class Weights (power=2.0)")
print(f"  ✓ Very Low Learning Rate (0.0001)")
print(f"  ✓ Very High Patience (100 epochs)")
print(f"  ✓ More Epochs (200+100)")
print(f"  ✓ Less aggressive augmentation (preserve minority examples)")
if len(unique_preds_final) == NUM_CLASSES:
    print(f"\n  ✅ Model predicts all classes - Excellent!")
elif len(unique_preds_final) >= 2:
    print(f"\n  🟡 Model predicts {len(unique_preds_final)} classes - Progress but needs more")
else:
    print(f"\n  🔴 Model still only predicts 1 class - Dataset may need manual balancing")
print("="*70 + "\n")

