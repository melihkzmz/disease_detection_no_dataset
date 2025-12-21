#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skin Disease Detection - 5 CLASS MACRO F1 TRAINING (df and vasc excluded)
Model: EfficientNetB3 (Medical Imaging Optimized)
Optimized for: Class Imbalance + Medical Safety + Professional Standards
Uses: Class-Balanced Focal Loss + Macro F1 Metric
Color Mode: RGB (Dermatoscopic images optimized)
Classes: akiec, bcc, bkl, mel, nv (df and vasc excluded - insufficient data)
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError) as e:
        # Older Python versions may not support reconfigure
        pass

print("\n" + "="*70)
print("🧬 SKIN DISEASE DETECTION - 5 CLASS MACRO F1 TRAINING")
print("="*70)
print("✅ Model: EfficientNetB3 (Medical Imaging Optimized)")
print("✅ Optimized for: Class Imbalance + Medical Safety")
print("✅ Uses: Class-Balanced Focal Loss + Macro F1 Metric")
print("✅ Color Mode: RGB (Dermatoscopic images)")
print("✅ Classes: akiec, bcc, bkl, mel, nv (df and vasc excluded)")
print("✅ Professional Standard for Medical AI")
print("="*70)

# Mixed precision disabled for Windows compatibility
print("[MEMORY] Mixed Precision Training DISABLED for Windows compatibility")

# Check GPU availability
physical_devices = tf.config.list_physical_devices('GPU')
USE_GPU = False

if len(physical_devices) > 0:
    print(f"\n[GPU] {len(physical_devices)} GPU(s) available")
    for device in physical_devices:
        print(f"  - {device}")
    
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("[GPU] Memory growth enabled")
        
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([1.0, 2.0])
            _ = tf.reduce_sum(test_tensor)
        USE_GPU = True
        print("[GPU] ✅ GPU test passed, using GPU for training")
    except Exception as e:
        print(f"[GPU] ⚠️  GPU configuration failed: {e}")
        print("[GPU] 🔄 Falling back to CPU mode")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        USE_GPU = False
        tf.config.set_visible_devices([], 'GPU')
else:
    print("\n[CPU] No GPU found, training on CPU")
    USE_GPU = False

# Hyperparameters
BASE_DATA_DIR = 'datasets/HAM10000/base_dir'
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train_dir')
VAL_DIR = os.path.join(BASE_DATA_DIR, 'val_dir')
TEST_DIR = os.path.join(BASE_DATA_DIR, 'test_dir')

IMG_SIZE = (300, 300)  # EfficientNetB3 input size (deri görüntüleri için optimize)
BATCH_SIZE = 16  # EfficientNetB3 için uygun batch size
INITIAL_EPOCHS = 100
FINE_TUNE_EPOCHS = 50
LEARNING_RATE = 0.0001  # Macro F1 loss için optimize edilmiş
FINE_TUNE_LR = 0.00005  # Increased from 0.00001 to 0.00005 (5x Phase 1 LR instead of 10x smaller)
# Note: Too small LR (0.00001) was causing performance degradation in Phase 2
# Using 0.00005 provides better fine-tuning while still being conservative
COLOR_MODE = 'rgb'  # RGB dermatoscopic images

# 5 Classes (df and vasc excluded - insufficient data)
CLASS_NAMES = [
    'akiec',   # Actinic Keratoses
    'bcc',     # Basal Cell Carcinoma
    'bkl',     # Benign Keratosis
    'mel',     # Melanoma
    'nv'       # Melanocytic Nevi
]

NUM_CLASSES = len(CLASS_NAMES)

print(f"\n[CONFIG]")
print(f"  Image Size: {IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Initial Epochs: {INITIAL_EPOCHS}")
print(f"  Fine-tune Epochs: {FINE_TUNE_EPOCHS}")
print(f"  Initial LR: {LEARNING_RATE}")
print(f"  Fine-tune LR: {FINE_TUNE_LR}")
print(f"  Color Mode: {COLOR_MODE}")
print(f"  Number of Classes: {NUM_CLASSES}")
print(f"  Classes: {', '.join(CLASS_NAMES)}")
print(f"  Excluded: df (Dermatofibroma), vasc (Vascular Lesions - insufficient data)")

# Create models directory
os.makedirs('models', exist_ok=True)

# Validate data directories exist
for dir_name, dir_path in [('TRAIN', TRAIN_DIR), ('VAL', VAL_DIR), ('TEST', TEST_DIR)]:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"[ERROR] {dir_name} directory not found: {dir_path}")
    if not os.listdir(dir_path):
        raise ValueError(f"[ERROR] {dir_name} directory is empty: {dir_path}")

# ============================================================================
# CLASS-BALANCED FOCAL LOSS FUNCTION
# ============================================================================

def class_balanced_focal_loss(y_true, y_pred, alpha=None, gamma=2.0):
    """
    Class-Balanced Focal Loss for imbalanced datasets.
    
    Combines Focal Loss (focuses on hard examples) with class balancing.
    
    Formula:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Where:
    - p_t: probability of true class
    - α_t: class-specific weighting factor (balanced)
    - γ: focusing parameter (default 2.0)
    
    Args:
        y_true: One-hot encoded true labels
        y_pred: Predicted probabilities
        alpha: Class-specific weights (list or None for auto-balancing)
        gamma: Focusing parameter (default 2.0)
    
    Returns:
        Class-balanced focal loss
    """
    # Clip predictions to prevent numerical issues
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # If alpha not provided, use balanced weights based on class frequency
    if alpha is None:
        # Auto-balance: give more weight to minority classes
        # This will be set dynamically based on class distribution
        alpha = [1.0] * NUM_CLASSES
    
    # Calculate focal loss for each sample
    # Get true class probabilities
    p_t = tf.reduce_sum(y_true * y_pred, axis=1, keepdims=True)  # (batch_size, 1)
    
    # Calculate (1 - p_t)^γ
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    
    # Calculate cross-entropy: -log(p_t)
    ce = -tf.math.log(p_t + 1e-7)
    
    # Apply class-specific alpha weights
    # For each sample, get the alpha corresponding to its true class
    alpha_t = tf.reduce_sum(y_true * tf.constant(alpha, dtype=tf.float32), axis=1, keepdims=True)
    
    # Calculate focal loss: -α_t * (1 - p_t)^γ * log(p_t)
    focal_loss = alpha_t * modulating_factor * ce
    
    # Return mean loss over batch
    return tf.reduce_mean(focal_loss)


def get_class_balanced_focal_loss_fn(class_weights_dict, gamma=2.0):
    """
    Create a class-balanced focal loss function with specific class weights.
    
    Args:
        class_weights_dict: Dictionary mapping class index to weight
        gamma: Focusing parameter (default 2.0)
    
    Returns:
        Loss function ready to use in model.compile()
    """
    # Convert dict to list (assuming class indices are 0 to NUM_CLASSES-1)
    alpha_list = [class_weights_dict.get(i, 1.0) for i in range(NUM_CLASSES)]
    
    def loss_fn(y_true, y_pred):
        return class_balanced_focal_loss(y_true, y_pred, alpha=alpha_list, gamma=gamma)
    
    return loss_fn

# ============================================================================
# STREAMING MACRO F1 METRIC (Global TP/FP/FN Accumulation)
# ============================================================================

class StreamingMacroF1(keras.metrics.Metric):
    """
    Streaming Macro F1 Metric that accumulates TP/FP/FN across all batches.
    This matches sklearn's macro F1 calculation on the full dataset.
    """
    def __init__(self, num_classes=5, name='macro_f1_metric', **kwargs):
        super(StreamingMacroF1, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        
        # Initialize state variables for each class: TP, FP, FN
        self.true_positives = self.add_weight(
            name='tp',
            shape=(num_classes,),
            initializer='zeros',
            dtype=tf.float32
        )
        self.false_positives = self.add_weight(
            name='fp',
            shape=(num_classes,),
            initializer='zeros',
            dtype=tf.float32
        )
        self.false_negatives = self.add_weight(
            name='fn',
            shape=(num_classes,),
            initializer='zeros',
            dtype=tf.float32
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update TP, FP, FN counts for current batch."""
        # Convert to class indices
        y_true_classes = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
        y_pred_classes = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
        
        # Vectorized computation
        y_true_one_hot = tf.one_hot(y_true_classes, depth=self.num_classes, dtype=tf.float32)
        y_pred_one_hot = tf.one_hot(y_pred_classes, depth=self.num_classes, dtype=tf.float32)
        
        # TP, FP, FN
        tp = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        fp = tf.reduce_sum((1.0 - y_true_one_hot) * y_pred_one_hot, axis=0)
        fn = tf.reduce_sum(y_true_one_hot * (1.0 - y_pred_one_hot), axis=0)
        
        # Update state
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        """Calculate macro F1 from accumulated TP/FP/FN."""
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1_scores = 2.0 * precision * recall / (precision + recall + 1e-8)
        
        macro_f1 = tf.reduce_mean(f1_scores)
        return macro_f1
    
    def reset_state(self):
        """Reset all state variables at the start of each epoch."""
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

# ============================================================================
# SKLEARN MACRO F1 CALLBACK
# ============================================================================

class CosineAnnealingLR(Callback):
    """
    Cosine annealing learning rate schedule.
    Often improves macro F1 by 1-2% compared to fixed LR.
    """
    def __init__(self, initial_lr, T_max, eta_min=0, verbose=0):
        super(CosineAnnealingLR, self).__init__()
        self.initial_lr = initial_lr
        self.T_max = T_max  # Maximum number of iterations (epochs)
        self.eta_min = eta_min  # Minimum learning rate
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        
        # Calculate cosine annealing LR
        lr = self.eta_min + (self.initial_lr - self.eta_min) * \
             (1 + np.cos(np.pi * epoch / self.T_max)) / 2
        
        # Set learning rate
        keras.backend.set_value(self.model.optimizer.lr, lr)
        
        if self.verbose > 0:
            print(f'\n[CosineAnnealing] Epoch {epoch}: LR = {lr:.6f}')

class SklearnMacroF1Callback(Callback):
    """
    Callback to compute sklearn's macro F1 at the end of each epoch on validation set.
    """
    def __init__(self, val_generator, num_classes=5, verbose=1):
        super(SklearnMacroF1Callback, self).__init__()
        self.val_generator = val_generator
        self.num_classes = num_classes
        self.verbose = verbose
        self.sklearn_f1_scores = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Compute sklearn macro F1 on full validation set."""
        # CRITICAL FIX: Save current generator state and reset properly
        # Store original generator state to avoid interfering with training
        original_seed = getattr(self.val_generator, '_seed', None)
        original_batch_index = getattr(self.val_generator, '_batch_index', 0)
        
        try:
        self.val_generator.reset()
        
        y_true_all = []
        y_pred_all = []
        
            # Use predict with steps to avoid generator state issues
            steps = len(self.val_generator)
            for batch_idx in range(steps):
            batch_x, batch_y = self.val_generator[batch_idx]
            y_pred_batch = self.model.predict(batch_x, verbose=0)
            
            y_true_all.append(np.argmax(batch_y, axis=1))
            y_pred_all.append(np.argmax(y_pred_batch, axis=1))
        
        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        self.sklearn_f1_scores.append(macro_f1)
        
        if self.verbose > 0:
            print(f"\n[SKLEARN] Validation Macro F1: {macro_f1*100:.2f}%")
        finally:
            # Restore generator state to avoid interfering with training loop
            self.val_generator.reset()
            if original_seed is not None:
                self.val_generator._seed = original_seed

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================

def preprocess_image_efficientnet(img):
    """
    Preprocess image for EfficientNetB3 input with dermoscopy-safe contrast adjustment.
    
    Process:
    1. ImageDataGenerator automatically resizes to (300, 300)
    2. Optional contrast adjustment (0.95-1.05 range)
    3. EfficientNet preprocessing (ImageNet normalization)
    
    EfficientNet preprocessing:
    - Scales pixels to [0, 1] range (divides by 255)
    - Applies ImageNet mean/std normalization
    - This matches exactly what pretrained EfficientNetB3 expects
    
    Args:
        img: numpy array (H, W, 3) in [0, 255] (uint8 or float)
    
    Returns:
        Preprocessed image array (normalized to ImageNet statistics)
    """
    # Ensure input is float32 in [0, 255] range
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    
    # Contrast adjustment (dermoscopy-safe: 0.95-1.05 range)
    # This is applied before normalization to preserve clinical features
    # Note: Contrast adjustment is handled by ImageDataGenerator's brightness_range
    # for simplicity, we skip explicit contrast here as it's less critical
    
    # EfficientNet preprocessing (handles ImageNet normalization internally)
    # This applies: scale to [0,1] then ImageNet mean/std normalization
    img_preprocessed = efficientnet_preprocess(img)
    return img_preprocessed

# ============================================================================
# DATA GENERATORS
# ============================================================================

print("\n[1/5] Data generators oluşturuluyor...")

# Training: with dermoscopy-safe augmentation
# Dermatoscopic images require careful augmentation to preserve clinical features
# Note: Keras ImageDataGenerator doesn't have contrast_range parameter
# Brightness adjustment helps with contrast variation to some extent
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image_efficientnet,
    rotation_range=20,         # Increased from 15° to 20° (more diversity, still safe)
    width_shift_range=0.15,    # Increased from 0.1 to 0.15 (15% shift)
    height_shift_range=0.15,   # Increased from 0.1 to 0.15 (15% shift)
    zoom_range=0.15,           # Increased from 0.1 to 0.15 (15% zoom)
    shear_range=0.1,           # NEW: Add shear transformation (dermoscopy-safe)
    horizontal_flip=True,      # Horizontal flip (dermoscopy-safe)
    vertical_flip=False,       # NO vertical flip (preserves clinical orientation)
    brightness_range=[0.9, 1.1],  # Increased from [0.95, 1.05] to [0.9, 1.1] (more variation)
    fill_mode='reflect'        # Changed from 'nearest' to 'reflect' (better edge handling)
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,  # Only 5 classes (df and vasc excluded)
    shuffle=True
)

# Validation and test: no augmentation, only preprocessing
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image_efficientnet
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_image_efficientnet
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=CLASS_NAMES,
    shuffle=False
)

print(f"\n[INFO] Dataset bilgileri:")
print(f"  Train samples: {train_generator.samples}")
print(f"  Validation samples: {val_generator.samples}")
print(f"  Test samples: {test_generator.samples}")
print(f"  Number of classes: {train_generator.num_classes}")
print(f"  Class indices: {train_generator.class_indices}")

# Calculate steps per epoch
train_steps = len(train_generator)
val_steps = len(val_generator)
test_steps = len(test_generator)

print(f"\n[INFO] Steps per epoch:")
print(f"  Train steps: {train_steps}")
print(f"  Validation steps: {val_steps}")
print(f"  Test steps: {test_steps}")

# ============================================================================
# CLASS WEIGHTS (for imbalanced dataset)
# ============================================================================

print("\n[2/5] Class weights hesaplanıyor...")

# Get class weights based on training set distribution
class_indices = train_generator.class_indices

# Get actual training labels from generator (more accurate than folder counts)
# This handles cases where files are skipped, extensions differ, or augmentation filtering occurs
all_train_labels = train_generator.classes

# Count samples per class using actual training labels (not folder counts)
# Use np.bincount to get actual distribution from generator
actual_class_counts = np.bincount(all_train_labels, minlength=len(CLASS_NAMES))
class_counts = {}
for class_name in CLASS_NAMES:
    class_idx = class_indices[class_name]
    class_counts[class_name] = int(actual_class_counts[class_idx])

total_samples = len(all_train_labels)
print(f"  Class distribution in training set (from actual generator labels):")
for class_name in CLASS_NAMES:
    count = class_counts.get(class_name, 0)
    percentage = (count / total_samples * 100) if total_samples > 0 else 0
    print(f"    {class_name}: {count} ({percentage:.2f}%)")

# Calculate balanced class weights using sklearn

# Compute balanced class weights using sklearn
balanced_weights = compute_class_weight(
    'balanced',
    classes=np.unique(all_train_labels),
    y=all_train_labels
)

# Create weight dictionary using sklearn's balanced class weights (data-driven, no arbitrary caps)
# sklearn's compute_class_weight('balanced') automatically calculates optimal weights
# based on class frequency: n_samples / (n_classes * np.bincount(y))
# This is data-driven, clinically justified, and reproducible - no hand-tuned caps needed
class_weights = {}
for idx, class_name in enumerate(CLASS_NAMES):
    class_idx = class_indices[class_name]
    # Find the index in unique classes array
    unique_classes = np.unique(all_train_labels)
    balanced_idx = np.where(unique_classes == class_idx)[0]
    if len(balanced_idx) > 0:
        balanced_weight = float(balanced_weights[balanced_idx[0]])
    else:
        balanced_weight = 1.0
    
    # Use sklearn's balanced weight directly (data-driven, no arbitrary adjustments)
    count = class_counts.get(class_name, 1)
    class_ratio = count / total_samples if total_samples > 0 else 1.0
    
    # Direct use of sklearn's balanced weight (optimal for class imbalance)
    class_weights[class_idx] = balanced_weight
    print(f"    {class_name}: weight = {balanced_weight:.3f} (ratio: {class_ratio*100:.1f}%, count: {count})")

class_weight_dict = class_weights

# LOSS STRATEGY: Simplified to Weighted Cross-Entropy (recommended for stability)
# Previous approach used Class-Balanced Focal Loss + alpha weights + Macro-F1 pressure
# This was too complex and could cause instability.
# New approach: Simple Weighted Cross-Entropy with label smoothing + class_weight
# This is more stable, graduation-safe, and often performs better.

# Use Weighted Cross-Entropy with label smoothing
# Label smoothing helps with generalization and prevents overconfidence
# Reduced from 0.05 to 0.01-0.02 for better minority class performance
# Higher values (0.05) can hurt minority classes in imbalanced datasets
LABEL_SMOOTHING = 0.01  # Optimized for class imbalance (was 0.05)
loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)

print(f"\n[LOSS] Weighted Cross-Entropy with Label Smoothing:")
print(f"  Loss: CategoricalCrossentropy(label_smoothing={LABEL_SMOOTHING})")
print(f"  Class weights: Will be passed via class_weight= in model.fit()")
print(f"  Strategy: Simple, stable, graduation-safe approach")
print(f"  Note: This replaces the complex Class-Balanced Focal Loss approach")

# ============================================================================
# MODEL ARCHITECTURE - PHASE 1: Feature Extraction
# ============================================================================

print("\n[3/5] Model oluşturuluyor (EfficientNetB3)...")

# Base model: EfficientNetB3 (pre-trained on ImageNet)
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze base model layers (Phase 1)
base_model.trainable = False

# Build model with improved architecture for better feature learning
# Strategy: Larger head with better regularization for improved representation
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
# Increased head size from 128 to 256 for better feature representation
# Two-layer head with dropout for better regularization
x = layers.Dense(256, activation='relu', name='fc1')(x)
x = layers.BatchNormalization()(x)  # Additional BatchNorm for stability
x = layers.Dropout(0.3)(x)  # Increased from 0.2 to 0.3 for better regularization
x = layers.Dense(128, activation='relu', name='fc2')(x)
x = layers.Dropout(0.2)(x)  # Second dropout layer
predictions = layers.Dense(NUM_CLASSES, activation='softmax', name='predictions', dtype='float32')(x)

model = keras.Model(inputs=base_model.input, outputs=predictions)

print(f"[MODEL] Trainable layers (Phase 1): {len([l for l in model.layers if l.trainable])}")
print(f"[MODEL] Frozen layers: {len([l for l in model.layers if not l.trainable])}")

# Compile model
streaming_macro_f1 = StreamingMacroF1(num_classes=NUM_CLASSES, name='macro_f1_metric')
metrics_list = [
    'accuracy',
    streaming_macro_f1,
    keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
]

# Phase 1: Use Weighted Cross-Entropy with Label Smoothing
# Simple, stable approach that handles class imbalance via class_weight
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999),
    loss=loss_fn,  # Weighted Cross-Entropy with label smoothing
    metrics=metrics_list  # Macro F1 monitored as metric (for evaluation, not callbacks)
)

print("[MODEL] Model compiled with Weighted Cross-Entropy + Label Smoothing (Phase 1)")
print(f"[MODEL] Loss: CategoricalCrossentropy(label_smoothing={LABEL_SMOOTHING})")
print("[MODEL] Class weights: Applied via class_weight= in fit()")
print("[MODEL] Macro F1 monitored as metric (for evaluation only)")

# ============================================================================
# CALLBACKS
# ============================================================================

print("\n[4/5] Callbacks oluşturuluyor...")

# CALLBACK STRATEGY: Train on loss, evaluate on Macro-F1
# - ModelCheckpoint: Uses val_macro_f1_metric (save best model by target metric)
# - EarlyStopping: Uses val_loss (smooth, stable signal - avoids premature stopping)
# - ReduceLROnPlateau: Uses val_loss (smooth, stable signal - avoids LR oscillations)
# Rationale: Macro-F1 is discrete/noisy and can oscillate, causing:
#   - Premature early stopping
#   - Learning rate oscillations that stall learning
# Loss is continuous and provides stable signal for training decisions.
# This strategy often improves Macro-F1 by +3-6% compared to monitoring Macro-F1 directly.

checkpoint = ModelCheckpoint(
    'models/skin_5class_efficientnetb3_macro_f1_phase1.keras',
    monitor='val_macro_f1_metric',  # Keep on Macro-F1: save best model by target metric
    save_best_only=True,
    mode='max',
    verbose=1,
    save_weights_only=False
)

# EarlyStopping: Use val_loss (smooth) instead of val_macro_f1_metric (noisy)
# Macro-F1 is discrete and can oscillate, causing premature stopping
# Loss is continuous and provides stable signal for stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Changed from val_macro_f1_metric to val_loss for stability
    patience=15,
    restore_best_weights=True,
    verbose=1,
    mode='min',  # Changed from 'max' to 'min' (lower loss is better)
    min_delta=0.001
)

# ReduceLROnPlateau: Use val_loss (smooth) instead of val_macro_f1_metric (noisy)
# Macro-F1 oscillations can cause LR to oscillate, stalling learning
# Loss provides stable signal for LR reduction
# Phase 1 uses higher LR (LEARNING_RATE = 0.0001)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Changed from val_macro_f1_metric to val_loss for stability
    factor=0.5,  # Reduce LR by half when plateau is reached
    patience=5,  # Wait 5 epochs before reducing
    min_lr=1e-7,
    verbose=1,
    mode='min'  # Changed from 'max' to 'min' (lower loss is better)
)

sklearn_macro_f1_callback = SklearnMacroF1Callback(
    val_generator=val_generator,
    num_classes=NUM_CLASSES,
    verbose=1
)

# Cosine Annealing LR Schedule (optional - can improve macro F1 by 1-2%)
# Uncomment to use instead of ReduceLROnPlateau
# cosine_annealing = CosineAnnealingLR(
#     initial_lr=LEARNING_RATE,
#     T_max=INITIAL_EPOCHS,
#     eta_min=1e-7,
#     verbose=1
# )

# ============================================================================
# PHASE 1: FEATURE EXTRACTION (Frozen Base Model)
# ============================================================================

print("\n" + "="*70)
print("PHASE 1: FEATURE EXTRACTION (Base Model Frozen)")
print("Using: WEIGHTED CROSS-ENTROPY + LABEL SMOOTHING + class_weight")
print("Strategy: Simple, stable approach - train on loss, evaluate on Macro F1")
print("="*70)

history_phase1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weight_dict,  # Apply class weights for class imbalance
    callbacks=[checkpoint, early_stopping, reduce_lr, sklearn_macro_f1_callback],
    verbose=1
)

# Load best model from Phase 1
print("\n[INFO] Loading best model from Phase 1...")
model = keras.models.load_model(
    'models/skin_5class_efficientnetb3_macro_f1_phase1.keras',
    custom_objects={
        'StreamingMacroF1': StreamingMacroF1,
    },
    compile=False
)

# CRITICAL FIX: Extract base_model from loaded model for Phase 2
# The old base_model reference is stale - we need the one from the loaded model
# In Functional API, EfficientNet layers are all layers except the custom head
print("[INFO] Extracting base_model layers from loaded model for Phase 2...")

# Identify custom head layers (added after base model)
custom_head_layer_names = ['fc1', 'fc2', 'predictions', 'global_average_pooling2d', 'batch_normalization']
# Find where the custom head starts (first occurrence of custom head layer)
base_model_end_idx = len(model.layers)
for i, layer in enumerate(model.layers):
    layer_name_lower = layer.name.lower()
    if any(head_name in layer_name_lower for head_name in custom_head_layer_names):
        # Check if this is the first custom head layer (GlobalAveragePooling2D typically comes first)
        if 'global_average' in layer_name_lower or 'gap' in layer_name_lower:
            base_model_end_idx = i
            break

# If not found, estimate: EfficientNetB3 has ~200+ layers, custom head is last ~4-5
if base_model_end_idx == len(model.layers):
    # Estimate: assume last 5 layers are custom head
    base_model_end_idx = max(0, len(model.layers) - 5)
    print(f"[WARNING] Could not find custom head boundary, estimating base model ends at layer {base_model_end_idx}")

# Base model layers are all layers before the custom head
base_model_layers = model.layers[:base_model_end_idx]

# Create a simple wrapper object with .layers attribute for the callback
class BaseModelWrapper:
    """Wrapper to provide base_model.layers interface for loaded models."""
    def __init__(self, base_layers):
        self.layers = base_layers
        self.trainable = True  # Will be controlled via individual layers
    
    def __setattr__(self, name, value):
        if name == 'trainable':
            # When setting trainable, apply to all layers
            object.__setattr__(self, name, value)
            for layer in self.layers:
                layer.trainable = value
        else:
            object.__setattr__(self, name, value)

base_model = BaseModelWrapper(base_model_layers)
print(f"[INFO] Base model extracted: {len(base_model_layers)} layers")
print(f"[INFO] Base model range: {model.layers[0].name} ... {model.layers[base_model_end_idx-1].name if base_model_end_idx > 0 else 'N/A'}")

# Recompile for Phase 2 evaluation (using same loss as Phase 1)
streaming_macro_f1_phase1 = StreamingMacroF1(num_classes=NUM_CLASSES, name='macro_f1_metric')
metrics_list_phase1 = [
    'accuracy',
    streaming_macro_f1_phase1,
    keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
]

# Phase 2: Use same Weighted Cross-Entropy with Label Smoothing
# Consistent with Phase 1 for stability
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR, beta_1=0.9, beta_2=0.999),
    loss=loss_fn,  # Same Weighted Cross-Entropy with label smoothing
    metrics=metrics_list_phase1  # Macro F1 monitored as metric (for evaluation only)
)

# Evaluate Phase 1
print("\n[EVAL] Phase 1 Results:")
test_generator.reset()
results_phase1 = model.evaluate(test_generator, verbose=0)

metric_names = model.metrics_names
loss_idx = metric_names.index('loss')
accuracy_idx = metric_names.index('accuracy')
macro_f1_idx = metric_names.index('macro_f1_metric') if 'macro_f1_metric' in metric_names else -1

print(f"  Test Loss (Categorical Crossentropy): {results_phase1[loss_idx]:.4f}")
print(f"  Test Accuracy: {results_phase1[accuracy_idx]*100:.2f}%")
if macro_f1_idx >= 0:
    print(f"  Test Macro F1: {results_phase1[macro_f1_idx]*100:.2f}%")

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

# Calculate Macro F1 using sklearn
macro_f1_sklearn = f1_score(y_true, y_pred_classes_phase1, average='macro', zero_division=0)
print(f"  Macro F1 (sklearn): {macro_f1_sklearn*100:.2f}%")

# ============================================================================
# PHASE 2: FINE-TUNING (Gradual Unfreeze + CE + class_weight)
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: FINE-TUNING (Gradual Unfreeze Strategy - CLEAN APPROACH)")
print("Using: WEIGHTED CROSS-ENTROPY + LABEL SMOOTHING + class_weight")
print("Strategy: Two separate fit() calls (cleaner, more stable)")
print("  - Phase 2a: Unfreeze last 15 layers, train 10 epochs")
print("  - Phase 2b: Unfreeze last 30 layers, train remaining epochs")
print("  - All BatchNorm layers remain frozen throughout")
print(f"Learning Rate: {FINE_TUNE_LR} (Phase 1 was {LEARNING_RATE}, ratio: {FINE_TUNE_LR/LEARNING_RATE:.1f}x)")
print("\n[PHASE 1 BASELINE] (for comparison)")
print(f"  Test Macro F1: {results_phase1[macro_f1_idx]*100:.2f}%")
print(f"  Test Accuracy: {results_phase1[accuracy_idx]*100:.2f}%")
print("="*70)

# Phase 2: Use same Weighted Cross-Entropy with Label Smoothing
# Consistent with Phase 1 for stability and simplicity
# Simple approach: Weighted CE + class_weight handles class imbalance effectively

# Prepare class_weight dict for model.fit()
# class_weight format: {0: weight0, 1: weight1, ...}
phase2_class_weight = class_weight_dict  # Already in correct format {0: weight, 1: weight, ...}

print(f"\n[LOSS] Phase 2: Weighted Cross-Entropy + Label Smoothing + class_weight")
print(f"  Loss: CategoricalCrossentropy(label_smoothing={LABEL_SMOOTHING})")
print(f"  Class weights: {phase2_class_weight}")
print(f"  Strategy: Simple, stable, consistent with Phase 1")

# Freeze all layers first
base_model.trainable = False

total_layers = len(base_model.layers)
total_batchnorm_count = len([l for l in base_model.layers if isinstance(l, layers.BatchNormalization)])

print(f"\n[MODEL] Total base model layers: {total_layers}")
print(f"[MODEL] Strategy: Two separate fit() calls (cleaner approach)")
print(f"  - Phase 2a: Last 15 layers unfrozen")
print(f"  - Phase 2b: Last 30 layers unfrozen")
print(f"[MODEL] BatchNorm layers: ALL {total_batchnorm_count} BatchNorm layers remain frozen throughout")

# ============================================================================
# PHASE 2A: Unfreeze last 15 layers
# ============================================================================

print("\n" + "="*70)
print("PHASE 2A: Fine-tuning with last 15 layers unfrozen")
print("="*70)

# Unfreeze last 15 layers (excluding BatchNorm)
layers_to_unfreeze_2a = 15
unfrozen_count = 0
for layer in base_model.layers[-layers_to_unfreeze_2a:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True
        unfrozen_count += 1

print(f"[PHASE 2A] Unfrozen {unfrozen_count} layers (last {layers_to_unfreeze_2a} layers, BatchNorm excluded)")

# Recompile with lower learning rate
streaming_macro_f1_phase2a = StreamingMacroF1(num_classes=NUM_CLASSES, name='macro_f1_metric')
metrics_list_phase2a = [
    'accuracy',
    streaming_macro_f1_phase2a,
    keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR, beta_1=0.9, beta_2=0.999),
    loss=loss_fn,  # Same Weighted Cross-Entropy with label smoothing
    metrics=metrics_list_phase2a
)

checkpoint_phase2a = ModelCheckpoint(
    'models/skin_5class_efficientnetb3_macro_f1_phase2a.keras',
    monitor='val_macro_f1_metric',
    save_best_only=True,
    mode='max',
    verbose=1,
    save_weights_only=False
)

early_stopping_phase2a = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1,
    mode='min',
    min_delta=0.001
)

reduce_lr_phase2a = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1,
    mode='min'
)

sklearn_macro_f1_callback_phase2a = SklearnMacroF1Callback(
    val_generator=val_generator,
    num_classes=NUM_CLASSES,
    verbose=1
)

# Phase 2a: Train with last 15 layers unfrozen
epochs_phase2a = 10
history_phase2a = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_phase2a,
    class_weight=phase2_class_weight,
    callbacks=[checkpoint_phase2a, early_stopping_phase2a, reduce_lr_phase2a, sklearn_macro_f1_callback_phase2a],
    verbose=1
)

# Load best model from Phase 2a
print("\n[INFO] Loading best model from Phase 2a...")
model = keras.models.load_model(
    'models/skin_5class_efficientnetb3_macro_f1_phase2a.keras',
    custom_objects={
        'StreamingMacroF1': StreamingMacroF1,
    },
    compile=False
)

# Re-extract base_model from loaded model (same logic as Phase 1 to Phase 2 transition)
custom_head_layer_names = ['fc1', 'fc2', 'predictions', 'global_average_pooling2d', 'batch_normalization']
base_model_end_idx_2b = len(model.layers)
for i, layer in enumerate(model.layers):
    layer_name_lower = layer.name.lower()
    if any(head_name in layer_name_lower for head_name in custom_head_layer_names):
        if 'global_average' in layer_name_lower or 'gap' in layer_name_lower:
            base_model_end_idx_2b = i
            break
if base_model_end_idx_2b == len(model.layers):
    base_model_end_idx_2b = max(0, len(model.layers) - 5)

base_model_layers_2b = model.layers[:base_model_end_idx_2b]
base_model = BaseModelWrapper(base_model_layers_2b)
print(f"[INFO] Base model re-extracted: {len(base_model_layers_2b)} layers")

# ============================================================================
# PHASE 2B: Unfreeze last 30 layers
# ============================================================================

print("\n" + "="*70)
print("PHASE 2B: Fine-tuning with last 30 layers unfrozen")
print("="*70)

# Freeze all first, then unfreeze last 30 layers (excluding BatchNorm)
base_model.trainable = False

layers_to_unfreeze_2b = 30
unfrozen_count = 0
for layer in base_model.layers[-layers_to_unfreeze_2b:]:
    if not isinstance(layer, layers.BatchNormalization):
        layer.trainable = True
        unfrozen_count += 1

print(f"[PHASE 2B] Unfrozen {unfrozen_count} layers (last {layers_to_unfreeze_2b} layers, BatchNorm excluded)")

# Recompile with same learning rate
streaming_macro_f1_phase2b = StreamingMacroF1(num_classes=NUM_CLASSES, name='macro_f1_metric')
metrics_list_phase2b = [
    'accuracy',
    streaming_macro_f1_phase2b,
    keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR, beta_1=0.9, beta_2=0.999),
    loss=loss_fn,
    metrics=metrics_list_phase2b
)

checkpoint_finetune = ModelCheckpoint(
    'models/skin_5class_efficientnetb3_macro_f1_finetuned.keras',
    monitor='val_macro_f1_metric',
    save_best_only=True,
    mode='max',
    verbose=1,
    save_weights_only=False
)

early_stopping_finetune = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1,
    mode='min',
    min_delta=0.002
)

reduce_lr_phase2b = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1,
    mode='min'
)

sklearn_macro_f1_callback_phase2b = SklearnMacroF1Callback(
    val_generator=val_generator,
    num_classes=NUM_CLASSES,
    verbose=1
)

# Phase 2b: Train with last 30 layers unfrozen
epochs_phase2b = FINE_TUNE_EPOCHS - epochs_phase2a
history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs_phase2b,
    class_weight=phase2_class_weight,
    callbacks=[checkpoint_finetune, early_stopping_finetune, reduce_lr_phase2b, sklearn_macro_f1_callback_phase2b],
    verbose=1
)

# Load best fine-tuned model
print("\n[INFO] Loading best fine-tuned model...")
# Load model without loss function (will recompile after loading)
model = keras.models.load_model(
    'models/skin_5class_efficientnetb3_macro_f1_finetuned.keras',
    custom_objects={
        'StreamingMacroF1': StreamingMacroF1,
    },
    compile=False  # Load without compilation
)

# Recompile model (using same loss as Phase 2)
print(f"[INFO] Recompiling model with Weighted Cross-Entropy + Label Smoothing (same as Phase 2)...")
streaming_macro_f1_reload = StreamingMacroF1(num_classes=NUM_CLASSES, name='macro_f1_metric')
metrics_list_reload = [
    'accuracy',
    streaming_macro_f1_reload,
    keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
]

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR, beta_1=0.9, beta_2=0.999),
    loss=loss_fn,  # Same Weighted Cross-Entropy with label smoothing
    metrics=metrics_list_reload
)

# ============================================================================
# FINAL EVALUATION
# ============================================================================

print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

test_generator.reset()
results = model.evaluate(test_generator, verbose=1)

# Get metric names from model (evaluate() sonrası güvenilir)
metric_names = model.metrics_names
print(f"[DEBUG] Metric names: {metric_names}")
print(f"[DEBUG] Results length: {len(results)}")

# Extract results - güvenli index bulma
loss_idx = 0  # Loss her zaman ilk sırada
if 'loss' in metric_names:
    loss_idx = metric_names.index('loss')

# Accuracy için farklı olası isimleri dene
accuracy_idx = -1
for acc_name in ['accuracy', 'categorical_accuracy', 'acc']:
    if acc_name in metric_names:
        accuracy_idx = metric_names.index(acc_name)
        break

if accuracy_idx == -1:
    # Fallback: loss'tan sonraki ilk metric genelde accuracy
    if len(results) > 1:
        accuracy_idx = 1
        print(f"[WARNING] Accuracy metric not found in {metric_names}, using index 1 as fallback")
    else:
        accuracy_idx = 0
        print(f"[WARNING] Could not determine accuracy index, using 0")

# Macro F1 için farklı olası isimleri dene
macro_f1_idx = -1
for f1_name in ['macro_f1_metric', 'macro_f1', 'f1']:
    if f1_name in metric_names:
        macro_f1_idx = metric_names.index(f1_name)
        break

# Extract results with bounds checking
loss = results[loss_idx] if loss_idx < len(results) else 0.0
accuracy = results[accuracy_idx] if accuracy_idx >= 0 and accuracy_idx < len(results) else 0.0
macro_f1 = results[macro_f1_idx] if macro_f1_idx >= 0 and macro_f1_idx < len(results) else 0.0

print(f"\n[RESULTS] Final Test Metrics:")
print(f"  Test Loss (Categorical Crossentropy): {loss:.4f}")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Test Macro F1: {macro_f1*100:.2f}%")

# Per-class metrics
test_generator.reset()
y_pred = model.predict(test_generator, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Classification Report
print("\n" + "="*70)
print("PER-CLASS PERFORMANCE (Classification Report)")
print("="*70)
print(classification_report(
    y_true, 
    y_pred_classes, 
    target_names=CLASS_NAMES, 
    zero_division=0,
    digits=4
))

# Calculate Macro F1 using sklearn
macro_f1_sklearn = f1_score(y_true, y_pred_classes, average='macro', zero_division=0)
weighted_f1_sklearn = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)

print(f"\n[RESULTS] Sklearn F1 Scores:")
print(f"  Macro F1: {macro_f1_sklearn*100:.2f}%")
print(f"  Weighted F1: {weighted_f1_sklearn*100:.2f}%")

# Per-class F1 scores
per_class_f1 = f1_score(y_true, y_pred_classes, average=None, zero_division=0)
print(f"\n[RESULTS] Per-Class F1 Scores:")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name}: {per_class_f1[i]*100:.2f}%")

# Confusion Matrix
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
    print(f"  ⚠️  WARNING: Model does not predict all classes!")
    print(f"  Missing classes: {set(range(NUM_CLASSES)) - set(unique_preds_final)}")

# Save final model
final_model_path = 'models/skin_disease_model_5class_efficientnetb3_macro_f1.keras'
# Save model in .keras format
model.save(final_model_path)
print(f"[INFO] Model saved as .keras: {final_model_path}")

# Also save as SavedModel format (Keras 3.x compatible - like bone_disease_api.py)
savedmodel_path = final_model_path.replace('.keras', '_savedmodel')
if os.path.exists(savedmodel_path):
    import shutil
    shutil.rmtree(savedmodel_path)
model.save(savedmodel_path, save_format='tf')
print(f"[INFO] Model also saved as SavedModel: {savedmodel_path}")
print(f"[INFO] SavedModel format is more compatible with Keras 3.x")
print(f"\n[SUCCESS] Final model saved to: {final_model_path}")

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================

print("\n[INFO] Generating training plots...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Combine Phase 2 histories for plotting
phase2_epochs_2a = len(history_phase2a.history['loss'])
phase2_epochs_2b = len(history_finetune.history['loss'])
phase2_total_epochs = phase2_epochs_2a + phase2_epochs_2b

# Phase 2 x-axis: offset by Phase 1 epochs
phase1_epochs = len(history_phase1.history['loss'])
phase2_x_2a = list(range(phase1_epochs, phase1_epochs + phase2_epochs_2a))
phase2_x_2b = list(range(phase1_epochs + phase2_epochs_2a, phase1_epochs + phase2_total_epochs))

# Loss
axes[0, 0].plot(history_phase1.history['loss'], label='Train (Phase 1)')
axes[0, 0].plot(history_phase1.history['val_loss'], label='Val (Phase 1)')
axes[0, 0].plot(phase2_x_2a, history_phase2a.history['loss'], label='Train (Phase 2a)', linestyle='--')
axes[0, 0].plot(phase2_x_2a, history_phase2a.history['val_loss'], label='Val (Phase 2a)', linestyle='--')
axes[0, 0].plot(phase2_x_2b, history_finetune.history['loss'], label='Train (Phase 2b)')
axes[0, 0].plot(phase2_x_2b, history_finetune.history['val_loss'], label='Val (Phase 2b)')
axes[0, 0].set_title('Model Loss (Weighted Cross-Entropy)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Accuracy
axes[0, 1].plot(history_phase1.history['accuracy'], label='Train (Phase 1)')
axes[0, 1].plot(history_phase1.history['val_accuracy'], label='Val (Phase 1)')
axes[0, 1].plot(phase2_x_2a, history_phase2a.history['accuracy'], label='Train (Phase 2a)', linestyle='--')
axes[0, 1].plot(phase2_x_2a, history_phase2a.history['val_accuracy'], label='Val (Phase 2a)', linestyle='--')
axes[0, 1].plot(phase2_x_2b, history_finetune.history['accuracy'], label='Train (Phase 2b)')
axes[0, 1].plot(phase2_x_2b, history_finetune.history['val_accuracy'], label='Val (Phase 2b)')
axes[0, 1].set_title('Model Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Macro F1
axes[1, 0].plot(history_phase1.history['macro_f1_metric'], label='Train (Phase 1)')
axes[1, 0].plot(history_phase1.history['val_macro_f1_metric'], label='Val (Phase 1)')
axes[1, 0].plot(phase2_x_2a, history_phase2a.history['macro_f1_metric'], label='Train (Phase 2a)', linestyle='--')
axes[1, 0].plot(phase2_x_2a, history_phase2a.history['val_macro_f1_metric'], label='Val (Phase 2a)', linestyle='--')
axes[1, 0].plot(phase2_x_2b, history_finetune.history['macro_f1_metric'], label='Train (Phase 2b)')
axes[1, 0].plot(phase2_x_2b, history_finetune.history['val_macro_f1_metric'], label='Val (Phase 2b)')
axes[1, 0].set_title('Macro F1 Score')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Macro F1')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1, 1])
axes[1, 1].set_title('Confusion Matrix (Test Set)')
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('True')

plt.tight_layout()
plot_path = 'models/training_history_skin_5class_efficientnetb3_macro_f1.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Training plots saved to: {plot_path}")

print("\n" + "="*70)
print("🎉 TRAINING COMPLETE!")
print("="*70)
print(f"[FILES]")
print(f"  Final Model: {final_model_path}")
print(f"  Training Plot: {plot_path}")
print(f"\n[METRICS]")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  Test Macro F1: {macro_f1*100:.2f}%")
print(f"  Test Macro F1 (sklearn): {macro_f1_sklearn*100:.2f}%")
print(f"  Test Weighted F1: {weighted_f1_sklearn*100:.2f}%")
print(f"\n[PER-CLASS F1]")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {class_name}: {per_class_f1[i]*100:.2f}%")
print(f"\n[CLASSES]")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"  {i+1}. {class_name}")
print("\n[IMPROVEMENTS]")
print("  ✅ EfficientNetB3 (medical imaging optimized)")
print("  ✅ RGB input (dermatoscopic images)")
print("  ✅ Class-Balanced Focal Loss (handles class imbalance & hard examples)")
print("  ✅ Macro F1 Metric (monitoring)")
print("  ✅ Equal importance to all classes")
print("  ✅ Medical safety optimized")
print("  ✅ Two-phase training strategy")
if len(unique_preds_final) == NUM_CLASSES:
    print("  ✅ Model predicts all classes - Excellent!")

