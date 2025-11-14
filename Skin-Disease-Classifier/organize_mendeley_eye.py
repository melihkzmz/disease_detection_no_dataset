#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mendeley Eye Disease Dataset Organization Script
Organizes Original + Augmented images into train/val/test structure
"""

import os
import sys
import shutil
from pathlib import Path
import random
from collections import defaultdict

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

print("\n" + "="*60)
print("MENDELEY EYE DISEASE DATASET ORGANIZER")
print("="*60)

# Directories
ORIGINAL_DIR = 'datasets/eye/Original Dataset'
AUGMENTED_DIR = 'datasets/eye/Augmented Dataset'
OUTPUT_DIR = 'datasets/Eye_Mendeley'

# Split ratios (only original images for val/test to prevent data leakage)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Class names (excluding empty Central Serous Chorioretinopathy)
CLASS_MAPPING = {
    'Central Serous Chorioretinopathy [Color Fundus]': None,  # Skip (empty)
    'Diabetic Retinopathy': 'Diabetic_Retinopathy',
    'Disc Edema': 'Disc_Edema',
    'Glaucoma': 'Glaucoma',
    'Healthy': 'Normal',
    'Macular Scar': 'Macular_Scar',
    'Myopia': 'Myopia',
    'Pterygium': 'Pterygium',
    'Retinal Detachment': 'Retinal_Detachment',
    'Retinitis Pigmentosa': 'Retinitis_Pigmentosa'
}

# Remove None values
CLASS_MAPPING = {k: v for k, v in CLASS_MAPPING.items() if v is not None}

print(f"\n[CONFIG]")
print(f"  Original Directory: {ORIGINAL_DIR}")
print(f"  Augmented Directory: {AUGMENTED_DIR}")
print(f"  Output Directory: {OUTPUT_DIR}")
print(f"  Split Ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
print(f"  Total Classes: {len(CLASS_MAPPING)}")

# Create output directories
for split in ['train', 'val', 'test']:
    for class_name in CLASS_MAPPING.values():
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

print("\n[INFO] Output directories created")

# Process each class
random.seed(42)
total_stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})

print("\n[PROCESSING] Organizing images...")

for original_class, target_class in CLASS_MAPPING.items():
    print(f"\n  Class: {target_class}")
    
    # Get original images
    original_path = os.path.join(ORIGINAL_DIR, original_class)
    original_images = []
    if os.path.exists(original_path):
        original_images = [f for f in os.listdir(original_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    # Get augmented images
    augmented_path = os.path.join(AUGMENTED_DIR, original_class)
    augmented_images = []
    if os.path.exists(augmented_path):
        augmented_images = [f for f in os.listdir(augmented_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"    Original: {len(original_images)} images")
    print(f"    Augmented: {len(augmented_images)} images")
    
    if len(original_images) == 0:
        print(f"    [SKIP] No images found")
        continue
    
    # Shuffle original images
    random.shuffle(original_images)
    
    # Split original images (for val/test)
    n_train = int(len(original_images) * TRAIN_RATIO)
    n_val = int(len(original_images) * VAL_RATIO)
    
    train_originals = original_images[:n_train]
    val_originals = original_images[n_train:n_train + n_val]
    test_originals = original_images[n_train + n_val:]
    
    # Copy validation images (original only)
    for img in val_originals:
        src = os.path.join(original_path, img)
        dst = os.path.join(OUTPUT_DIR, 'val', target_class, img)
        shutil.copy2(src, dst)
        total_stats[target_class]['val'] += 1
    
    # Copy test images (original only)
    for img in test_originals:
        src = os.path.join(original_path, img)
        dst = os.path.join(OUTPUT_DIR, 'test', target_class, img)
        shutil.copy2(src, dst)
        total_stats[target_class]['test'] += 1
    
    # Copy training images (original + corresponding augmented)
    # For each original training image, find its augmented versions
    for img in train_originals:
        # Copy original
        src = os.path.join(original_path, img)
        dst = os.path.join(OUTPUT_DIR, 'train', target_class, img)
        shutil.copy2(src, dst)
        total_stats[target_class]['train'] += 1
        
        # Find and copy augmented versions
        # Augmented images typically have the same base name with suffixes
        base_name = os.path.splitext(img)[0]
        
        for aug_img in augmented_images:
            # Check if augmented image belongs to this original
            if aug_img.startswith(base_name):
                src_aug = os.path.join(augmented_path, aug_img)
                dst_aug = os.path.join(OUTPUT_DIR, 'train', target_class, aug_img)
                if not os.path.exists(dst_aug):  # Avoid duplicates
                    shutil.copy2(src_aug, dst_aug)
                    total_stats[target_class]['train'] += 1
    
    print(f"    [DONE] Train: {total_stats[target_class]['train']}, "
          f"Val: {total_stats[target_class]['val']}, "
          f"Test: {total_stats[target_class]['test']}")

# Print final statistics
print("\n" + "="*60)
print("ORGANIZATION COMPLETE")
print("="*60)

print("\n[SUMMARY] Dataset Distribution:")
print(f"\n{'Class':<25} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
print("-" * 65)

total_train = 0
total_val = 0
total_test = 0

for class_name in sorted(total_stats.keys()):
    stats = total_stats[class_name]
    train_count = stats['train']
    val_count = stats['val']
    test_count = stats['test']
    total_count = train_count + val_count + test_count
    
    total_train += train_count
    total_val += val_count
    total_test += test_count
    
    print(f"{class_name:<25} {train_count:<10} {val_count:<10} {test_count:<10} {total_count:<10}")

print("-" * 65)
print(f"{'TOTAL':<25} {total_train:<10} {total_val:<10} {total_test:<10} {total_train + total_val + total_test:<10}")

print("\n[INFO] Class distribution percentages:")
for class_name in sorted(total_stats.keys()):
    total_class = sum(total_stats[class_name].values())
    percentage = (total_class / (total_train + total_val + total_test)) * 100
    print(f"  {class_name}: {total_class} ({percentage:.1f}%)")

print("\n[SUCCESS] Dataset organized successfully!")
print(f"  Output location: {OUTPUT_DIR}/")
print(f"  Total images: {total_train + total_val + total_test}")
print(f"  Total classes: {len(total_stats)}")
print("\n[NEXT STEP] Run: python train_mendeley_eye.py")
print("="*60 + "\n")

