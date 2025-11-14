#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check Bone Dataset Class Distribution
"""
import os
from collections import Counter

TRAIN_DIR = 'datasets/bone/Bone_4Class_Final/train'
VAL_DIR = 'datasets/bone/Bone_4Class_Final/val'
TEST_DIR = 'datasets/bone/Bone_4Class_Final/test'

CLASS_NAMES = ['Normal', 'Fracture', 'Benign_Tumor', 'Malignant_Tumor']

def count_images(directory):
    """Count images per class in a directory"""
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(directory, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            counts[class_name] = len(files)
        else:
            counts[class_name] = 0
    return counts

print("\n" + "="*70)
print("BONE DATASET CLASS DISTRIBUTION")
print("="*70)

train_counts = count_images(TRAIN_DIR)
val_counts = count_images(VAL_DIR)
test_counts = count_images(TEST_DIR)

total_train = sum(train_counts.values())
total_val = sum(val_counts.values())
total_test = sum(test_counts.values())
total_all = total_train + total_val + total_test

print("\n[TRAIN]")
for class_name in CLASS_NAMES:
    count = train_counts[class_name]
    pct = (count / total_train * 100) if total_train > 0 else 0
    print(f"  {class_name:20s}: {count:5d} ({pct:5.1f}%)")

print(f"\n  Total: {total_train}")

print("\n[VAL]")
for class_name in CLASS_NAMES:
    count = val_counts[class_name]
    pct = (count / total_val * 100) if total_val > 0 else 0
    print(f"  {class_name:20s}: {count:5d} ({pct:5.1f}%)")

print(f"\n  Total: {total_val}")

print("\n[TEST]")
for class_name in CLASS_NAMES:
    count = test_counts[class_name]
    pct = (count / total_test * 100) if total_test > 0 else 0
    print(f"  {class_name:20s}: {count:5d} ({pct:5.1f}%)")

print(f"\n  Total: {total_test}")

print("\n[TOTAL]")
for class_name in CLASS_NAMES:
    count = train_counts[class_name] + val_counts[class_name] + test_counts[class_name]
    pct = (count / total_all * 100) if total_all > 0 else 0
    print(f"  {class_name:20s}: {count:5d} ({pct:5.1f}%)")

print(f"\n  Total: {total_all}")

# Calculate imbalance ratio
max_train = max(train_counts.values())
min_train = min([c for c in train_counts.values() if c > 0])
imbalance_ratio = max_train / min_train if min_train > 0 else float('inf')

print(f"\n[IMBALANCE]")
print(f"  Max class (train): {max([(v, k) for k, v in train_counts.items()], key=lambda x: x[0])[1]} ({max_train})")
print(f"  Min class (train): {min([(v, k) for k, v in train_counts.items() if v > 0], key=lambda x: x[0])[1]} ({min_train})")
print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
print(f"  {'⚠️ EXTREME IMBALANCE!' if imbalance_ratio > 10 else '⚠️ Severe imbalance' if imbalance_ratio > 5 else '✓ Moderate imbalance'}")

# Recommendations
print(f"\n[RECOMMENDATIONS]")
if imbalance_ratio > 10:
    print("  🔴 EXTREME IMBALANCE DETECTED!")
    print("     - Need MUCH more aggressive class weights")
    print("     - Consider oversampling minority classes")
    print("     - May need exponential class weights (power > 1.5)")
    print("     - Consider focal loss gamma > 3.0")
elif imbalance_ratio > 5:
    print("  🟠 Severe imbalance detected")
    print("     - Need aggressive class weights")
    print("     - Focal loss recommended")
else:
    print("  🟢 Moderate imbalance - standard techniques should work")

print("\n" + "="*70)

