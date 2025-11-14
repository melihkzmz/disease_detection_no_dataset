#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check data distribution for Mendeley Eye Dataset"""

import os
from pathlib import Path
from collections import defaultdict

DATASET_DIR = 'datasets/Eye_Mendeley'

print("\n" + "="*70)
print("DATASET DISTRIBUTION ANALYSIS")
print("="*70)

stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})

for split in ['train', 'val', 'test']:
    split_path = os.path.join(DATASET_DIR, split)
    if not os.path.exists(split_path):
        continue
    
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            stats[class_name][split] = count

print("\n" + f"{'Class':<30} {'Train':<12} {'Val':<12} {'Test':<12} {'Total':<12} {'%':<10}")
print("-" * 90)

total_train = total_val = total_test = 0

for class_name in sorted(stats.keys()):
    train = stats[class_name]['train']
    val = stats[class_name]['val']
    test = stats[class_name]['test']
    total = train + val + test
    
    total_train += train
    total_val += val
    total_test += test
    
    percentage = (total / (total_train + total_val + total_test)) * 100 if (total_train + total_val + total_test) > 0 else 0
    
    print(f"{class_name:<30} {train:<12} {val:<12} {test:<12} {total:<12} {percentage:<10.1f}")

print("-" * 90)
grand_total = total_train + total_val + total_test
print(f"{'TOTAL':<30} {total_train:<12} {total_val:<12} {total_test:<12} {grand_total:<12} {100.0:<10.1f}")

print("\n[ANALYSIS]")
print(f"  Total Images: {grand_total:,}")
print(f"  Train: {total_train:,} ({total_train/grand_total*100:.1f}%)")
print(f"  Val: {total_val:,} ({total_val/grand_total*100:.1f}%)")
print(f"  Test: {total_test:,} ({total_test/grand_total*100:.1f}%)")

# Find imbalanced classes
print("\n[IMBALANCE DETECTION]")
avg_per_class = grand_total / len(stats) if len(stats) > 0 else 0
print(f"  Average per class: {avg_per_class:.0f}")

imbalanced = []
for class_name in sorted(stats.keys()):
    total = sum(stats[class_name].values())
    ratio = total / avg_per_class if avg_per_class > 0 else 0
    if ratio < 0.1:  # Less than 10% of average
        imbalanced.append((class_name, total, ratio))
    elif ratio > 3.0:  # More than 3x of average
        imbalanced.append((class_name, total, ratio))

if imbalanced:
    print("  ⚠️  Highly imbalanced classes:")
    for class_name, total, ratio in imbalanced:
        print(f"     - {class_name}: {total} images ({ratio:.2f}x average)")
else:
    print("  ✓ Classes are reasonably balanced")

print("="*70 + "\n")


