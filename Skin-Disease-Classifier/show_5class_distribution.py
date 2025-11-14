#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show distribution for 5 selected classes"""

import os
from pathlib import Path
from collections import defaultdict

DATASET_DIR = 'datasets/Eye_Mendeley'

# Selected classes (Medium + High only)
SELECTED_CLASSES = [
    'Macular_Scar',           # Medium
    'Myopia',                 # Medium
    'Diabetic_Retinopathy',   # High
    'Glaucoma',               # High
    'Normal'                  # High
]

print("\n" + "="*80)
print("5-CLASS DATASET DISTRIBUTION (Medium + High Classes Only)")
print("="*80)

stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})

for split in ['train', 'val', 'test']:
    split_path = os.path.join(DATASET_DIR, split)
    if not os.path.exists(split_path):
        continue
    
    for class_name in SELECTED_CLASSES:
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            stats[class_name][split] = count
            stats[class_name]['total'] += count

print("\n" + f"{'Class Name':<30} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10} {'%':<10}")
print("-" * 90)

total_all = sum(s['total'] for s in stats.values())

for class_name in SELECTED_CLASSES:
    train = stats[class_name]['train']
    val = stats[class_name]['val']
    test = stats[class_name]['test']
    total = stats[class_name]['total']
    
    percentage = (total / total_all * 100) if total_all > 0 else 0
    
    print(f"{class_name:<30} {train:<10} {val:<10} {test:<10} {total:<10} {percentage:<10.1f}")

print("-" * 90)
grand_total = sum(s['total'] for s in stats.values())
print(f"{'TOTAL':<30} {sum(s['train'] for s in stats.values()):<10} "
      f"{sum(s['val'] for s in stats.values()):<10} "
      f"{sum(s['test'] for s in stats.values()):<10} "
      f"{grand_total:<10} {100.0:<10.1f}")

print("\n" + "="*80)
print("BALANCE ANALYSIS")
print("="*80)

# Calculate average
num_classes = len(stats)
avg_per_class = grand_total / num_classes if num_classes > 0 else 0

print(f"\nAverage per class: {avg_per_class:.0f} images")

print("\n[CLASS BALANCE RATIO]")
for class_name in SELECTED_CLASSES:
    total = stats[class_name]['total']
    ratio = total / avg_per_class if avg_per_class > 0 else 0
    status = "GOOD" if 0.8 <= ratio <= 1.2 else "ACCEPTABLE" if 0.6 <= ratio <= 1.4 else "IMBALANCED"
    print(f"  {class_name:<30} {ratio:.2f}x average ({status})")

min_ratio = min(stats[c]['total'] / avg_per_class for c in SELECTED_CLASSES)
max_ratio = max(stats[c]['total'] / avg_per_class for c in SELECTED_CLASSES)

print(f"\n[OVERALL BALANCE]")
print(f"  Min ratio: {min_ratio:.2f}x")
print(f"  Max ratio: {max_ratio:.2f}x")
print(f"  Ratio spread: {max_ratio/min_ratio:.2f}x")

if max_ratio/min_ratio < 2.0:
    print("  -> EXCELLENT BALANCE! Very balanced dataset.")
elif max_ratio/min_ratio < 3.0:
    print("  -> GOOD BALANCE. Acceptable for training.")
else:
    print("  -> MODERATE IMBALANCE. May need class weights.")

print("\n[REMOVED CLASSES]")
REMOVED = ['Pterygium', 'Disc_Edema', 'Retinal_Detachment', 'Retinitis_Pigmentosa']
for removed in REMOVED:
    removed_path_train = os.path.join(DATASET_DIR, 'train', removed)
    removed_total = 0
    if os.path.exists(removed_path_train):
        removed_total = len([f for f in os.listdir(removed_path_train) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    print(f"  - {removed}: {removed_total} training images (will be removed)")

print("="*80 + "\n")


