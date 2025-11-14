#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Show detailed class distribution for Mendeley Eye Dataset"""

import os
from pathlib import Path
from collections import defaultdict

DATASET_DIR = 'datasets/Eye_Mendeley'

print("\n" + "="*80)
print("DETAILED CLASS DISTRIBUTION ANALYSIS")
print("="*80)

stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})

for split in ['train', 'val', 'test']:
    split_path = os.path.join(DATASET_DIR, split)
    if not os.path.exists(split_path):
        continue
    
    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            stats[class_name][split] = count
            stats[class_name]['total'] += count

print("\n" + f"{'Class Name':<35} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10} {'%':<10}")
print("-" * 95)

total_all = sum(s['total'] for s in stats.values())

for class_name in sorted(stats.keys()):
    train = stats[class_name]['train']
    val = stats[class_name]['val']
    test = stats[class_name]['test']
    total = stats[class_name]['total']
    
    percentage = (total / total_all * 100) if total_all > 0 else 0
    
    print(f"{class_name:<35} {train:<10} {val:<10} {test:<10} {total:<10} {percentage:<10.1f}")

print("-" * 95)
grand_total = sum(s['total'] for s in stats.values())
print(f"{'TOTAL':<35} {sum(s['train'] for s in stats.values()):<10} "
      f"{sum(s['val'] for s in stats.values()):<10} "
      f"{sum(s['test'] for s in stats.values()):<10} "
      f"{grand_total:<10} {100.0:<10.1f}")

print("\n" + "="*80)
print("ANALYSIS & RECOMMENDATIONS")
print("="*80)

# Calculate average
num_classes = len(stats)
avg_per_class = grand_total / num_classes if num_classes > 0 else 0

print(f"\nAverage per class: {avg_per_class:.0f} images")

print("\n[CLASS CATEGORIZATION]")

# Very low (less than 5% of average)
very_low = []
low = []
medium = []
high = []

for class_name in sorted(stats.keys()):
    total = stats[class_name]['total']
    ratio = total / avg_per_class if avg_per_class > 0 else 0
    
    if ratio < 0.05:
        very_low.append((class_name, total, ratio))
    elif ratio < 0.3:
        low.append((class_name, total, ratio))
    elif ratio < 1.5:
        medium.append((class_name, total, ratio))
    else:
        high.append((class_name, total, ratio))

if very_low:
    print("\n  [CRITICAL - Consider Removing] (< 5% of average):")
    for class_name, total, ratio in very_low:
        print(f"    - {class_name}: {total} images ({ratio:.2%} of average)")

if low:
    print("\n  [LOW - Needs Oversampling] (< 30% of average):")
    for class_name, total, ratio in low:
        print(f"    - {class_name}: {total} images ({ratio:.2%} of average)")

if medium:
    print("\n  [MEDIUM - Acceptable] (30-150% of average):")
    for class_name, total, ratio in medium:
        print(f"    - {class_name}: {total} images ({ratio:.2%} of average)")

if high:
    print("\n  [HIGH - May need downsampling] (> 150% of average):")
    for class_name, total, ratio in high:
        print(f"    - {class_name}: {total} images ({ratio:.2%} of average)")

print("\n[RECOMMENDATIONS]")

if very_low:
    print("  1. Remove classes with < 5% of average data:")
    for class_name, total, _ in very_low:
        print(f"     - {class_name} ({total} images)")

if low:
    print("\n  2. Classes that need aggressive oversampling:")
    for class_name, total, _ in low:
        print(f"     - {class_name} ({total} images)")

print("\n[SUGGESTED ACTIONS]")
print("  - Remove: Pterygium (too few samples)")
if low:
    print("  - Consider removing: " + ", ".join([c[0] for c in low[:2]]))
print("  - Keep all medium and high classes")

print("="*80 + "\n")


