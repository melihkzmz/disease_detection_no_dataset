#!/usr/bin/env python3
"""
9 sınıflı birleştirilmiş veri setinin final istatistiklerini kontrol et
"""
from pathlib import Path

OUTPUT_DIR = Path("datasets/bone/Bone_9Class_Combined")

FINAL_CLASSES = [
    'Normal',
    'Osteochondroma',
    'Osteosarcoma',
    'Multiple_Osteochondromas',
    'Other_Benign',
    'Simple_Bone_Cyst',
    'Giant_Cell_Tumor',
    'Other_Malignant',
    'Fracture'
]

print("=" * 80)
print("9 SINIFLI VERI SETI - FINAL ISTATISTIKLER")
print("=" * 80)

total_all = 0
for split_name in ['train', 'val', 'test']:
    split_dir = OUTPUT_DIR / split_name
    if not split_dir.exists():
        continue
    
    print(f"\n{split_name.upper()}:")
    
    split_total = 0
    for class_name in FINAL_CLASSES:
        class_dir = split_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg")))
            split_total += count
            percentage = (count / split_total * 100) if split_total > 0 else 0
            print(f"   {class_name:30s}: {count:4d}")
        else:
            print(f"   {class_name:30s}:    0")
    
    print(f"   {'TOPLAM':30s}: {split_total:4d}")
    total_all += split_total
    
    # Split yüzdeleri
    if total_all > 0:
        split_pct = (split_total / total_all * 100) if total_all > 0 else 0
        print(f"   Yuzde: {split_pct:.1f}%")

print(f"\n{'GENEL TOPLAM':30s}: {total_all:4d}")

# Train/Val/Test oranları
if total_all > 0:
    train_count = len(list((OUTPUT_DIR / "train").glob("*/*.jpg"))) + len(list((OUTPUT_DIR / "train").glob("*/*.jpeg")))
    val_count = len(list((OUTPUT_DIR / "val").glob("*/*.jpg"))) + len(list((OUTPUT_DIR / "val").glob("*/*.jpeg")))
    test_count = len(list((OUTPUT_DIR / "test").glob("*/*.jpg"))) + len(list((OUTPUT_DIR / "test").glob("*/*.jpeg")))
    
    print(f"\n[SPLIT ORANLARI]")
    print(f"  Train: {train_count} ({train_count/total_all*100:.1f}%)")
    print(f"  Val:   {val_count} ({val_count/total_all*100:.1f}%)")
    print(f"  Test:  {test_count} ({test_count/total_all*100:.1f}%)")

print("\n" + "=" * 80)
print("VERI SETI HAZIR!")
print("=" * 80)

