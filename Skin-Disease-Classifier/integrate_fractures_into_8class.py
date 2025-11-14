#!/usr/bin/env python3
"""
Bone Fractures dataset'ini mevcut 8 sınıflı veri setine entegre et
- Tüm kırık tiplerini 'Fracture' kategorisine birleştir
- 'Healthy' ile 'Normal'i birleştir
- Final: 9 sınıf
"""
import pandas as pd
import shutil
from pathlib import Path
from collections import Counter
import random

# Yollar
BASE_DIR = Path("datasets/bone")
FRACTURES_DIR = BASE_DIR / "Bone Fractures Detection"
EXISTING_8CLASS_DIR = BASE_DIR / "Bone_8Class"

OUTPUT_DIR = BASE_DIR / "Bone_9Class_Combined"
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

# 9 Sınıf (8 mevcut + Fracture, Normal ve Healthy birleşik)
FINAL_CLASSES = [
    'Normal',                    # Normal + Healthy birleşik
    'Osteochondroma',
    'Osteosarcoma',
    'Multiple_Osteochondromas',
    'Other_Benign',
    'Simple_Bone_Cyst',
    'Giant_Cell_Tumor',
    'Other_Malignant',
    'Fracture'                   # Tüm kırık tipleri
]

# Kırık sınıf isimleri (data.yaml'dan)
FRACTURE_CLASSES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental',
    'Spiral', 'Transverse Displaced', 'Transverse'
]

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)

print("=" * 80)
print("BONE FRACTURES ENTEGRASYONU - 9 SINIFLI VERI SETI")
print("=" * 80)
print()

# Mevcut 8 sınıflı veri setinden Normal sınıfını al
print("[1/4] Mevcut 8 sinifli veri setinden veriler kopyalaniyor...")
existing_normal_count = 0
for split_dir in [EXISTING_8CLASS_DIR / "train", EXISTING_8CLASS_DIR / "val", EXISTING_8CLASS_DIR / "test"]:
    if not split_dir.exists():
        continue
    
    split_name = split_dir.name
    normal_source = split_dir / "Normal"
    
    if normal_source.exists():
        normal_files = list(normal_source.glob("*.jpeg")) + list(normal_source.glob("*.jpg"))
        existing_normal_count += len(normal_files)

print(f"   Mevcut Normal goruntuler: {existing_normal_count}")

# Diğer 7 sınıfı da kopyala (Normal hariç)
other_classes = [c for c in FINAL_CLASSES if c not in ['Normal', 'Fracture']]
for class_name in other_classes:
    for split_dir in [EXISTING_8CLASS_DIR / "train", EXISTING_8CLASS_DIR / "val", EXISTING_8CLASS_DIR / "test"]:
        if not split_dir.exists():
            continue
        
        source_class_dir = split_dir / class_name
        if source_class_dir.exists():
            target_split_dir = OUTPUT_DIR / split_dir.name / class_name
            target_split_dir.mkdir(parents=True, exist_ok=True)
            
            for img_file in list(source_class_dir.glob("*.jpeg")) + list(source_class_dir.glob("*.jpg")):
                shutil.copy2(img_file, target_split_dir / img_file.name)

print(f"   Diger 7 sinif kopyalandi")

# Bone Fractures dataset'ini parse et
print("\n[2/4] Bone Fractures dataset'i parse ediliyor...")

fracture_stats = {split: Counter() for split in ['train', 'valid', 'test']}

for split_name in ['train', 'valid', 'test']:
    labels_dir = FRACTURES_DIR / split_name / "labels"
    images_dir = FRACTURES_DIR / split_name / "images"
    
    if not labels_dir.exists():
        continue
    
    label_files = list(labels_dir.glob("*.txt"))
    print(f"   {split_name}: {len(label_files)} label dosyasi")
    
    fracture_images = []
    healthy_images = []
    
    for label_file in label_files:
        # İlgili image dosyasını bul
        label_stem = label_file.stem
        image_file = None
        
        # Önce tam eşleşme dene
        for img_file in list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")):
            if img_file.stem == label_stem or img_file.stem.startswith(label_stem) or label_stem in img_file.stem:
                image_file = img_file
                break
        
        # Eğer bulunamazsa, sayısal kısımları kullan
        if not image_file:
            import re
            label_numbers = re.findall(r'\d+', label_stem)
            for img_file in list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")):
                img_numbers = re.findall(r'\d+', img_file.stem)
                if label_numbers and img_numbers:
                    # İlk birkaç sayıyı karşılaştır
                    if any(num in img_numbers[:3] for num in label_numbers[:3] if len(num) >= 2):
                        image_file = img_file
                        break
        
        # Son çare: farklı uzantı dene
        if not image_file:
            for ext in ['.jpg', '.jpeg']:
                possible_images = list(images_dir.glob(f"{label_stem}*{ext}"))
                if possible_images:
                    image_file = possible_images[0]
                    break
        
        if not image_file:
            continue
        
        # Label dosyasını oku
        with open(label_file, 'r') as f:
            content = f.read().strip()
        
        if not content:
            # Boş label = Healthy
            healthy_images.append(image_file)
        else:
            # YOLO formatı: class_id x y w h
            lines = content.split('\n')
            class_ids = []
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_ids.append(class_id)
            
            if class_ids:
                # Dominant sınıfı bul (en çok görünen)
                dominant_class = Counter(class_ids).most_common(1)[0][0]
                
                # Healthy (class_id=2) değilse Fracture
                if dominant_class != 2:  # 2 = Healthy
                    fracture_images.append((image_file, dominant_class))
                else:
                    healthy_images.append(image_file)
    
    fracture_stats[split_name]['fracture'] = len(fracture_images)
    fracture_stats[split_name]['healthy'] = len(healthy_images)
    
    print(f"      Fracture: {len(fracture_images)}")
    print(f"      Healthy: {len(healthy_images)}")
    
    # Fracture görüntülerini kaydet
    if fracture_images:
        fracture_dir = OUTPUT_DIR / split_name / "Fracture"
        fracture_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file, _ in fracture_images:
            shutil.copy2(img_file, fracture_dir / img_file.name)
    
    # Healthy görüntülerini Normal klasörüne ekle
    if healthy_images:
        normal_dir = OUTPUT_DIR / split_name / "Normal"
        normal_dir.mkdir(parents=True, exist_ok=True)
        
        for img_file in healthy_images:
            shutil.copy2(img_file, normal_dir / img_file.name)

# Mevcut Normal görüntüleri de kopyala
print("\n[3/4] Mevcut Normal goruntuler kopyalaniyor...")
for split_name in ['train', 'val', 'test']:
    source_normal = EXISTING_8CLASS_DIR / split_name / "Normal"
    target_normal = OUTPUT_DIR / split_name / "Normal"
    
    if source_normal.exists():
        target_normal.mkdir(parents=True, exist_ok=True)
        for img_file in list(source_normal.glob("*.jpeg")) + list(source_normal.glob("*.jpg")):
            shutil.copy2(img_file, target_normal / img_file.name)

# Final istatistikler
print("\n[4/4] Final istatistikler hesaplaniyor...")

print("\n" + "=" * 80)
print("FINAL ISTATISTIKLER - 9 SINIFLI VERI SETI")
print("=" * 80)

total_all = 0
for split_name in ['train', 'val', 'test']:
    split_dir = OUTPUT_DIR / split_name
    print(f"\n{split_name.upper()}:")
    
    split_total = 0
    for class_name in FINAL_CLASSES:
        class_dir = split_dir / class_name
        if class_dir.exists():
            count = len(list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg")))
            split_total += count
            print(f"   {class_name:30s}: {count:4d}")
        else:
            print(f"   {class_name:30s}:    0")
    
    print(f"   {'TOPLAM':30s}: {split_total:4d}")
    total_all += split_total

print(f"\n{'GENEL TOPLAM':30s}: {total_all:4d}")

# Class mapping dosyasını güncelle
class_mapping_file = OUTPUT_DIR / "class_mapping.txt"
with open(class_mapping_file, 'w', encoding='utf-8') as f:
    f.write("9 SINIFLI KEMIK HASTALIGI VERI SETI\n")
    f.write("=" * 80 + "\n\n")
    f.write("Bu veri seti iki kaynaktan birlesmistir:\n")
    f.write("1. Tumor & Normal Dataset (8 sinif)\n")
    f.write("2. Bone Fractures Detection Dataset (kırık tipleri)\n\n")
    f.write("SINIFLAR:\n")
    for i, class_name in enumerate(FINAL_CLASSES, 1):
        f.write(f"{i}. {class_name}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("\nNOTLAR:\n")
    f.write("- Normal: Tumor dataset'inden Normal + Fracture dataset'inden Healthy\n")
    f.write("- Fracture: Tüm kırık tipleri birleştirilmiş (Comminuted, Greenstick, Linear, etc.)\n")
    f.write("- Diğer 7 sınıf: Tumor & Normal dataset'inden\n")

print(f"\n[INFO] Class mapping kaydedildi: {class_mapping_file}")

print("\n" + "=" * 80)
print("ENTEGRASYON TAMAMLANDI!")
print("=" * 80)
print(f"\nOutput dizini: {OUTPUT_DIR}")
print("9 sinifli veri seti egitim icin hazir!")

