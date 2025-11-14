#!/usr/bin/env python3
"""
8 Sınıflı Kemik Hastalığı Veri Seti Organizasyonu
Tumor & Normal dataset'ini parse edip organize eder
"""
import pandas as pd
import json
import shutil
from pathlib import Path
from collections import Counter
import random

# Yollar
BASE_DIR = Path("datasets/bone")
SOURCE_DIR = BASE_DIR / "Tumor & Normal"
EXCEL_PATH = SOURCE_DIR / "dataset.xlsx"
ANNOTATIONS_DIR = SOURCE_DIR / "Annotations"
IMAGES_DIR = SOURCE_DIR / "images"

OUTPUT_DIR = BASE_DIR / "Bone_8Class"
TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

# 8 Sınıf Tanımları
CLASS_MAPPING = {
    'Normal': {
        'conditions': ['normal'],  # Hiçbir label aktif değilse
    },
    'Osteosarcoma': {
        'conditions': ['osteosarcoma'],
    },
    'Other_Malignant': {
        'conditions': ['other mt'],
    },
    'Osteochondroma': {
        'conditions': ['osteochondroma'],
        'exclude': ['multiple osteochondromas']  # Multiple olmayan
    },
    'Multiple_Osteochondromas': {
        'conditions': ['multiple osteochondromas'],
    },
    'Simple_Bone_Cyst': {
        'conditions': ['simple bone cyst'],
    },
    'Giant_Cell_Tumor': {
        'conditions': ['giant cell tumor'],
    },
    'Other_Benign': {
        'conditions': ['other bt', 'osteofibroma', 'synovial osteochondroma'],
    }
}

# Split oranları
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

print("=" * 80)
print("8 SINIFLI KEMIK HASTALIGI VERI SETI ORGANIZASYONU")
print("=" * 80)
print()

# Excel dosyasını oku
print("[1/5] Excel dosyasi okunuyor...")
df = pd.read_excel(EXCEL_PATH)
print(f"   Toplam satir: {len(df)}")

# Her satır için sınıf belirle
def get_class_for_row(row):
    """Bir satır için sınıf belirle"""
    # Aktif label'ları bul
    active_labels = []
    for col in df.columns:
        if col in ['tumor', 'benign', 'malignant', 'osteochondroma', 
                   'multiple osteochondromas', 'simple bone cyst',
                   'giant cell tumor', 'osteofibroma', 'synovial osteochondroma',
                   'other bt', 'osteosarcoma', 'other mt']:
            if pd.notna(row[col]) and (row[col] == 1 or row[col] == '1' or row[col] == True):
                active_labels.append(col.lower())
    
    # Normal kontrolü
    if len(active_labels) == 0:
        return 'Normal'
    
    # Her sınıf için kontrol et (öncelik sırasına göre)
    for class_name, class_info in CLASS_MAPPING.items():
        if class_name == 'Normal':
            continue
        
        conditions = class_info.get('conditions', [])
        exclude = class_info.get('exclude', [])
        
        # Conditions kontrolü
        if any(label in active_labels for label in conditions):
            # Exclude kontrolü
            if not any(excl in active_labels for excl in exclude):
                return class_name
    
    # Eğer hiçbir sınıfa uymazsa (olması gerekmez ama güvenlik için)
    return 'Normal'

print("[2/5] Siniflar belirleniyor...")
df['class'] = df.apply(get_class_for_row, axis=1)

# Sınıf dağılımını göster
class_counts = df['class'].value_counts()
print("\n   Sinif dagilimi:")
for cls, count in class_counts.items():
    print(f"     {cls:30s}: {count:4d}")

# Output dizinlerini oluştur
print("\n[3/5] Output dizinleri olusturuluyor...")
for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    for class_name in CLASS_MAPPING.keys():
        (split_dir / class_name).mkdir(parents=True, exist_ok=True)
    print(f"   {split_dir.name}/ dizinleri hazir")

# Train/Val/Test split yap
print("\n[4/5] Train/Val/Test split yapiliyor...")
random.seed(42)  # Reproducibility için

for class_name in CLASS_MAPPING.keys():
    class_df = df[df['class'] == class_name].copy()
    image_ids = class_df['image_id'].tolist()
    
    # Shuffle
    random.shuffle(image_ids)
    
    # Split
    n_total = len(image_ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:n_train + n_val]
    test_ids = image_ids[n_train + n_val:]
    
    print(f"\n   {class_name}:")
    print(f"     Train: {len(train_ids)}")
    print(f"     Val:   {len(val_ids)}")
    print(f"     Test:  {len(test_ids)}")
    
    # Dosyaları kopyala
    splits = [
        (train_ids, TRAIN_DIR / class_name),
        (val_ids, VAL_DIR / class_name),
        (test_ids, TEST_DIR / class_name),
    ]
    
    for img_ids, dest_dir in splits:
        copied = 0
        not_found = 0
        
        for img_id in img_ids:
            # Image dosyasını bul
            src_image = IMAGES_DIR / img_id
            if not src_image.exists():
                # .jpeg ve .jpg denemesi
                if img_id.endswith('.jpeg'):
                    src_image = IMAGES_DIR / img_id.replace('.jpeg', '.jpg')
                elif img_id.endswith('.jpg'):
                    src_image = IMAGES_DIR / img_id.replace('.jpg', '.jpeg')
            
            if src_image.exists():
                dst_image = dest_dir / img_id
                shutil.copy2(src_image, dst_image)
                copied += 1
            else:
                not_found += 1
        
        if not_found > 0:
            print(f"     [WARNING] {dest_dir.name}: {not_found} dosya bulunamadi")

print("\n[5/5] Final kontrol...")

# Final istatistikler
print("\n" + "=" * 80)
print("FINAL ISTATISTIKLER")
print("=" * 80)

total_copied = 0
for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    print(f"\n{split_dir.name.upper()}:")
    split_total = 0
    for class_name in CLASS_MAPPING.keys():
        class_dir = split_dir / class_name
        count = len(list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg")))
        split_total += count
        print(f"   {class_name:30s}: {count:4d}")
    print(f"   {'TOPLAM':30s}: {split_total:4d}")
    total_copied += split_total

print(f"\n{'GENEL TOPLAM':30s}: {total_copied:4d}")

# Class mapping dosyasını kaydet
class_mapping_file = OUTPUT_DIR / "class_mapping.txt"
with open(class_mapping_file, 'w', encoding='utf-8') as f:
    f.write("8 SINIFLI KEMIK HASTALIGI SINIFLARI\n")
    f.write("=" * 80 + "\n\n")
    for i, class_name in enumerate(CLASS_MAPPING.keys(), 1):
        f.write(f"{i}. {class_name}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("\nExcel sutunlari ile eslesme:\n")
    for class_name, class_info in CLASS_MAPPING.items():
        f.write(f"\n{class_name}:\n")
        f.write(f"  Conditions: {class_info.get('conditions', [])}\n")
        if 'exclude' in class_info:
            f.write(f"  Exclude: {class_info.get('exclude', [])}\n")

print(f"\n[INFO] Class mapping kaydedildi: {class_mapping_file}")

print("\n" + "=" * 80)
print("ORGANIZASYON TAMAMLANDI!")
print("=" * 80)
print(f"\nOutput dizini: {OUTPUT_DIR}")
print("Egitim icin hazir: train/, val/, test/ klasorleri olusturuldu")

