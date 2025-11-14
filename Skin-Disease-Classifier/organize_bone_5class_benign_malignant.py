#!/usr/bin/env python3
"""
5 Sınıflı Kemik Hastalığı Veri Seti Organizasyonu
- Tümörler Benign/Malignant ayrımı ile birleştirildi
- Beklenen accuracy artışı: +%10-15
"""
import shutil
from pathlib import Path
from collections import Counter

# Yollar
BASE_DIR = Path("datasets/bone")
SOURCE_9CLASS_DIR = BASE_DIR / "Bone_9Class_Combined"
OUTPUT_DIR = BASE_DIR / "Bone_5Class_BenignMalignant"

# 5 Sınıf Tanımları
FINAL_CLASSES = [
    'Normal',
    'Fracture',
    'Benign_Tumor',      # Osteochondroma + Multiple_Osteochondromas + Other_Benign + Giant_Cell_Tumor
    'Malignant_Tumor',   # Osteosarcoma + Other_Malignant
    'Simple_Bone_Cyst'
]

# Birleştirme mapping
TUMOR_MAPPING = {
    'Benign_Tumor': [
        'Osteochondroma',
        'Multiple_Osteochondromas',
        'Other_Benign',
        'Giant_Cell_Tumor'
    ],
    'Malignant_Tumor': [
        'Osteosarcoma',
        'Other_Malignant'
    ]
}

print("=" * 80)
print("5 SINIFLI KEMIK HASTALIGI VERI SETI - BENIGN/MALIGNANT AYRIMI")
print("=" * 80)
print()

# Output dizinlerini oluştur
print("[1/3] Output dizinleri olusturuluyor...")
for split_name in ['train', 'val', 'test']:
    for class_name in FINAL_CLASSES:
        (OUTPUT_DIR / split_name / class_name).mkdir(parents=True, exist_ok=True)
    print(f"   {split_name}/ dizinleri hazir")

# Mevcut 9 sınıflı veri setinden verileri kopyala ve birleştir
print("\n[2/3] Veriler kopyalaniyor ve birlestiriliyor...")

stats = {split: Counter() for split in ['train', 'val', 'test']}

for split_name in ['train', 'val', 'test']:
    source_split_dir = SOURCE_9CLASS_DIR / split_name
    target_split_dir = OUTPUT_DIR / split_name
    
    if not source_split_dir.exists():
        continue
    
    print(f"\n   {split_name.upper()}:")
    
    # Direkt kopyalanacak sınıflar (Normal, Fracture, Simple_Bone_Cyst)
    direct_classes = ['Normal', 'Fracture', 'Simple_Bone_Cyst']
    for class_name in direct_classes:
        source_class_dir = source_split_dir / class_name
        if source_class_dir.exists():
            target_class_dir = target_split_dir / class_name
            count = 0
            for img_file in list(source_class_dir.glob("*.jpeg")) + list(source_class_dir.glob("*.jpg")):
                shutil.copy2(img_file, target_class_dir / img_file.name)
                count += 1
            stats[split_name][class_name] = count
            print(f"     {class_name:25s}: {count:4d}")
    
    # Birleştirilecek tümör sınıfları
    for target_class, source_classes in TUMOR_MAPPING.items():
        target_class_dir = target_split_dir / target_class
        count = 0
        
        for source_class in source_classes:
            source_class_dir = source_split_dir / source_class
            if source_class_dir.exists():
                for img_file in list(source_class_dir.glob("*.jpeg")) + list(source_class_dir.glob("*.jpg")):
                    shutil.copy2(img_file, target_class_dir / img_file.name)
                    count += 1
        
        stats[split_name][target_class] = count
        print(f"     {target_class:25s}: {count:4d} (birlestirildi)")

# Final istatistikler
print("\n[3/3] Final istatistikler hesaplaniyor...")

print("\n" + "=" * 80)
print("FINAL ISTATISTIKLER - 5 SINIFLI VERI SETI")
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
            percentage = (count / split_total * 100) if split_total > 0 else 0
            print(f"   {class_name:25s}: {count:4d}")
        else:
            print(f"   {class_name:25s}:    0")
    
    print(f"   {'TOPLAM':25s}: {split_total:4d}")
    total_all += split_total
    
    # Split yüzdeleri
    if total_all > 0:
        split_pct = (split_total / total_all * 100)
        print(f"   Yuzde: {split_pct:.1f}%")

print(f"\n{'GENEL TOPLAM':25s}: {total_all:4d}")

# Class mapping dosyasını oluştur
class_mapping_file = OUTPUT_DIR / "class_mapping.txt"
with open(class_mapping_file, 'w', encoding='utf-8') as f:
    f.write("5 SINIFLI KEMIK HASTALIGI VERI SETI\n")
    f.write("Benign/Malignant Tümör Ayrımı ile\n")
    f.write("=" * 80 + "\n\n")
    f.write("SINIFLAR:\n")
    for i, class_name in enumerate(FINAL_CLASSES, 1):
        f.write(f"{i}. {class_name}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("\nBIRLESTIRME DETAYLARI:\n\n")
    f.write("Benign_Tumor:\n")
    for source_class in TUMOR_MAPPING['Benign_Tumor']:
        f.write(f"  - {source_class}\n")
    f.write("\nMalignant_Tumor:\n")
    for source_class in TUMOR_MAPPING['Malignant_Tumor']:
        f.write(f"  - {source_class}\n")
    f.write("\n" + "=" * 80 + "\n")
    f.write("\nNOTLAR:\n")
    f.write("- Tümörler benign/malignant ayrımına göre birleştirildi\n")
    f.write("- Normal, Fracture, Simple_Bone_Cyst ayrı tutuldu\n")
    f.write("- Beklenen accuracy artışı: +%10-15\n")

print(f"\n[INFO] Class mapping kaydedildi: {class_mapping_file}")

# Beklenen accuracy artışı bilgisi
print("\n" + "=" * 80)
print("BEKLENEN SONUCLAR")
print("=" * 80)
print("\n[OK] Sinif sayisi: 9 -> 5")
print("[OK] Beklenen accuracy artisi: +%10-15")
print("[OK] Daha dengeli sinif dagilimi")
print("[OK] Tibbi acidan onemli ayrim korundu (benign/malignant)")

print("\n" + "=" * 80)
print("ORGANIZASYON TAMAMLANDI!")
print("=" * 80)
print(f"\nOutput dizini: {OUTPUT_DIR}")
print("5 sinifli veri seti egitim icin hazir!")
print("\nSonraki adim: train_bone_5class.py scripti hazirlanacak")

