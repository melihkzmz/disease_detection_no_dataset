#!/usr/bin/env python3
"""
Kemik veri setlerini analiz et ve yapılarını göster
"""
import os
import json
from pathlib import Path
from collections import Counter

# Veri seti yolları
BONE_FRACTURES_DIR = "datasets/bone/Bone Fractures Detection"
TUMOR_NORMAL_DIR = "datasets/bone/Tumor & Normal"

print("=" * 80)
print("KEMIK VERI SETLERI ANALİZİ")
print("=" * 80)
print()

# ======================================================================
# 1. BONE FRACTURES DETECTION
# ======================================================================
print("1. BONE FRACTURES DETECTION")
print("-" * 80)

fractures_dir = Path(BONE_FRACTURES_DIR)
if fractures_dir.exists():
    # data.yaml dosyasını oku
    yaml_file = fractures_dir / "data.yaml"
    if yaml_file.exists():
        print("[FILE] data.yaml icerigi:")
        with open(yaml_file, 'r', encoding='utf-8') as f:
            print(f.read())
        print()
    
    # Train labels analizi
    train_labels_dir = fractures_dir / "train" / "labels"
    if train_labels_dir.exists():
        label_files = list(train_labels_dir.glob("*.txt"))
        print(f"[DIR] Train labels: {len(label_files)} dosya")
        
        # Label sınıflarını say
        class_counts = Counter()
        sample_label = label_files[0] if label_files else None
        
        if sample_label:
            print(f"\n[SAMPLE] Ornek label dosyasi: {sample_label.name}")
            with open(sample_label, 'r') as f:
                content = f.read().strip()
                print(f"   İçerik: {content[:100]}...")
                if content:
                    # YOLO formatı: class_id x_center y_center width height
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip():
                            parts = line.split()
                            if len(parts) > 0:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1
            
            # YAML'den sınıf isimlerini al
            if yaml_file.exists():
                with open(yaml_file, 'r') as f:
                    yaml_content = f.read()
                    if 'names:' in yaml_content:
                        import re
                        names_match = re.search(r"names:\s*\[(.*?)\]", yaml_content, re.DOTALL)
                        if names_match:
                            names_str = names_match.group(1)
                            names = [n.strip().strip("'\"") for n in names_str.split(',')]
                            print(f"\n[CLASSES] Siniflar ({len(names)} sinif):")
                            for i, name in enumerate(names):
                                count = class_counts.get(i, 0)
                                print(f"   {i}: {name} ({count} ornek)")
        
        print(f"\n[STATS] Veri dagilimi:")
        print(f"   Train images: {len(list((fractures_dir / 'train' / 'images').glob('*.jpg')))}")
        print(f"   Valid images: {len(list((fractures_dir / 'valid' / 'images').glob('*.jpg')))}")
        print(f"   Test images: {len(list((fractures_dir / 'test' / 'images').glob('*.jpg')))}")
        
        print()
    
    # README dosyası
    readme_file = fractures_dir / "README.roboflow.txt"
    if readme_file.exists():
        print("[FILE] README.roboflow.txt ozeti:")
        with open(readme_file, 'r') as f:
            lines = f.readlines()
            print(f"   {' '.join(lines[:5])}")
        print()

# ======================================================================
# 2. TUMOR & NORMAL
# ======================================================================
print("\n2. TUMOR & NORMAL")
print("-" * 80)

tumor_dir = Path(TUMOR_NORMAL_DIR)
if tumor_dir.exists():
    # Excel dosyası kontrolü
    excel_file = tumor_dir / "dataset.xlsx"
    if excel_file.exists():
        print(f"[FILE] Excel dosyasi bulundu: {excel_file.name}")
        print(f"   Boyut: {excel_file.stat().st_size / 1024:.2f} KB")
        print("   [INFO] Excel dosyasini okumak icin pandas/openpyxl gerekli")
        print("   [TIPS] Excel dosyasini manuel kontrol et veya pandas ile oku")
        print()
    
    # Images klasörü
    images_dir = tumor_dir / "images"
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.jpg"))
        print(f"[DIR] Images: {len(image_files)} goruntu")
        print(f"   Formatlar: JPEG, JPG")
        
        # İlk birkaç görüntü ismini göster
        if image_files:
            print(f"\n[SAMPLE] Ornek goruntu isimleri:")
            for img in image_files[:5]:
                print(f"   - {img.name}")
            if len(image_files) > 5:
                print(f"   ... ve {len(image_files) - 5} tane daha")
        print()
    
    # Annotations klasörü
    annotations_dir = tumor_dir / "Annotations"
    if annotations_dir.exists():
        json_files = list(annotations_dir.glob("*.json"))
        print(f"[DIR] Annotations: {len(json_files)} JSON dosyasi")
        
        # İlk birkaç JSON'u analiz et
        if json_files:
            print(f"\n[ANALYSIS] Ornek annotation analizi:")
            sample_json = json_files[0]
            print(f"   Dosya: {sample_json.name}")
            
            try:
                with open(sample_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"   JSON yapısı:")
                print(f"     - Keys: {list(data.keys())}")
                
                # Shapes bilgisi varsa
                if 'shapes' in data:
                    shapes = data['shapes']
                    print(f"     - Shapes sayısı: {len(shapes)}")
                    if shapes:
                        first_shape = shapes[0]
                        print(f"     - İlk shape keys: {list(first_shape.keys())}")
                        if 'label' in first_shape:
                            print(f"     - Label: {first_shape['label']}")
                
                # Version bilgisi
                if 'version' in data:
                    print(f"     - Version: {data['version']}")
                
            except Exception as e:
                print(f"   ⚠️  JSON okuma hatası: {e}")
            
            # Tüm annotation'lardaki label'ları say
            print(f"\n[LABELS] Label analizi (ilk 100 dosya):")
            all_labels = Counter()
            for json_file in json_files[:100]:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if 'shapes' in data:
                            for shape in data['shapes']:
                                if 'label' in shape:
                                    all_labels[shape['label']] += 1
                except Exception as e:
                    pass
            
            if all_labels:
                print("   Bulunan label'lar:")
                for label, count in all_labels.most_common():
                    print(f"     - {label}: {count}")
            
            print()

# ======================================================================
# ÖZET ve ÖNERİLER
# ======================================================================
print("\n" + "=" * 80)
print("OZET VE ONERILER")
print("=" * 80)
print()
print("[DATASETS] Bulunan veri setleri:")
print("   1. Bone Fractures Detection:")
print("      - 10 sinifli kirik tespiti")
print("      - YOLO formatinda (object detection)")
print("      - Train/Valid/Test split mevcut")
print("      - data.yaml dosyasi ile sinif bilgileri")
print()
print("   2. Tumor & Normal:")
print("      - Excel dosyasi: dataset.xlsx (detayli analiz gerekli)")
print("      - JSON annotations: LabelMe formati")
print("      - Goruntuler: JPEG format")
print()
print("[NEXT] Sonraki adimlar:")
print("   1. Excel dosyasini oku (pandas ile)")
print("   2. JSON annotations'lari parse et")
print("   3. Veri setlerini birlestir")
print("   4. Organizasyon scripti yaz")
print()

