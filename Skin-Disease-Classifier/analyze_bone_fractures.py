#!/usr/bin/env python3
"""
Bone Fractures Detection dataset'ini analiz et
YOLO formatından classification'a çevirme stratejisi belirle
"""
from pathlib import Path
from collections import Counter

FRACTURES_DIR = Path("datasets/bone/Bone Fractures Detection")

print("=" * 80)
print("BONE FRACTURES DETECTION DATASET ANALIZI")
print("=" * 80)
print()

# Sınıf isimlerini oku (data.yaml'dan)
yaml_file = FRACTURES_DIR / "data.yaml"
class_names = []
with open(yaml_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'names:' in line:
            # names: ['Comminuted', 'Greenstick', ...] formatını parse et
            import re
            names_match = re.search(r"names:\s*\[(.*?)\]", ''.join(lines), re.DOTALL)
            if names_match:
                names_str = names_match.group(1)
                class_names = [n.strip().strip("'\"") for n in names_str.split(',')]
            break

print(f"[CLASSES] {len(class_names)} sinif bulundu:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# Her split için label analizi
print("\n" + "=" * 80)
print("LABEL ANALIZI")
print("=" * 80)

for split in ['train', 'valid', 'test']:
    labels_dir = FRACTURES_DIR / split / "labels"
    if not labels_dir.exists():
        continue
    
    print(f"\n[{split.upper()}]")
    
    label_files = list(labels_dir.glob("*.txt"))
    print(f"  Toplam label dosyasi: {len(label_files)}")
    
    # Her label dosyasındaki sınıfları say
    class_counts = Counter()
    multi_class_images = 0
    empty_images = 0
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            content = f.read().strip()
            
        if not content:
            empty_images += 1
            continue
        
        lines = content.split('\n')
        classes_in_image = set()
        
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    classes_in_image.add(class_id)
                    class_counts[class_id] += 1
        
        if len(classes_in_image) > 1:
            multi_class_images += 1
    
    print(f"  Bos label dosyasi: {empty_images}")
    print(f"  Coklu sinif iceren goruntu: {multi_class_images}")
    print(f"  Sinif dagilimi:")
    for class_id, count in sorted(class_counts.items()):
        if class_id < len(class_names):
            print(f"    {class_id}: {class_names[class_id]:25s} - {count:4d} obje")

# Örnek label dosyaları incele
print("\n" + "=" * 80)
print("ORNEK LABEL DOSYALARI")
print("=" * 80)

train_labels = list((FRACTURES_DIR / "train" / "labels").glob("*.txt"))[:5]
for label_file in train_labels:
    print(f"\n  Dosya: {label_file.name}")
    with open(label_file, 'r') as f:
        content = f.read().strip()
        if content:
            lines = content.split('\n')
            print(f"    {len(lines)} obje bulundu")
            for i, line in enumerate(lines[:3], 1):
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown({class_id})"
                    print(f"      Obje {i}: {class_name}")
        else:
            print("    Bos (Normal/Healthy)")

# Öneriler
print("\n" + "=" * 80)
print("ONERILER")
print("=" * 80)

print("\n[STRATEJI 1] YOLO'dan Classification'a Cevirme:")
print("  - Her goruntu icin dominant sinifi bul")
print("  - Eger coklu sinif varsa, en sik gorulen sinifi sec")
print("  - Eger label dosyasi bos ise -> 'Healthy' sinifi")
print()
print("  Avantaj: Basit, hizli")
print("  Dezavantaj: Tek bir goruntude birden fazla kirik varsa bilgi kaybi")

print("\n[STRATEJI 2] Kategorize Etme:")
print("  - Tum kirik tiplerini 'Fracture' kategorisine birleştir")
print("  - 'Healthy' ayrı tut")
print("  - Toplam: 2 sinif (Fracture, Healthy)")
print()
print("  Avantaj: Basit, dengeli")
print("  Dezavantaj: Kirik tipleri arasinda ayrim yapamaz")

print("\n[STRATEJI 3] Ana Kategoriler:")
print("  - Comminuted, Linear, Oblique Displaced, etc. -> 'Complex_Fracture'")
print("  - Greenstick, Linear, Oblique, etc. -> 'Simple_Fracture'")
print("  - Healthy -> 'Healthy'")
print()
print("  Avantaj: Orta seviye detay")
print("  Dezavantaj: Kategorizasyon subjektif")

print("\n[STRATEJI 4] Tum Siniflari Ayri Tut:")
print("  - Her kirik tipini ayri sinif olarak tut")
print("  - 10 sinif (9 kirik tipi + Healthy)")
print("  - Mevcut 8 sinif ile birleştir -> 18 sinif")
print()
print("  Avantaj: En detayli")
print("  Dezavantaj: Cok fazla sinif, bazilari cok kucuk olabilir")

print("\n" + "=" * 80)
print("ONERILEN YAKLASIM")
print("=" * 80)
print("\n[ONERI] Strateji 1 veya 2:")
print("  1. Her goruntu icin dominant sinifi belirle")
print("  2. 10 kirik sinifini 'Fracture' kategorisine birleştir")
print("  3. Mevcut 8 sinif + Fracture + Healthy = 10 sinif (veya Healthy zaten var mi kontrol et)")
print()
print("  Final siniflar:")
print("    1. Normal (Tumor dataset'inden)")
print("    2-8. Mevcut 8 sinif")
print("    9. Fracture (tum kirik tipleri)")
print("    10. Healthy (Fracture dataset'inden, eger Normal'den farkli ise)")
print()
print("  VEYA:")
print("    Normal ve Healthy'yi birleştir -> 9 sinif")

