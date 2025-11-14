#!/usr/bin/env python3
"""
Valid setindeki Fracture görüntülerini ekle
Görüntü dosyası isimlerini daha iyi eşleştir
"""
from pathlib import Path
import shutil
from collections import Counter

FRACTURES_DIR = Path("datasets/bone/Bone Fractures Detection")
OUTPUT_DIR = Path("datasets/bone/Bone_9Class_Combined")
VALID_FRACTURE_DIR = OUTPUT_DIR / "val" / "Fracture"
VALID_FRACTURE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("VALID SETI FRACTURE GORUNTULERI DUZELTILIYOR")
print("=" * 80)
print()

# Valid klasörlerini kontrol et
valid_labels_dir = FRACTURES_DIR / "valid" / "labels"
valid_images_dir = FRACTURES_DIR / "valid" / "images"

print(f"[CHECK] Valid labels: {valid_labels_dir.exists()}")
print(f"[CHECK] Valid images: {valid_images_dir.exists()}")
print(f"[CHECK] Output dir: {OUTPUT_DIR.exists()}")

if not valid_labels_dir.exists() or not valid_images_dir.exists():
    print("[ERROR] Valid klasorleri bulunamadi!")
    print(f"  Labels: {valid_labels_dir}")
    print(f"  Images: {valid_images_dir}")
    exit(1)

# Tüm valid görüntülerini listele
valid_image_files = list(valid_images_dir.glob("*.jpg"))
print(f"\n[INFO] Valid images bulundu: {len(valid_image_files)}")

# Valid label dosyalarını oku
label_files = list(valid_labels_dir.glob("*.txt"))
print(f"[INFO] Valid labels bulundu: {len(label_files)}")

# Görüntü-label eşleştirmesi yap
VALID_FRACTURE_DIR.mkdir(parents=True, exist_ok=True)

fracture_count = 0
healthy_count = 0
not_found = 0

print("\n[PROCESSING] Valid goruntuler isleniyor...")

for label_file in label_files:
    label_stem = label_file.stem
    
    # Label dosyasını oku
    with open(label_file, 'r') as f:
        content = f.read().strip()
    
    # Görüntü dosyasını bul - daha esnek arama
    image_file = None
    
    # 1. Tam eşleşme
    for img_file in valid_image_files:
        if img_file.stem == label_stem or img_file.stem.startswith(label_stem) or label_stem in img_file.stem:
            image_file = img_file
            break
    
    # 2. Eğer bulunamazsa, isim formatlarını dene
    if not image_file:
        # Roboflow formatı: label_stem genelde uzun bir hash içerir
        # Image dosyası ismi farklı format olabilir
        for img_file in valid_image_files:
            # Label'daki sayısal kısmı bul ve image'de ara
            import re
            numbers_in_label = re.findall(r'\d+', label_stem)
            numbers_in_image = re.findall(r'\d+', img_file.stem)
            
            if numbers_in_label and numbers_in_image:
                if any(num in numbers_in_image for num in numbers_in_label[:2]):  # İlk 2 sayı
                    image_file = img_file
                    break
    
    if not image_file:
        not_found += 1
        if not_found <= 5:  # İlk 5 örneği göster
            print(f"  [WARNING] Goruntu bulunamadi: {label_file.name}")
        continue
    
    # Label içeriğine göre sınıf belirle
    if not content:
        # Boş label = Healthy -> Normal klasörüne
        healthy_count += 1
        normal_dir = OUTPUT_DIR / "val" / "Normal"
        normal_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_file, normal_dir / image_file.name)
    else:
        # YOLO formatı parse et
        lines = content.split('\n')
        class_ids = []
        for line in lines:
            if line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_ids.append(class_id)
        
        if class_ids:
            # Dominant sınıf
            dominant_class = Counter(class_ids).most_common(1)[0][0]
            
            if dominant_class != 2:  # 2 = Healthy
                # Fracture
                shutil.copy2(image_file, VALID_FRACTURE_DIR / image_file.name)
                fracture_count += 1
            else:
                # Healthy -> Normal
                healthy_count += 1
                normal_dir = OUTPUT_DIR / "val" / "Normal"
                normal_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_file, normal_dir / image_file.name)

print(f"\n[RESULTS]")
print(f"  Fracture: {fracture_count}")
print(f"  Healthy->Normal: {healthy_count}")
print(f"  Bulunamadi: {not_found}")

# Final kontrol
final_fracture_count = len(list(VALID_FRACTURE_DIR.glob("*.jpg")))
print(f"\n[FINAL] Valid/Fracture klasorundeki toplam goruntu: {final_fracture_count}")

print("\n" + "=" * 80)
print("DUZELTME TAMAMLANDI!")
print("=" * 80)

