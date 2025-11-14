#!/usr/bin/env python3
"""
Excel dosyasını detaylı analiz et - Label sütunlarını incele
"""
import pandas as pd
from pathlib import Path

excel_path = Path("datasets/bone/Tumor & Normal/dataset.xlsx")

print("=" * 80)
print("EXCEL DOSYASI DETAYLI ANALIZ")
print("=" * 80)
print()

df = pd.read_excel(excel_path)

# Label sütunlarını bul
label_columns = [
    'tumor', 'benign', 'malignant', 
    'osteochondroma', 'multiple osteochondromas', 'simple bone cyst',
    'giant cell tumor', 'osteofibroma', 'synovial osteochondroma',
    'other bt', 'osteosarcoma', 'other mt'
]

print("[LABEL SUTUNLARI] Hastalik kategorileri:")
print("-" * 80)

# Her label sütunu için sayıları göster
for col in label_columns:
    if col in df.columns:
        counts = df[col].value_counts()
        total = df[col].sum() if df[col].dtype in ['int64', 'float64'] else counts.get(1, 0)
        print(f"\n{col}:")
        print(f"  Toplam pozitif: {total}")
        print(f"  Dagilim:")
        for val, count in counts.items():
            print(f"    {val}: {count}")

# Kombine label analizi
print("\n" + "=" * 80)
print("[KOMBINE LABEL ANALIZI]")
print("=" * 80)

# Her satır için hangi label'lar aktif
def get_active_labels(row):
    active = []
    for col in label_columns:
        if col in df.columns:
            if pd.notna(row[col]) and (row[col] == 1 or row[col] == '1' or row[col] == True):
                active.append(col)
    return active

# Aktif label'ları hesapla
df['active_labels'] = df.apply(get_active_labels, axis=1)

# Label kombinasyonlarını say
print("\n[LABEL KOMBINASYONLARI]")
label_combos = df['active_labels'].apply(lambda x: ', '.join(sorted(x))).value_counts()
print(f"Toplam farkli kombinasyon: {len(label_combos)}")
print("\nEn sik 20 kombinasyon:")
for combo, count in label_combos.head(20).items():
    print(f"  {combo}: {count}")

# Her hastalık için ayrı ayrı sayım
print("\n" + "=" * 80)
print("[HASTALIK SINIF SAYIMLARI]")
print("=" * 80)

disease_counts = {}
for col in label_columns:
    if col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            count = int(df[col].sum())
        else:
            count = int(df[col].value_counts().get(1, 0))
        if count > 0:
            disease_counts[col] = count

print("\nPozitif ornek sayilari:")
for disease, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / len(df)) * 100
    print(f"  {disease:30s}: {count:4d} ({percentage:5.2f}%)")

# Normal olanları bul (hiçbir hastalık yok)
normal_mask = df['active_labels'].apply(lambda x: len(x) == 0)
normal_count = normal_mask.sum()
print(f"\n  {'Normal (hastalik yok)':30s}: {normal_count:4d} ({(normal_count/len(df)*100):5.2f}%)")

# Metadata analizi
print("\n" + "=" * 80)
print("[METADATA ANALIZI]")
print("=" * 80)

print(f"\nGender dagilimi:")
print(df['gender'].value_counts())

print(f"\nAge istatistikleri:")
print(df['age'].describe())

print(f"\nVucut bolgeleri:")
body_parts = ['upper limb', 'lower limb', 'pelvis']
for part in body_parts:
    if part in df.columns:
        count = int(df[part].sum()) if df[part].dtype in ['int64', 'float64'] else df[part].value_counts().get(1, 0)
        print(f"  {part}: {count}")

print(f"\nGoruntu acilari:")
angles = ['frontal', 'lateral', 'oblique']
for angle in angles:
    if angle in df.columns:
        count = int(df[angle].sum()) if df[angle].dtype in ['int64', 'float64'] else df[angle].value_counts().get(1, 0)
        print(f"  {angle}: {count}")

# Önerilen sınıf yapısı
print("\n" + "=" * 80)
print("[ONERILEN SINIF YAPISI]")
print("=" * 80)

# Basit kategoriler
print("\nBasit kategoriler (5 sinif):")
print("  1. Normal (hastalik yok)")
print("  2. Benign Tumor (benign == 1)")
print("  3. Malignant Tumor (malignant == 1)")
print("  4. Bone Cyst (simple bone cyst == 1)")
print("  5. Other (diğer durumlar)")

# Detaylı kategoriler
print("\nDetayli kategoriler:")
print("  1. Normal")
print("  2. Osteosarcoma")
print("  3. Other Malignant Tumor")
print("  4. Osteochondroma")
print("  5. Multiple Osteochondromas")
print("  6. Simple Bone Cyst")
print("  7. Giant Cell Tumor")
print("  8. Other Benign Tumor")

# Özet
print("\n" + "=" * 80)
print("OZET")
print("=" * 80)
print(f"Toplam goruntu: {len(df)}")
print(f"Toplam label kategorisi: {len([c for c in label_columns if c in df.columns])}")
print(f"Normal ornek sayisi: {normal_count}")
print(f"Hastalikli ornek sayisi: {len(df) - normal_count}")

