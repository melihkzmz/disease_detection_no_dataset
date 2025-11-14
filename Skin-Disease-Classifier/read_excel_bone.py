#!/usr/bin/env python3
"""
Tumor & Normal dataset Excel dosyasını oku ve analiz et
"""
import sys
from pathlib import Path

excel_path = Path("datasets/bone/Tumor & Normal/dataset.xlsx")

try:
    import pandas as pd
    print("[OK] pandas yuklu")
    
    print("\n" + "=" * 80)
    print("EXCEL DOSYASI ANALIZI")
    print("=" * 80)
    print()
    
    # Excel'i oku
    print(f"[READ] Excel dosyasi okunuyor: {excel_path}")
    df = pd.read_excel(excel_path)
    
    print(f"\n[SHAPE] Dosya boyutu: {df.shape}")
    print(f"   Satir: {df.shape[0]}")
    print(f"   Sutun: {df.shape[1]}")
    
    print(f"\n[COLUMNS] Sutunlar:")
    for i, col in enumerate(df.columns):
        print(f"   {i+1}. {col}")
    
    print(f"\n[HEAD] Ilk 10 satir:")
    print(df.head(10))
    
    # Label sütunu varsa
    label_cols = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower() or 'disease' in col.lower() or 'category' in col.lower()]
    if label_cols:
        print(f"\n[LABELS] Label sutunu bulundu: {label_cols}")
        for label_col in label_cols:
            print(f"\n   {label_col} dagilimi:")
            print(df[label_col].value_counts())
    else:
        # Son sütunu dene
        print(f"\n[LABELS] Son sutun analizi:")
        last_col = df.columns[-1]
        print(f"   Son sutun: {last_col}")
        print(df[last_col].value_counts())
    
    # Image name sütunu varsa
    image_cols = [col for col in df.columns if 'image' in col.lower() or 'file' in col.lower() or 'name' in col.lower()]
    if image_cols:
        print(f"\n[IMAGES] Image sutunu: {image_cols}")
    
    print("\n" + "=" * 80)
    print("OZET")
    print("=" * 80)
    print(f"Toplam satir: {len(df)}")
    print(f"Sutun sayisi: {len(df.columns)}")
    
except ImportError:
    print("[ERROR] pandas yuklu degil!")
    print("[INFO] Yuklemek icin: pip install pandas openpyxl")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Hata: {e}")
    sys.exit(1)

