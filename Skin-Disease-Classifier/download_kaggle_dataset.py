# -*- coding: utf-8 -*-
"""
Kaggle HAM10000 Dataset Indirme ve Hazırlama
"""
import os
import sys
import subprocess
import zipfile
from pathlib import Path
import shutil

print("="*70)
print(" KAGGLE DATASET INDIRME ASISTANI")
print("="*70)

# 1. Kaggle API kontrolu
print("\n1. Kaggle API kontrol ediliyor...")

try:
    import kaggle
    print("   OK - Kaggle API yuklu")
except ImportError:
    print("   Kaggle API yuklu degil, yukleniyor...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    import kaggle
    print("   OK - Kaggle API yuklendi")

# 2. Kaggle credentials kontrolu
kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'

if not kaggle_json.exists():
    print("\n" + "="*70)
    print(" KAGGLE API KEY GEREKLI")
    print("="*70)
    print("\nAdimlar:")
    print("  1. https://www.kaggle.com/account adresine gidin")
    print("  2. 'Create New API Token' butonuna tiklayin")
    print("  3. kaggle.json dosyasi indirilecek")
    print("  4. Bu dosyayi su konuma kopyalayin:")
    print(f"     {kaggle_json.parent}")
    print("\n  VEYA:")
    print("  5. Dosyayi su klasore kopyalayin:")
    print(f"     {Path.cwd()}")
    
    print("\nKaggle.json dosyasinin yolu: ", end="")
    user_path = input().strip()
    
    if user_path and Path(user_path).exists():
        kaggle_json.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(user_path, kaggle_json)
        os.chmod(kaggle_json, 0o600)
        print("   OK - API key kopyalandi")
    else:
        # Mevcut dizinde kaggle.json var mi?
        local_kaggle = Path('kaggle.json')
        if local_kaggle.exists():
            kaggle_json.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(local_kaggle, kaggle_json)
            os.chmod(kaggle_json, 0o600)
            print("   OK - API key bulundu ve kopyalandi")
        else:
            print("\n   HATA: kaggle.json bulunamadi!")
            print("   Lutfen once Kaggle API key'inizi alin")
            sys.exit(1)
else:
    print("   OK - Kaggle API key mevcut")

# 3. Dataset secimi
print("\n" + "="*70)
print(" DATASET SECIMI")
print("="*70)

datasets = {
    '1': {
        'name': 'HAM10000',
        'id': 'kmader/skin-cancer-mnist-ham10000',
        'size': '~600 MB',
        'classes': 7,
        'description': 'En populer cilt hastanigi dataset (10,015 goruntu)'
    },
    '2': {
        'name': 'Psoriasis-Skin-Dataset',
        'id': 'pallapurajkumar/psoriasis-skin-dataset',
        'size': '~50 MB',
        'classes': 2,
        'description': 'Psoriasis + Normal cilt dataset'
    },
    '3': {
        'name': 'DermNet',
        'id': 'shubhamgoel27/dermnet',
        'size': '~1.2 GB',
        'classes': 23,
        'description': '23 farkli cilt durumu'
    },
    '4': {
        'name': 'Fitzpatrick17k',
        'id': 'mattop/fitzpatrick17k',
        'size': '~4 GB',
        'classes': 114,
        'description': 'Buyuk ve kapsamli dataset'
    }
}

print("\nMevcut Dataset'ler:\n")
for key, ds in datasets.items():
    print(f"{key}. {ds['name']}")
    print(f"   - Boyut: {ds['size']}")
    print(f"   - Siniflar: {ds['classes']}")
    print(f"   - Aciklama: {ds['description']}\n")

print("Hangi dataset'i indirmek istiyorsunuz? [1/2/3/4] (Onerilir: 1+2): ", end="")
choice = input().strip() or '1'

if choice not in datasets:
    print("Gecersiz secim, HAM10000 indiriliyor...")
    choice = '1'

selected_dataset = datasets[choice]

print(f"\nSecilen: {selected_dataset['name']}")
print(f"Dataset ID: {selected_dataset['id']}")

# 4. Indirme klasoru
download_dir = Path('datasets') / selected_dataset['name']
download_dir.mkdir(parents=True, exist_ok=True)

print(f"\nIndirme klasoru: {download_dir}")

# 5. Dataset indirme
print("\n" + "="*70)
print(" DATASET INDIRILIYOR")
print("="*70)
print(f"\nBoyut: {selected_dataset['size']} - Bu biraz zaman alabilir...")
print("Indirme basliyor...\n")

try:
    # Kaggle API ile indir
    kaggle.api.dataset_download_files(
        selected_dataset['id'],
        path=str(download_dir),
        unzip=True
    )
    print("\n   OK - Dataset indirildi ve acildi")
    
except Exception as e:
    print(f"\n   HATA: {e}")
    print("\nAlternatif indirme yontemi:")
    print(f"  1. https://www.kaggle.com/datasets/{selected_dataset['id']} adresine gidin")
    print("  2. 'Download' butonuna tiklayin")
    print(f"  3. Zip dosyasini {download_dir} klasorune kopyalayin")
    print("  4. Zip'i acin")
    sys.exit(1)

# 6. Dataset yapisi
print("\n" + "="*70)
print(" INDIRILEN DOSYALAR")
print("="*70)

files = list(download_dir.rglob('*'))
print(f"\nToplam dosya sayisi: {len(files)}")
print("\nIlk 10 dosya:")
for f in files[:10]:
    print(f"  - {f.name}")

# 7. Sonraki adim
print("\n" + "="*70)
print(" BASARILI!")
print("="*70)
print(f"\nDataset indirildi: {download_dir}")
print("\nSonraki adim:")
print("  python organize_and_train_multiclass.py")
print("\nBu script:")
print("  1. Indirilen veriyi organize edecek")
print("  2. Psoriasis verilerinizi ekleyecek")
print("  3. 8+ sinifli model egitecek")
print("  4. API'yi guncelleyecek")
print("\nHazir oldugunuzda yukaridaki komutu calistirin!")

