"""
Psoriasis görüntülerini organize et
Kaynak: PSORIASIS klasörü
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

def organize_psoriasis():
    """1,752 psoriasis görüntüsünü organize et"""
    
    print("\n" + "="*70)
    print(" 🔬 Psoriasis Görüntüleri Organize Ediliyor")
    print("="*70 + "\n")
    
    # Yollar
    source_folder = r'C:\Users\melih\dev\disease_detection\PSORIASIS'
    base_dir = r'C:\Users\melih\dev\disease_detection\skin_disease_data'
    train_dir = os.path.join(base_dir, 'train', 'psoriasis')
    val_dir = os.path.join(base_dir, 'validation', 'psoriasis')
    
    # Kaynak kontrolü
    if not os.path.exists(source_folder):
        print(f"❌ HATA: Kaynak klasör bulunamadı!")
        return False
    
    # Görüntü dosyalarını bul
    print(f"📂 Kaynak: {source_folder}")
    print(f"🔍 Görüntüler aranıyor...\n")
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
    images = []
    
    # Tüm alt klasörleri de dahil et
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith(image_extensions):
                full_path = os.path.join(root, file)
                images.append((full_path, file))
    
    print(f"✅ {len(images)} adet görüntü bulundu!\n")
    
    if len(images) == 0:
        print("❌ Görüntü bulunamadı!")
        return False
    
    # Çıkış klasörlerini oluştur
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Hedef klasörler:")
    print(f"   Train:      {train_dir}")
    print(f"   Validation: {val_dir}\n")
    
    # Train/Val bölümü (80/20)
    random.seed(42)
    train_imgs, val_imgs = train_test_split(
        images, 
        train_size=0.8, 
        random_state=42,
        shuffle=True
    )
    
    train_count = len(train_imgs)
    val_count = len(val_imgs)
    
    print(f"📊 Veri Bölümü:")
    print(f"   Train:      {train_count:4d} görüntü (80%)")
    print(f"   Validation: {val_count:4d} görüntü (20%)\n")
    
    # Train klasörüne kopyala
    print("📋 Train görüntüleri kopyalanıyor...")
    for i, (src_path, filename) in enumerate(train_imgs, 1):
        # Aynı isimli dosya varsa numaralandır
        dst_path = os.path.join(train_dir, filename)
        counter = 1
        base_name, ext = os.path.splitext(filename)
        
        while os.path.exists(dst_path):
            dst_path = os.path.join(train_dir, f"{base_name}_{counter}{ext}")
            counter += 1
        
        shutil.copy2(src_path, dst_path)
        
        if i % 100 == 0 or i == train_count:
            print(f"   {i}/{train_count} kopyalandı...")
    
    print(f"   ✅ {train_count} görüntü kopyalandı\n")
    
    # Validation klasörüne kopyala
    print("📋 Validation görüntüleri kopyalanıyor...")
    for i, (src_path, filename) in enumerate(val_imgs, 1):
        # Aynı isimli dosya varsa numaralandır
        dst_path = os.path.join(val_dir, filename)
        counter = 1
        base_name, ext = os.path.splitext(filename)
        
        while os.path.exists(dst_path):
            dst_path = os.path.join(val_dir, f"{base_name}_{counter}{ext}")
            counter += 1
        
        shutil.copy2(src_path, dst_path)
        
        if i % 100 == 0 or i == val_count:
            print(f"   {i}/{val_count} kopyalandı...")
    
    print(f"   ✅ {val_count} görüntü kopyalandı\n")
    
    print("="*70)
    print(" ✅ Psoriasis Verileri Organize Edildi!")
    print("="*70 + "\n")
    
    return True


def check_all_diseases():
    """Tüm hastalıklar için durum kontrolü"""
    print("\n" + "="*70)
    print(" 📊 TÜM HASTALIKLAR İÇİN VERİ DURUMU")
    print("="*70 + "\n")
    
    base_dir = r'C:\Users\melih\dev\disease_detection\skin_disease_data'
    diseases = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc', 
                'psoriasis', 'eczema']
    
    print("TRAIN:\n")
    train_ready = 0
    for disease in diseases:
        path = os.path.join(base_dir, 'train', disease)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) 
                        if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])
        else:
            count = 0
        
        if count == 0:
            status = "❌ BOŞŞ"
        elif count < 100:
            status = "⚠️  AZ  "
        else:
            status = "✅  OK  "
            train_ready += 1
        
        print(f"  {status} {disease:12s}: {count:5d} görüntü")
    
    print(f"\n  {'─'*50}")
    print(f"  📈 Hazır: {train_ready}/9 hastalık\n")
    
    print("VALIDATION:\n")
    val_ready = 0
    for disease in diseases:
        path = os.path.join(base_dir, 'validation', disease)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) 
                        if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])
        else:
            count = 0
        
        if count == 0:
            status = "❌ BOŞŞ"
        elif count < 20:
            status = "⚠️  AZ  "
        else:
            status = "✅  OK  "
            val_ready += 1
        
        print(f"  {status} {disease:12s}: {count:5d} görüntü")
    
    print(f"\n  {'─'*50}")
    print(f"  📈 Hazır: {val_ready}/9 hastalık")
    print("\n" + "="*70 + "\n")
    
    return train_ready, val_ready


if __name__ == '__main__':
    # Psoriasis'i organize et
    success = organize_psoriasis()
    
    if success:
        # Tüm durumu kontrol et
        train_ready, val_ready = check_all_diseases()
        
        print("🎯 SONRAKİ ADIMLAR:\n")
        
        if train_ready >= 8 and val_ready >= 8:
            print("✅ Çoğu hastalık için veri hazır!")
            print("\n🚀 Model eğitimine başlayabilirsiniz:")
            print("   cd Skin-Disease-Classifier")
            print("   python train_new_model.py\n")
        else:
            print(f"⚠️  Hazır: {train_ready}/9 hastalık")
            print("\n📝 Eksik hastalıklar için:")
            print("   1. Eczema verilerini toplayın (önemli!)")
            print("   2. Diğer 7 hastalık için HAM10000 dataset'ini indirin")
            print("   3. Veya sadece hazır olanlarla küçük model eğitin\n")
            print("💡 HAM10000: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000\n")

