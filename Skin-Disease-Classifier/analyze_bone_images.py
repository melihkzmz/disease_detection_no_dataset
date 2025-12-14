#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bone Dataset Image Analysis
Kemik veri setindeki görüntüleri analiz eder
"""
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def analyze_image(image_path):
    """Bir görüntüyü analiz eder"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        info = {
            'path': str(image_path),
            'format': img.format,
            'mode': img.mode,  # RGB, L (grayscale), RGBA, etc.
            'size': img.size,  # (width, height)
            'width': img.size[0],
            'height': img.size[1],
            'channels': len(img.getbands()) if hasattr(img, 'getbands') else (1 if img.mode == 'L' else 3),
            'is_grayscale': img.mode in ('L', 'LA', 'P'),
            'is_rgb': img.mode in ('RGB', 'RGBA'),
            'file_size_kb': os.path.getsize(image_path) / 1024,
        }
        
        # İstatistikler (eğer grayscale ise)
        if img.mode == 'L':
            info['min_pixel'] = int(img_array.min())
            info['max_pixel'] = int(img_array.max())
            info['mean_pixel'] = float(img_array.mean())
            info['std_pixel'] = float(img_array.std())
        elif img.mode == 'RGB':
            # Her kanal için istatistikler
            info['min_pixel'] = [int(img_array[:,:,i].min()) for i in range(3)]
            info['max_pixel'] = [int(img_array[:,:,i].max()) for i in range(3)]
            info['mean_pixel'] = [float(img_array[:,:,i].mean()) for i in range(3)]
        
        return info
    except Exception as e:
        return {'error': str(e), 'path': str(image_path)}

def analyze_dataset(dataset_path):
    """Tüm veri setini analiz eder"""
    print("\n" + "="*70)
    print("🔬 KEMIK VERI SETI GORUNTU ANALIZI")
    print("="*70)
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Hata: {dataset_path} bulunamadı!")
        return
    
    print(f"\n📁 Veri Seti: {dataset_path}")
    
    # Tüm sınıfları bul
    classes = []
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if split_path.exists():
            classes.extend([d.name for d in split_path.iterdir() if d.is_dir()])
    
    classes = sorted(list(set(classes)))
    print(f"\n🏷️  Sınıflar: {', '.join(classes)}")
    
    # Her split için analiz
    results = {}
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if not split_path.exists():
            continue
            
        print(f"\n" + "-"*70)
        print(f"📊 {split.upper()} SETI ANALIZI")
        print("-"*70)
        
        split_results = {
            'total_images': 0,
            'by_class': defaultdict(list),
            'formats': defaultdict(int),
            'modes': defaultdict(int),
            'sizes': [],
            'file_sizes': [],
            'sample_images': []
        }
        
        for class_name in classes:
            class_path = split_path / class_name
            if not class_path.exists():
                continue
            
            # Görüntü dosyalarını bul
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + \
                         list(class_path.glob('*.png')) + list(class_path.glob('*.JPG')) + \
                         list(class_path.glob('*.JPEG')) + list(class_path.glob('*.PNG'))
            
            if not image_files:
                continue
            
            print(f"\n  📂 {class_name}:")
            print(f"    Toplam görüntü: {len(image_files)}")
            
            # İlk 5 görüntüyü analiz et (örnek olarak)
            class_images = []
            for i, img_path in enumerate(image_files[:10]):  # İlk 10'u analiz et
                info = analyze_image(img_path)
                if 'error' not in info:
                    class_images.append(info)
                    split_results['total_images'] += 1
                    
                    # İstatistikler
                    split_results['by_class'][class_name].append(info)
                    split_results['formats'][info['format']] += 1
                    split_results['modes'][info['mode']] += 1
                    split_results['sizes'].append(info['size'])
                    split_results['file_sizes'].append(info['file_size_kb'])
                    
                    if len(split_results['sample_images']) < 3:
                        split_results['sample_images'].append(info)
            
            # İlk görüntü hakkında detaylı bilgi
            if class_images:
                first_img = class_images[0]
                print(f"    Format: {first_img['format']}")
                print(f"    Mode: {first_img['mode']} ({'Grayscale' if first_img['is_grayscale'] else 'RGB' if first_img['is_rgb'] else 'Other'})")
                print(f"    Boyut: {first_img['width']}x{first_img['height']}")
                print(f"    Kanallar: {first_img['channels']}")
                print(f"    Dosya boyutu: {first_img['file_size_kb']:.2f} KB")
                
                if 'mean_pixel' in first_img:
                    if isinstance(first_img['mean_pixel'], list):
                        print(f"    Ortalama pixel (RGB): {[f'{x:.1f}' for x in first_img['mean_pixel']]}")
                    else:
                        print(f"    Ortalama pixel: {first_img['mean_pixel']:.1f}")
        
        results[split] = split_results
        
        # Genel istatistikler
        if split_results['total_images'] > 0:
            print(f"\n  📈 {split.upper()} Genel İstatistikler:")
            print(f"    Toplam görüntü: {split_results['total_images']}")
            
            if split_results['formats']:
                print(f"    Formatlar: {dict(split_results['formats'])}")
            
            if split_results['modes']:
                print(f"    Modlar: {dict(split_results['modes'])}")
            
            if split_results['sizes']:
                sizes_array = np.array(split_results['sizes'])
                print(f"    Ortalama boyut: {sizes_array.mean(axis=0).astype(int)}")
                print(f"    Min boyut: {sizes_array.min(axis=0)}")
                print(f"    Max boyut: {sizes_array.max(axis=0)}")
            
            if split_results['file_sizes']:
                file_sizes = np.array(split_results['file_sizes'])
                print(f"    Ortalama dosya boyutu: {file_sizes.mean():.2f} KB")
                print(f"    Min dosya boyutu: {file_sizes.min():.2f} KB")
                print(f"    Max dosya boyutu: {file_sizes.max():.2f} KB")
    
    # Örnek görüntüleri göster
    print(f"\n" + "="*70)
    print("🖼️  ORNEK GORUNTULER")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        if split not in results or not results[split]['sample_images']:
            continue
        
        print(f"\n{split.upper()} Seti - Örnek Görüntüler:")
        for i, img_info in enumerate(results[split]['sample_images'][:3], 1):
            print(f"\n  {i}. {Path(img_info['path']).name}")
            print(f"     Sınıf: {Path(img_info['path']).parent.name}")
            print(f"     Boyut: {img_info['width']}x{img_info['height']}")
            print(f"     Mode: {img_info['mode']} ({'Grayscale' if img_info['is_grayscale'] else 'RGB' if img_info['is_rgb'] else 'Other'})")
            print(f"     Format: {img_info['format']}")
    
    # Özet rapor
    print(f"\n" + "="*70)
    print("📋 OZET RAPOR")
    print("="*70)
    
    total_all = sum(r['total_images'] for r in results.values())
    print(f"\nToplam analiz edilen görüntü: {total_all}")
    
    # Format dağılımı
    all_formats = defaultdict(int)
    for r in results.values():
        for fmt, count in r['formats'].items():
            all_formats[fmt] += count
    
    if all_formats:
        print(f"\nFormat dağılımı:")
        for fmt, count in sorted(all_formats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {fmt}: {count}")
    
    # Mode dağılımı
    all_modes = defaultdict(int)
    for r in results.values():
        for mode, count in r['modes'].items():
            all_modes[mode] += count
    
    if all_modes:
        print(f"\nMode (renk) dağılımı:")
        for mode, count in sorted(all_modes.items(), key=lambda x: x[1], reverse=True):
            mode_desc = 'Grayscale' if mode == 'L' else 'RGB' if mode == 'RGB' else mode
            print(f"  {mode} ({mode_desc}): {count}")
    
    # Boyut analizi
    all_sizes = []
    for r in results.values():
        all_sizes.extend(r['sizes'])
    
    if all_sizes:
        sizes_array = np.array(all_sizes)
        print(f"\nBoyut istatistikleri:")
        print(f"  Ortalama: {sizes_array.mean(axis=0).astype(int)}")
        print(f"  Min: {sizes_array.min(axis=0)}")
        print(f"  Max: {sizes_array.max(axis=0)}")
        print(f"  Benzersiz boyutlar: {len(np.unique(sizes_array, axis=0))}")
    
    print(f"\n" + "="*70)
    print("✅ Analiz tamamlandı!")
    print("="*70 + "\n")

if __name__ == '__main__':
    # Varsayılan yol
    dataset_path = 'datasets/bone/Bone_4Class_Final'
    
    # Komut satırından yol al
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    analyze_dataset(dataset_path)

