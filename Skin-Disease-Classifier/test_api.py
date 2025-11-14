# -*- coding: utf-8 -*-
"""
API Test Script - Basit test görüntüsü ile
"""
import requests
import os
from PIL import Image
import io

API_URL = 'http://localhost:5000'

def test_api_status():
    """API durumunu test et"""
    print("\n" + "="*70)
    print(" API DURUM KONTROLU")
    print("="*70)
    
    try:
        response = requests.get(API_URL, timeout=5)
        data = response.json()
        
        print("OK - API Calisyor!")
        print(f"   Model: {data['model']}")
        print(f"   Accuracy: {data['accuracy']}")
        print(f"   Endpoint: {data['endpoint']}")
        return True
    except Exception as e:
        print(f"HATA - API Calismiyor: {e}")
        return False

def test_prediction():
    """Tahmin testi"""
    print("\n" + "="*70)
    print(" TAHMIN TESTI")
    print("="*70)
    
    # Test görüntüsü oluştur (basit siyah kare)
    img = Image.new('RGB', (224, 224), color='black')
    
    # BytesIO'ya kaydet
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    try:
        print("\nTest goruntusu gonderiliyor...")
        
        files = {'image': ('test.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(f'{API_URL}/predict', files=files, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("OK - Tahmin Basarili!")
            print(f"   Sinif: {data['class']}")
            print(f"   Guven: {data['percentage']}")
            print(f"   Psoriasis: {'Evet' if data['is_psoriasis'] else 'Hayir'}")
        else:
            print(f"HATA: {response.status_code}")
            print(f"   {response.text}")
    
    except Exception as e:
        print(f"HATA - Test Hatasi: {e}")

if __name__ == '__main__':
    print("\nPSORIASIS API TEST\n")
    
    if test_api_status():
        test_prediction()
    
    print("\n" + "="*70)
    print(" TEST TAMAMLANDI")
    print("="*70)
    print("\nWeb arayuzunu test etmek icin:")
    print("  http://localhost:8001/index_api.html")
    print()
