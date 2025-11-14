#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Lung Disease API
"""

import requests
import sys

# UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

API_URL = "http://localhost:5000"

print("=" * 70)
print(" LUNG DISEASE API TEST")
print("=" * 70)

# Test 1: API Status
print("\n[TEST 1] API Durumu kontrol ediliyor...")
try:
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        data = response.json()
        print("[OK] API calisıyor!")
        print(f"  Model: {data.get('model')}")
        print(f"  Accuracy: {data.get('accuracy')}")
        print(f"  Siniflar: {', '.join(data.get('classes', []))}")
    else:
        print(f"[FAIL] Status code: {response.status_code}")
except Exception as e:
    print(f"[ERROR] {e}")

# Test 2: Prediction with sample image
print("\n[TEST 2] Gorsel dosya ile tahmin testi...")
print("Not: Test etmek icin bir akciger X-Ray goruntusu gerekiyor.")
print("Dataset'ten ornek bir goruntu kullanabilirsiniz:")
print("  datasets/Lung Segmentation Data/Lung Segmentation Data/Test/COVID-19/images/")
print("\nManuel test icin:")
print(f"  Web arayuzu: {API_URL}/web")
print(f"  veya:")
print(f"  curl -X POST -F 'image=@xray.jpg' {API_URL}/predict")

print("\n" + "=" * 70)
print(" API HAZIR!")
print("=" * 70)
print(f"\nWeb Arayuzu: {API_URL}/web")
print(f"API Endpoint: {API_URL}/predict")
print("\nTarayicinizda acin ve test edin!")
print("=" * 70)

