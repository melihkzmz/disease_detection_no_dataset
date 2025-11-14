#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Eye Disease Detection API
"""

import requests
import json
import sys

# Windows console UTF-8 support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

API_URL = "http://localhost:5001"

print("\n" + "="*60)
print("EYE DISEASE DETECTION API - TEST SCRIPT")
print("="*60)

# Test 1: Check API status
print("\n[TEST 1] Checking API status...")
try:
    response = requests.get(f"{API_URL}/")
    if response.status_code == 200:
        data = response.json()
        print("✅ API is running!")
        print(f"   Model: {data.get('model', 'N/A')}")
        print(f"   Classes: {data.get('num_classes', 'N/A')}")
        print(f"   Test Accuracy: {data.get('test_accuracy', 'N/A')}")
        print(f"   Top-3 Accuracy: {data.get('top_3_accuracy', 'N/A')}")
    else:
        print(f"❌ API returned status code: {response.status_code}")
        sys.exit(1)
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to API!")
    print("   Make sure the API is running: python eye_disease_api.py")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test 2: List classes
print("\n[TEST 2] Listing disease classes...")
try:
    response = requests.get(f"{API_URL}/classes")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {data['total_classes']} classes:")
        for cls in data['classes']:
            print(f"   - {cls['name']}")
    else:
        print(f"❌ Failed to list classes: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: Web interface check
print("\n[TEST 3] Checking web interface...")
try:
    response = requests.get(f"{API_URL}/web")
    if response.status_code == 200:
        print("✅ Web interface is accessible!")
        print(f"   URL: {API_URL}/web")
    else:
        print(f"❌ Web interface returned: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*60)
print("MANUAL TESTING")
print("="*60)
print("\n📝 To test image prediction:")
print(f"   1. Open web browser: {API_URL}/web")
print(f"   2. Upload a fundus image")
print(f"   3. View prediction results")
print("\n💡 Or use curl/Postman:")
print(f"   curl -X POST -F 'image=@path/to/image.jpg' {API_URL}/predict")
print("\n" + "="*60 + "\n")

