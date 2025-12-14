#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy Malignant (cancer=1) images from bone-cancer-normal to Bone_4Class_Final

This script:
1. Reads classes.csv from bone-cancer-normal/train, test, valid
2. Finds images with cancer=1 (malignant)
3. Copies them to Bone_4Class_Final/train/test/val/Malignant_Tumor/
"""

import os
import shutil
import pandas as pd
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
SOURCE_BASE = BASE_DIR / 'datasets' / 'bone' / 'bone-cancer-normal'
TARGET_BASE = BASE_DIR / 'datasets' / 'bone' / 'Bone_4Class_Final'

# Split names mapping (bone-cancer-normal -> Bone_4Class_Final)
SPLIT_MAPPING = {
    'train': 'train',
    'test': 'test',
    'valid': 'val'  # valid -> val
}

TARGET_CLASS = 'Malignant_Tumor'

def copy_malignant_images(split_name):
    """
    Copy malignant images from source split to target Malignant_Tumor folder
    
    Args:
        split_name: 'train', 'test', or 'valid'
    """
    source_dir = SOURCE_BASE / split_name
    target_split = SPLIT_MAPPING.get(split_name, split_name)
    target_dir = TARGET_BASE / target_split / TARGET_CLASS
    
    # Try both classes.csv and _classes.csv
    classes_csv = source_dir / '_classes.csv'
    if not classes_csv.exists():
        classes_csv = source_dir / 'classes.csv'
    
    # Check if files exist
    if not source_dir.exists():
        print(f"⚠️  Source directory not found: {source_dir}")
        return 0
    
    if not classes_csv.exists():
        print(f"⚠️  classes.csv not found: {classes_csv}")
        return 0
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Read classes.csv
    try:
        df = pd.read_csv(classes_csv)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        print(f"\n📊 [{split_name}] Loaded {len(df)} entries from classes.csv")
    except Exception as e:
        print(f"❌ Error reading {classes_csv}: {e}")
        return 0
    
    # Check required columns
    if 'cancer' not in df.columns:
        print(f"❌ 'cancer' column not found in {classes_csv}")
        print(f"   Available columns: {list(df.columns)}")
        return 0
    
    # Find image filename column (strip whitespace from column names first)
    image_col = None
    for col in ['filename', 'image', 'file', 'img', 'image_name', 'name']:
        # Check both original and stripped versions
        if col in df.columns:
            image_col = col
            break
        # Also check with stripped whitespace
        stripped_cols = {c.strip(): c for c in df.columns}
        if col in stripped_cols:
            image_col = stripped_cols[col]
            break
    
    if image_col is None:
        # Try first column if no standard name found
        image_col = df.columns[0]
        print(f"⚠️  Using '{image_col}' as image filename column")
    
    # Filter malignant images (cancer == 1)
    malignant_df = df[df['cancer'] == 1]
    print(f"   Found {len(malignant_df)} malignant images (cancer=1)")
    
    if len(malignant_df) == 0:
        print(f"   ⚠️  No malignant images found in {split_name}")
        return 0
    
    # Copy images
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx, row in malignant_df.iterrows():
        image_filename = str(row[image_col])
        source_path = source_dir / image_filename
        
        if not source_path.exists():
            # Try different extensions
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                test_path = source_dir / f"{image_filename}{ext}"
                if test_path.exists():
                    source_path = test_path
                    found = True
                    break
            
            if not found:
                # Try with image filename as-is but check all files in directory
                all_files = list(source_dir.glob(f"{image_filename}*"))
                if all_files:
                    source_path = all_files[0]
                    found = True
            
            if not found or not source_path.exists():
                print(f"   ⚠️  Image not found: {image_filename}")
                error_count += 1
                continue
        
        # Target filename (handle duplicates by adding prefix)
        target_filename = source_path.name
        target_path = target_dir / target_filename
        
        # Handle duplicates
        if target_path.exists():
            # Add prefix to avoid overwrite
            stem = source_path.stem
            suffix = source_path.suffix
            counter = 1
            while target_path.exists():
                target_filename = f"{stem}_from_{split_name}_{counter}{suffix}"
                target_path = target_dir / target_filename
                counter += 1
        
        try:
            shutil.copy2(source_path, target_path)
            copied_count += 1
            if copied_count % 10 == 0:
                print(f"   ✓ Copied {copied_count} images...")
        except Exception as e:
            print(f"   ❌ Error copying {source_path.name}: {e}")
            error_count += 1
    
    print(f"\n✅ [{split_name}] Summary:")
    print(f"   Copied: {copied_count}")
    print(f"   Skipped: {skipped_count}")
    print(f"   Errors: {error_count}")
    print(f"   Target: {target_dir}")
    
    return copied_count

def main():
    print("="*70)
    print("COPY MALIGNANT IMAGES FROM bone-cancer-normal TO Bone_4Class_Final")
    print("="*70)
    
    # Check if source directory exists
    if not SOURCE_BASE.exists():
        print(f"❌ Source directory not found: {SOURCE_BASE}")
        return
    
    # Check if target directory exists
    if not TARGET_BASE.exists():
        print(f"❌ Target directory not found: {TARGET_BASE}")
        return
    
    print(f"\n📁 Source: {SOURCE_BASE}")
    print(f"📁 Target: {TARGET_BASE}")
    
    # Process each split
    splits = ['train', 'test', 'valid']
    total_copied = 0
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"Processing: {split}")
        print(f"{'='*70}")
        copied = copy_malignant_images(split)
        total_copied += copied
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total images copied: {total_copied}")
    print(f"\n✅ Done! Malignant images have been copied to:")
    print(f"   - {TARGET_BASE / 'train' / TARGET_CLASS}")
    print(f"   - {TARGET_BASE / 'test' / TARGET_CLASS}")
    print(f"   - {TARGET_BASE / 'val' / TARGET_CLASS}")

if __name__ == '__main__':
    main()

