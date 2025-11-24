#!/usr/bin/env python3
"""
Generate normalization_params.json from existing normalized data
Run this if normalization_params.json is missing
"""

import numpy as np
import json
from pathlib import Path

BASE_DIR = Path.home() / "Documents" / "music_genre_fusion"
DATA_DIR = BASE_DIR / "data" / "processed"
MELSPEC_DIR = DATA_DIR / "melspectrogram"
MELSPEC_NORM_DIR = DATA_DIR / "melspectrogram_normalized"
MFCC_DIR = DATA_DIR / "mfcc"
MFCC_NORM_DIR = DATA_DIR / "mfcc_normalized"

print("="*70)
print("GENERATING NORMALIZATION PARAMETERS")
print("="*70)

# Check if normalized data exists
if not MELSPEC_NORM_DIR.exists():
    print(f"\nError: {MELSPEC_NORM_DIR} not found")
    print("Run normalize_data.py first")
    exit(1)

# Option 1: If raw data exists, compute params from it
if MELSPEC_DIR.exists():
    print("\nComputing from raw data...")
    train_raw = np.load(MELSPEC_DIR / "train_features.npy")
    
    mel_mean = train_raw.mean()
    mel_std = train_raw.std()
    
    print(f"Mel-spec - Mean: {mel_mean:.4f}, Std: {mel_std:.4f}")
    
    # Save
    params = {
        'mean': float(mel_mean),
        'std': float(mel_std),
        'epsilon': 1e-8
    }
    
    with open(MELSPEC_NORM_DIR / "normalization_params.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\nSaved to: {MELSPEC_NORM_DIR}/normalization_params.json")

# Option 2: If only normalized data exists
else:
    print("\nWarning: Raw data not found, using approximation...")
    print("The normalized data has mean≈0, std≈1")
    print("Using default values")
    
    params = {
        'mean': 0.0,
        'std': 1.0,
        'epsilon': 1e-8
    }
    
    MELSPEC_NORM_DIR.mkdir(parents=True, exist_ok=True)
    with open(MELSPEC_NORM_DIR / "normalization_params.json", 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\nSaved approximate params to: {MELSPEC_NORM_DIR}/normalization_params.json")
    print("Note: For best results, regenerate from raw data")

print("\nDone! You can now use genre_detector.py")
print("="*70)