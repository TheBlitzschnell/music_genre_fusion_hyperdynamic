"""
CREATE CLEANED FEATURES - Remove Zero-Variance Features
=========================================================
This script creates a cleaned version of the 896d features by removing
zero-variance (dead) neurons. The fusion model will train on only the
active features.
"""

import numpy as np
import json
from pathlib import Path

BASE_DIR = Path.home() / "Documents" / "music_genre_fusion"
FUSION_DIR = BASE_DIR / "data" / "fusion_features"

print("="*80)
print("FEATURE CLEANUP - Remove Zero-Variance Features")
print("="*80)

# Load original features
print("\n[1/4] Loading original features...")
train_features = np.load(FUSION_DIR / "train_features_896d.npy")
train_labels = np.load(FUSION_DIR / "train_labels.npy")
val_features = np.load(FUSION_DIR / "val_features_896d.npy")
val_labels = np.load(FUSION_DIR / "val_labels.npy")

print(f"✓ Original shape: {train_features.shape}")

# Identify zero-variance features
print("\n[2/4] Identifying zero-variance features...")
feature_stds = train_features.std(axis=0)
zero_var_mask = feature_stds < 1e-6
active_mask = ~zero_var_mask

zero_var_count = zero_var_mask.sum()
active_count = active_mask.sum()

print(f"  Zero-variance: {zero_var_count}/896")
print(f"  Active: {active_count}/896 ({100*active_count/896:.1f}%)")

# Analyze by model
zero_var_indices = np.where(zero_var_mask)[0]
bigru_zeros = np.sum((zero_var_indices >= 0) & (zero_var_indices < 256))
dpa_zeros = np.sum((zero_var_indices >= 256) & (zero_var_indices < 768))
ecas_zeros = np.sum((zero_var_indices >= 768) & (zero_var_indices < 896))

print(f"\n  Breakdown by model:")
print(f"    CNN-BiGRU [0-255]:   {bigru_zeros}/256 removed ({256-bigru_zeros} kept)")
print(f"    DPA-CNN [256-767]:   {dpa_zeros}/512 removed ({512-dpa_zeros} kept)")
print(f"    ECAS-CNN [768-895]:  {ecas_zeros}/128 removed ({128-ecas_zeros} kept)")

# Remove zero-variance features
print("\n[3/4] Creating cleaned features...")
train_features_clean = train_features[:, active_mask]
val_features_clean = val_features[:, active_mask]

print(f"✓ Cleaned shape: {train_features_clean.shape}")
print(f"  Reduced from 896 to {active_count} features")

# Verify cleaned features
print("\n[4/4] Verifying cleaned features...")
clean_mean = train_features_clean.mean()
clean_std = train_features_clean.std()
clean_min = train_features_clean.min()
clean_max = train_features_clean.max()

print(f"\n  Cleaned training features:")
print(f"    Mean: {clean_mean:.6f}")
print(f"    Std:  {clean_std:.6f}")
print(f"    Min:  {clean_min:.4f}")
print(f"    Max:  {clean_max:.4f}")

# Check for remaining zero-variance
remaining_stds = train_features_clean.std(axis=0)
remaining_zeros = np.sum(remaining_stds < 1e-6)
print(f"\n  Remaining zero-variance features: {remaining_zeros} (should be 0)")

if remaining_zeros > 0:
    print(f"  ⚠️  WARNING: {remaining_zeros} zero-variance features remain!")
else:
    print(f"  ✓ All zero-variance features removed successfully")

# Save cleaned features
print("\n" + "="*80)
print("SAVING CLEANED FEATURES")
print("="*80)

np.save(FUSION_DIR / f"train_features_{active_count}d_clean.npy", train_features_clean)
np.save(FUSION_DIR / "train_labels_clean.npy", train_labels)
np.save(FUSION_DIR / f"val_features_{active_count}d_clean.npy", val_features_clean)
np.save(FUSION_DIR / "val_labels_clean.npy", val_labels)
np.save(FUSION_DIR / "active_feature_mask.npy", active_mask)

# Save mapping information
feature_mapping = {
    'original_dim': 896,
    'cleaned_dim': int(active_count),
    'removed_count': int(zero_var_count),
    'active_mask': active_mask.tolist(),
    'zero_variance_indices': zero_var_indices.tolist(),
    'breakdown': {
        'cnn_bigru': {
            'original': 256,
            'removed': int(bigru_zeros),
            'kept': int(256 - bigru_zeros)
        },
        'dpa_cnn': {
            'original': 512,
            'removed': int(dpa_zeros),
            'kept': int(512 - dpa_zeros)
        },
        'ecas_cnn': {
            'original': 128,
            'removed': int(ecas_zeros),
            'kept': int(128 - ecas_zeros)
        }
    },
    'usage_note': f'Use train_features_{active_count}d_clean.npy for fusion training'
}

with open(FUSION_DIR / "feature_mapping.json", 'w') as f:
    json.dump(feature_mapping, f, indent=2)

print(f"\n✓ Saved cleaned features:")
print(f"  - train_features_{active_count}d_clean.npy")
print(f"  - val_features_{active_count}d_clean.npy")
print(f"  - train_labels_clean.npy")
print(f"  - val_labels_clean.npy")
print(f"  - active_feature_mask.npy")
print(f"  - feature_mapping.json")

# Summary
print("\n" + "="*80)
print("CLEANUP SUMMARY")
print("="*80)

print(f"\n✓ Feature dimension reduced: 896 → {active_count}")
print(f"✓ Zero-variance features removed: {zero_var_count}")
print(f"✓ All features now have non-zero variance")
print(f"✓ Ready for fusion training!")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print(f"\n1. Use the cleaned features for fusion training:")
print(f"   - Input dimension: {active_count} (not 896)")
print(f"   - Features: train_features_{active_count}d_clean.npy")
print(f"   - Labels: train_labels_clean.npy")

print(f"\n2. Update fusion model architecture:")
print(f"   - Change input_dim from 896 to {active_count}")
print(f"   - Everything else remains the same")

print(f"\n3. For inference on new data:")
print(f"   - Extract 896d features as usual")
print(f"   - Apply active_feature_mask.npy to get {active_count}d")
print(f"   - Then pass to fusion model")

print("\n" + "="*80)