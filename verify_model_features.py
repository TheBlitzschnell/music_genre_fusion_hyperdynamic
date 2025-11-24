"""
FEATURE QUALITY VERIFICATION - FIXED VERSION
==============================================
Handles zero-variance features and uses correct scikit-learn API
"""

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from collections import Counter

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

BASE_DIR = Path.home() / "music_genre_fusion"
FUSION_DIR = BASE_DIR / "data" / "fusion_features"
VIZ_DIR = BASE_DIR / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("FEATURE QUALITY VERIFICATION - FIXED")
print("="*80)

# Load features
print("\n[1/8] Loading features...")
try:
    train_features = np.load(FUSION_DIR / "train_features_896d.npy")
    train_labels = np.load(FUSION_DIR / "train_labels.npy")
    val_features = np.load(FUSION_DIR / "val_features_896d.npy")
    val_labels = np.load(FUSION_DIR / "val_labels.npy")
    feature_mean = np.load(FUSION_DIR / "feature_mean.npy")
    feature_std = np.load(FUSION_DIR / "feature_std.npy")
    
    with open(FUSION_DIR / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded successfully:")
    print(f"  Train: {train_features.shape}")
    print(f"  Val:   {val_features.shape}")
except FileNotFoundError as e:
    print(f"\n❌ Error: Could not find feature files!")
    exit(1)

checks_passed = 0
total_checks = 0

# Dimensionality checks
print("\n[2/8] Dimensionality checks...")
total_checks += 1
if train_features.shape[1] == 896 and val_features.shape[1] == 896:
    print(f"✓ Feature dimension: 896")
    checks_passed += 1
else:
    print(f"✗ Feature dimension mismatch")

# NaN/Inf detection
print("\n[3/8] NaN/Inf detection...")
train_nan = np.isnan(train_features).sum()
train_inf = np.isinf(train_features).sum()
val_nan = np.isnan(val_features).sum()
val_inf = np.isinf(val_features).sum()

total_checks += 1
if train_nan == 0 and train_inf == 0 and val_nan == 0 and val_inf == 0:
    print(f"✓ No NaN/Inf values detected")
    checks_passed += 1
else:
    print(f"✗ Found {train_nan + val_nan} NaN, {train_inf + val_inf} Inf values")

# Zero-variance analysis
print("\n[4/8] Zero-variance feature analysis...")
feature_stds = train_features.std(axis=0)
zero_var_mask = feature_stds < 1e-6
zero_var_count = zero_var_mask.sum()
zero_var_indices = np.where(zero_var_mask)[0]

print(f"\nZero-variance features: {zero_var_count}/896 ({100*zero_var_count/896:.1f}%)")

if zero_var_count > 0:
    # Analyze which model contributes zero-variance features
    bigru_zeros = np.sum((zero_var_indices >= 0) & (zero_var_indices < 256))
    dpa_zeros = np.sum((zero_var_indices >= 256) & (zero_var_indices < 768))
    ecas_zeros = np.sum((zero_var_indices >= 768) & (zero_var_indices < 896))
    
    print(f"\nBreakdown by model:")
    print(f"  CNN-BiGRU [0-255]:    {bigru_zeros}/256 ({100*bigru_zeros/256:.1f}%)")
    print(f"  DPA-CNN [256-767]:    {dpa_zeros}/512 ({100*dpa_zeros/512:.1f}%)")
    print(f"  ECAS-CNN [768-895]:   {ecas_zeros}/128 ({100*ecas_zeros/128:.1f}%)")
    
    print(f"\nFirst 20 zero-variance indices: {zero_var_indices[:20].tolist()}")
    
    # Decision: Remove or keep?
    if zero_var_count > 100:  # More than ~11% are dead
        print(f"\n⚠️  WARNING: {zero_var_count} zero-variance features is concerning!")
        print(f"   Recommendation: Remove these features before fusion training")
        print(f"   This suggests some neurons are 'dead' in the base models")
    else:
        print(f"\n✓ Acceptable number of zero-variance features")
        print(f"   Can proceed but consider investigating the dead neurons")

# Normalization with zero-variance handling
print("\n[5/8] Normalization verification...")

# For non-zero variance features only
active_mask = ~zero_var_mask
train_mean = train_features[:, active_mask].mean()
train_std = train_features[:, active_mask].std()
val_mean = val_features[:, active_mask].mean()
val_std = val_features[:, active_mask].std()

print(f"\nTraining set statistics (active features only, n={active_mask.sum()}):")
print(f"  Mean: {train_mean:.6f} (should be ~0.000000)")
print(f"  Std:  {train_std:.6f} (should be ~1.000000)")
print(f"  Min:  {train_features.min():.4f}")
print(f"  Max:  {train_features.max():.4f}")

print(f"\nValidation set statistics:")
print(f"  Mean: {val_mean:.6f}")
print(f"  Std:  {val_std:.6f}")
print(f"  Min:  {val_features.min():.4f}")
print(f"  Max:  {val_features.max():.4f}")

total_checks += 1
if abs(train_mean) < 0.01 and 0.85 <= train_std <= 1.15:
    print(f"\n✓ Normalization acceptable (std within 15% of 1.0)")
    checks_passed += 1
else:
    print(f"\n⚠️  Normalization could be better")

# Class distribution
print("\n[6/8] Class distribution analysis...")
train_dist = Counter(train_labels)
val_dist = Counter(val_labels)

print(f"\nTraining set distribution:")
for cls in range(10):
    count = train_dist[cls]
    pct = 100.0 * count / len(train_labels)
    print(f"  Class {cls} ({GENRES[cls]:10s}): {count:4d} samples ({pct:5.1f}%)")

total_checks += 1
train_counts = [train_dist[i] for i in range(10)]
train_balance = max(train_counts) / min(train_counts)
if train_balance < 1.5:
    print(f"\n✓ Class distribution balanced (ratio: {train_balance:.2f}x)")
    checks_passed += 1
else:
    print(f"\n⚠️  Class imbalance detected: {train_balance:.2f}x")

# Feature correlation (only on active features)
print("\n[7/8] Feature correlation analysis...")
if active_mask.sum() > 50:  # Only if we have enough features
    sample_size = min(300, len(train_features))
    sample_indices = np.random.choice(len(train_features), sample_size, replace=False)
    
    # Use only active features
    train_sample = train_features[sample_indices][:, active_mask]
    
    # Sample subset for correlation (correlation on 700+ features is slow)
    if train_sample.shape[1] > 200:
        feature_sample_indices = np.random.choice(train_sample.shape[1], 200, replace=False)
        train_sample = train_sample[:, feature_sample_indices]
    
    try:
        corr_matrix = np.corrcoef(train_sample.T)
        # Remove diagonal
        n = corr_matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        avg_corr = np.abs(corr_matrix[mask]).mean()
        
        print(f"  Average correlation (sampled active features): {avg_corr:.3f}")
        
        total_checks += 1
        if avg_corr < 0.7:
            print(f"  ✓ Features show good diversity (correlation < 0.7)")
            checks_passed += 1
        else:
            print(f"  ⚠️  High correlation detected (>{avg_corr:.3f})")
    except:
        print("  ⚠️  Could not compute correlations")
else:
    print("  ⚠️  Too few active features for correlation analysis")

# t-SNE visualization
print("\n[8/8] t-SNE visualization...")
try:
    from sklearn.manifold import TSNE
    
    # Use validation set, only active features
    tsne_sample_size = min(150, len(val_features))
    tsne_indices = np.random.choice(len(val_features), tsne_sample_size, replace=False)
    tsne_features = val_features[tsne_indices][:, active_mask]
    tsne_labels = val_labels[tsne_indices]
    
    print(f"  Computing t-SNE for {tsne_sample_size} samples, {active_mask.sum()} active features...")
    
    # Use max_iter instead of n_iter (scikit-learn API change)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, tsne_sample_size-1), 
                max_iter=1000, verbose=0)
    tsne_result = tsne.fit_transform(tsne_features)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for cls in range(10):
        mask = tsne_labels == cls
        if mask.sum() > 0:
            plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                       c=[colors[cls]], label=GENRES[cls], alpha=0.7, s=50)
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(f't-SNE Visualization ({active_mask.sum()} active features)', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    viz_path = VIZ_DIR / "feature_tsne_fixed.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ t-SNE visualization saved: {viz_path}")
    
    total_checks += 1
    checks_passed += 1  # t-SNE completed
    
except ImportError:
    print("  ⚠️  scikit-learn not installed, skipping t-SNE")
except Exception as e:
    print(f"  ⚠️  t-SNE failed: {e}")

# Class separability
print("\n[9/9] Class separability analysis...")
centroids = np.zeros((10, active_mask.sum()))
for cls in range(10):
    mask = train_labels == cls
    centroids[cls] = train_features[mask][:, active_mask].mean(axis=0)

centroid_distances = squareform(pdist(centroids, metric='euclidean'))
np.fill_diagonal(centroid_distances, np.inf)

min_dist = centroid_distances[centroid_distances > 0].min()
max_dist = centroid_distances.max()
mean_dist = centroid_distances[centroid_distances > 0].mean()

print(f"\nClass centroid separability (active features):")
print(f"  Min distance:  {min_dist:.2f}")
print(f"  Max distance:  {max_dist:.2f}")
print(f"  Mean distance: {mean_dist:.2f}")

min_idx = np.unravel_index(centroid_distances.argmin(), centroid_distances.shape)
max_idx = np.unravel_index(centroid_distances.argmax(), centroid_distances.shape)

print(f"\nMost similar:    {GENRES[min_idx[0]]} ↔ {GENRES[min_idx[1]]} ({centroid_distances[min_idx]:.2f})")
print(f"Most dissimilar: {GENRES[max_idx[0]]} ↔ {GENRES[max_idx[1]]} ({centroid_distances[max_idx]:.2f})")

plt.figure(figsize=(10, 8))
sns.heatmap(centroid_distances, annot=True, fmt='.1f', cmap='YlOrRd',
           xticklabels=GENRES, yticklabels=GENRES, 
           cbar_kws={'label': 'Euclidean Distance'})
plt.title(f'Class Centroid Distances ({active_mask.sum()} active features)', 
         fontsize=14, fontweight='bold')
plt.tight_layout()
heatmap_path = VIZ_DIR / "centroid_distances_fixed.png"
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✓ Distance heatmap saved: {heatmap_path}")

total_checks += 1
if min_dist > 3.0:  # Adjusted threshold for fewer features
    print(f"\n✓ Good class separability (min distance: {min_dist:.2f})")
    checks_passed += 1
else:
    print(f"\n⚠️  Moderate separability (min distance: {min_dist:.2f})")

# Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

print(f"\nChecks passed: {checks_passed}/{total_checks}")
print(f"Active features: {active_mask.sum()}/{896} ({100*active_mask.sum()/896:.1f}%)")
print(f"Zero-variance features: {zero_var_count}")

# Decision logic
if zero_var_count > 200:
    status = "NEEDS_FIXING"
    print("\n❌ TOO MANY ZERO-VARIANCE FEATURES")
    print("   Action required: Investigate feature extraction")
elif zero_var_count > 100:
    status = "ACCEPTABLE_WITH_CLEANUP"
    print("\n⚠️  PROCEED WITH CAUTION")
    print("   Recommendation: Remove zero-variance features before fusion")
else:
    status = "READY"
    print("\n✅ FEATURES READY FOR FUSION")
    print("   Can proceed to fusion training")

# Save report
report = {
    'status': status,
    'checks_passed': f"{checks_passed}/{total_checks}",
    'active_features': int(active_mask.sum()),
    'zero_variance_features': int(zero_var_count),
    'zero_variance_breakdown': {
        'cnn_bigru': int(bigru_zeros) if zero_var_count > 0 else 0,
        'dpa_cnn': int(dpa_zeros) if zero_var_count > 0 else 0,
        'ecas_cnn': int(ecas_zeros) if zero_var_count > 0 else 0
    },
    'normalization': {
        'train_mean': float(train_mean),
        'train_std': float(train_std)
    },
    'separability': {
        'min_distance': float(min_dist),
        'mean_distance': float(mean_dist)
    }
}

with open(FUSION_DIR / "verification_report_fixed.json", 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n📄 Report saved: {FUSION_DIR / 'verification_report_fixed.json'}")

# Save list of zero-variance features if needed
if zero_var_count > 0:
    np.save(FUSION_DIR / "zero_variance_indices.npy", zero_var_indices)
    np.save(FUSION_DIR / "active_feature_mask.npy", active_mask)
    print(f"📄 Zero-variance info saved for cleanup")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if status == "READY":
    print("\n✅ You can proceed to fusion training!")
    print("   The features are good quality despite some dead neurons.")
    
elif status == "ACCEPTABLE_WITH_CLEANUP":
    print("\n⚠️  Option 1: Use only active features (recommended)")
    print("   - Run: python create_cleaned_features.py")
    print("   - This will create a version with only active features")
    print("   - Fusion model will train on ~710 features instead of 896")
    
    print("\n⚠️  Option 2: Proceed as-is")
    print("   - Zero-variance features won't hurt training (just ignored)")
    print("   - Slightly less efficient but will work fine")
    
else:
    print("\n❌ Action required before fusion training:")
    print("   1. Check which base models have many dead neurons")
    print("   2. Consider retraining problematic models")
    print("   3. Or remove zero-variance features and proceed")

print("\n" + "="*80)