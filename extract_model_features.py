"""
FEATURE EXTRACTION FROM TRAINED MODELS - FIXED VERSION
=======================================================
Uses DUPLICATION instead of random projection for CNN-BiGRU (128→256).
This ensures consistency between training and inference.

Changes from original:
- FeatureExtractor now uses duplication instead of random nn.Linear projection
- This matches what the detector does, ensuring consistent features

Run this, then:
1. python3 create_cleaned_features.py
2. python3 train_fusion.py
3. python3 genre_detector.py <audio_file>
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import time

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path.home() / "Documents" / "music_genre_fusion"
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
FUSION_DIR = BASE_DIR / "data" / "fusion_features"
FUSION_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32

print("="*80)
print("FEATURE EXTRACTION FROM TRAINED MODELS (FIXED)")
print("="*80)
print(f"Device: {device}")
print(f"Project: {BASE_DIR}")
print(f"Target: 896-dim features (256 + 512 + 128)")
print(f"CNN-BiGRU: Using DUPLICATION (not random projection)")
print("="*80)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class CNNBiGRU_Reduced(nn.Module):
    """CNN-BiGRU architecture - same as training"""
    def __init__(self, nc=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.15)
        self.drop3 = nn.Dropout2d(0.2)
        self.drop4 = nn.Dropout2d(0.25)
        
        self.bigru = nn.GRU(1024, 64, 1, bidirectional=True, batch_first=True, dropout=0)
        
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(128, nc)
        )
    
    def forward(self, x, return_features=False):
        x = self.drop1(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.drop3(self.pool(F.relu(self.bn3(self.conv3(x)))))
        x = self.drop4(self.pool(F.relu(self.bn4(self.conv4(x)))))
        
        b, c, h, w = x.size()
        x = F.adaptive_avg_pool2d(x, (4, w))
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, -1)
        
        x, _ = self.bigru(x)
        
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        features = torch.sum(x * attn_weights, dim=1)  # 128-dim
        
        if return_features:
            return features
        
        return self.classifier(features)


class DPAModule(nn.Module):
    """Dual Parallel Attention"""
    def __init__(self, channels, reduction=8):
        super(DPAModule, self).__init__()
        self.theta = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.phi = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.psi = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.g = nn.Conv2d(channels // reduction, channels, 3, padding=1, bias=False)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        theta = self.theta(x).view(batch_size, -1, H * W)
        phi = self.phi(x).view(batch_size, -1, H * W)
        psi = self.psi(x).view(batch_size, -1, H * W)
        
        attention = torch.bmm(theta.permute(0, 2, 1), phi)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(psi, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, H, W)
        out = self.g(out)
        channel_att = torch.sigmoid(out)
        
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_input))
        
        x = x * channel_att * spatial_att
        return x


class DPACNN_HyperDynamic(nn.Module):
    """DPA-CNN architecture - same as training"""
    def __init__(self, num_classes=10):
        super(DPACNN_HyperDynamic, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(96)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.1)
        
        self.conv2 = nn.Conv2d(96, 160, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(160)
        self.dpa2 = DPAModule(160)
        self.avgpool2 = nn.AvgPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.15)
        
        self.conv3 = nn.Conv2d(160, 320, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(320)
        self.dpa3 = DPAModule(320)
        self.avgpool3 = nn.AvgPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.2)
        
        self.conv4 = nn.Conv2d(320, 384, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(384)
        self.dpa4 = DPAModule(384)
        self.avgpool4 = nn.AvgPool2d(2, 2)
        self.drop4 = nn.Dropout2d(0.2)
        
        self.conv5 = nn.Conv2d(384, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.dpa5 = DPAModule(512)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.drop5 = nn.Dropout2d(0.2)
        
        self.feature_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, return_features=False):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.maxpool(x)
        x = self.drop1(x)
        
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.dpa2(x)
        x = self.avgpool2(x)
        x = self.drop2(x)
        
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.dpa3(x)
        x = self.avgpool3(x)
        x = self.drop3(x)
        
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.dpa4(x)
        x = self.avgpool4(x)
        x = self.drop4(x)
        
        x = F.relu(self.bn5(self.conv5(x)), inplace=True)
        x = self.dpa5(x)
        x = self.global_pool(x)
        x = self.drop5(x)
        
        x = x.view(x.size(0), -1)
        features = self.feature_proj(x)  # 512-dim
        
        if return_features:
            return features
        
        return self.classifier(features)


class ECAModule(nn.Module):
    """Efficient Channel Attention - matches training"""
    def __init__(self, c, g=2, b=1):
        super().__init__()
        k = int(abs((np.log2(c) + b) / g))
        k = k if k % 2 else k + 1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, k, padding=k//2, bias=False)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sig(y).expand_as(x)


class ECASCNN_HyperDynamic(nn.Module):
    """ECAS-CNN architecture - matches training"""
    def __init__(self, nc=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 192, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(192)
        self.eca1 = ECAModule(192)
        self.conv2 = nn.Conv2d(192, 256, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.eca2 = ECAModule(256)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(384)
        self.eca3 = ECAModule(384)
        self.conv4 = nn.Conv2d(384, 512, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.eca4 = ECAModule(512)
        self.pool = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d((1, 2))
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.drop1, self.drop2 = nn.Dropout2d(0.1), nn.Dropout2d(0.15)
        self.drop3, self.drop4 = nn.Dropout2d(0.2), nn.Dropout2d(0.2)
        self.feat = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(0.3)
        )
        self.cls = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(128, nc)
        )
    
    def forward(self, x, return_features=False):
        x = self.drop1(self.pool(self.eca1(F.relu(self.bn1(self.conv1(x))))))
        x = self.drop2(self.pool(self.eca2(F.relu(self.bn2(self.conv2(x))))))
        x = self.drop3(self.pool(self.eca3(F.relu(self.bn3(self.conv3(x))))))
        x = self.drop4(self.pool4(self.eca4(F.relu(self.bn4(self.conv4(x))))))
        x = self.gpool(x).view(x.size(0), -1)
        features = self.feat(x)  # 128-dim
        
        if return_features:
            return features
        
        return self.cls(features)


# ============================================================================
# FEATURE EXTRACTION - FIXED VERSION
# ============================================================================

class FeatureExtractor:
    """Feature extractor using DUPLICATION instead of random projection"""
    
    def __init__(self, model, feature_dim, target_dim=None):
        """
        Args:
            model: Trained PyTorch model
            feature_dim: Dimension of features extracted by model
            target_dim: Target dimension (if different, uses duplication)
        """
        self.model = model
        self.model.eval()
        self.feature_dim = feature_dim
        self.target_dim = target_dim if target_dim else feature_dim
        self.use_duplication = target_dim is not None and target_dim != feature_dim
        
        if self.use_duplication:
            assert target_dim == feature_dim * 2, \
                f"Duplication only supports 2x expansion, got {feature_dim}→{target_dim}"
    
    def extract(self, data_loader):
        """Extract features for entire dataset"""
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc=f"Extracting {self.target_dim}d", 
                                      leave=False, ncols=100):
                inputs = inputs.to(device)
                
                # Get features from model
                features = self.model(inputs, return_features=True)
                
                # Apply duplication if needed (instead of random projection!)
                if self.use_duplication:
                    features = torch.cat([features, features], dim=1)
                
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())
        
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        return features, labels


class SimpleDataset(Dataset):
    """Simple dataset wrapper"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1) if len(X.shape) == 3 else torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model(model_class, checkpoint_path, **kwargs):
    """Load a trained model from checkpoint"""
    model = model_class(**kwargs).to(device)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded: {checkpoint_path.name}")
    return model


def extract_all_features():
    """Extract features from all three models"""
    
    print("\n" + "="*80)
    print("STEP 1: LOAD TRAINED MODELS")
    print("="*80)
    
    # Load CNN-BiGRU
    bigru_paths = [
        MODEL_DIR / "cnn_bigru" / "cnn_bigru_hyperdynamic_best.pth",
        MODEL_DIR / "dpa_cnn" / "cnn_bigru_hyperdynamic_best.pth",
    ]
    bigru_path = None
    for p in bigru_paths:
        if p.exists():
            bigru_path = p
            break
    
    if bigru_path is None:
        raise FileNotFoundError("CNN-BiGRU model not found!")
    
    bigru_model = load_model(CNNBiGRU_Reduced, bigru_path, nc=10)
    # FIXED: Use duplication (target_dim=256) instead of random projection
    bigru_extractor = FeatureExtractor(bigru_model, feature_dim=128, target_dim=256)
    
    # Load DPA-CNN
    dpa_path = MODEL_DIR / "dpa_cnn" / "dpa_cnn_best.pth"
    dpa_model = load_model(DPACNN_HyperDynamic, dpa_path, num_classes=10)
    dpa_extractor = FeatureExtractor(dpa_model, feature_dim=512)
    
    # Load ECAS-CNN
    ecas_path = MODEL_DIR / "ecas_cnn" / "ecas_cnn_hyperdynamic_best.pth"
    ecas_model = load_model(ECASCNN_HyperDynamic, ecas_path, nc=10)
    ecas_extractor = FeatureExtractor(ecas_model, feature_dim=128)
    
    print(f"\n✓ All models loaded successfully!")
    print(f"  CNN-BiGRU: 128d → 256d (DUPLICATION)")
    print(f"  DPA-CNN:   512d")
    print(f"  ECAS-CNN:  128d")
    print(f"  Total:     896d concatenated")
    
    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("STEP 2: LOAD DATA")
    print("="*80)
    
    # Mel-spectrograms
    mel_dir = DATA_DIR / "melspectrogram_normalized"
    if not mel_dir.exists():
        mel_dir = DATA_DIR / "melspectrogram"
    
    print(f"\nLoading Mel-spectrograms from: {mel_dir}")
    train_mel_X = np.load(mel_dir / "train_features.npy")
    train_mel_y = np.load(mel_dir / "train_labels.npy")
    val_mel_X = np.load(mel_dir / "val_features.npy")
    val_mel_y = np.load(mel_dir / "val_labels.npy")
    print(f"✓ Train: {train_mel_X.shape}, Val: {val_mel_X.shape}")
    
    # MFCC
    mfcc_dir = DATA_DIR / "mfcc_normalized"
    if not mfcc_dir.exists():
        mfcc_dir = DATA_DIR / "mfcc"
    
    print(f"\nLoading MFCC from: {mfcc_dir}")
    train_mfcc_X = np.load(mfcc_dir / "train_features.npy")
    train_mfcc_y = np.load(mfcc_dir / "train_labels.npy")
    val_mfcc_X = np.load(mfcc_dir / "val_features.npy")
    val_mfcc_y = np.load(mfcc_dir / "val_labels.npy")
    print(f"✓ Train: {train_mfcc_X.shape}, Val: {val_mfcc_X.shape}")
    
    # -------------------------------------------------------------------------
    # Extract features
    # -------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("STEP 3: EXTRACT FEATURES")
    print("="*80)
    
    start_time = time.time()
    
    # Create data loaders
    train_mel_loader = DataLoader(SimpleDataset(train_mel_X, train_mel_y), 
                                  batch_size=BATCH_SIZE, shuffle=False)
    val_mel_loader = DataLoader(SimpleDataset(val_mel_X, val_mel_y), 
                                batch_size=BATCH_SIZE, shuffle=False)
    train_mfcc_loader = DataLoader(SimpleDataset(train_mfcc_X, train_mfcc_y), 
                                   batch_size=BATCH_SIZE, shuffle=False)
    val_mfcc_loader = DataLoader(SimpleDataset(val_mfcc_X, val_mfcc_y), 
                                 batch_size=BATCH_SIZE, shuffle=False)
    
    # Extract features
    print("\n[1/3] CNN-BiGRU Features (128d → 256d via duplication)")
    train_bigru_features, train_labels = bigru_extractor.extract(train_mel_loader)
    val_bigru_features, val_labels = bigru_extractor.extract(val_mel_loader)
    print(f"  ✓ Train: {train_bigru_features.shape}, Val: {val_bigru_features.shape}")
    
    print("\n[2/3] DPA-CNN Features (512d)")
    train_dpa_features, _ = dpa_extractor.extract(train_mel_loader)
    val_dpa_features, _ = dpa_extractor.extract(val_mel_loader)
    print(f"  ✓ Train: {train_dpa_features.shape}, Val: {val_dpa_features.shape}")
    
    print("\n[3/3] ECAS-CNN Features (128d)")
    train_ecas_features, _ = ecas_extractor.extract(train_mfcc_loader)
    val_ecas_features, _ = ecas_extractor.extract(val_mfcc_loader)
    print(f"  ✓ Train: {train_ecas_features.shape}, Val: {val_ecas_features.shape}")
    
    extraction_time = time.time() - start_time
    print(f"\n✓ Feature extraction complete in {extraction_time:.1f}s")
    
    # -------------------------------------------------------------------------
    # Concatenate features
    # -------------------------------------------------------------------------
    
    print("\n" + "="*80)
    print("STEP 4: CONCATENATE FEATURES")
    print("="*80)
    
    train_features = np.concatenate([
        train_bigru_features,  # 256d (duplicated)
        train_dpa_features,    # 512d
        train_ecas_features    # 128d
    ], axis=1)
    
    val_features = np.concatenate([
        val_bigru_features,
        val_dpa_features,
        val_ecas_features
    ], axis=1)
    
    print(f"\n✓ Concatenated features:")
    print(f"  Train: {train_features.shape}")
    print(f"  Val:   {val_features.shape}")
    
    assert train_features.shape[1] == 896, f"Expected 896d, got {train_features.shape[1]}d"
    
    return train_features, train_labels, val_features, val_labels


def normalize_features(train_features, val_features):
    """Normalize features using training set statistics"""
    print("\n" + "="*80)
    print("STEP 5: FEATURE NORMALIZATION")
    print("="*80)
    
    # Compute mean and std per feature dimension
    feature_mean = train_features.mean(axis=0, keepdims=True)
    feature_std = train_features.std(axis=0, keepdims=True)
    
    # Prevent division by zero
    feature_std = np.where(feature_std < 1e-8, 1.0, feature_std)
    
    print(f"\nNormalization statistics:")
    print(f"  Mean range: [{feature_mean.min():.4f}, {feature_mean.max():.4f}]")
    print(f"  Std range:  [{feature_std.min():.4f}, {feature_std.max():.4f}]")
    
    # Normalize
    train_features_norm = (train_features - feature_mean) / feature_std
    val_features_norm = (val_features - feature_mean) / feature_std
    
    print(f"\nNormalized training features:")
    print(f"  Mean: {train_features_norm.mean():.6f} (should be ~0)")
    print(f"  Std:  {train_features_norm.std():.6f} (should be ~1)")
    
    return train_features_norm, val_features_norm, feature_mean, feature_std


def save_features(train_features, train_labels, val_features, val_labels,
                 feature_mean, feature_std):
    """Save all features"""
    print("\n" + "="*80)
    print("STEP 6: SAVE FEATURES")
    print("="*80)
    
    print(f"\nSaving to: {FUSION_DIR}")
    
    # Save features
    np.save(FUSION_DIR / "train_features_896d.npy", train_features)
    np.save(FUSION_DIR / "train_labels.npy", train_labels)
    np.save(FUSION_DIR / "val_features_896d.npy", val_features)
    np.save(FUSION_DIR / "val_labels.npy", val_labels)
    
    # Save normalization statistics
    np.save(FUSION_DIR / "feature_mean.npy", feature_mean)
    np.save(FUSION_DIR / "feature_std.npy", feature_std)
    
    print(f"\n✓ Saved:")
    print(f"  - train_features_896d.npy ({train_features.shape})")
    print(f"  - val_features_896d.npy ({val_features.shape})")
    print(f"  - train_labels.npy")
    print(f"  - val_labels.npy")
    print(f"  - feature_mean.npy")
    print(f"  - feature_std.npy")


def main():
    """Main feature extraction pipeline"""
    
    print("\n🚀 Starting FIXED feature extraction pipeline...")
    print("   Using DUPLICATION for CNN-BiGRU (128→256)")
    print("   This ensures consistency with the detector!")
    start_time = time.time()
    
    try:
        # Extract features
        train_features, train_labels, val_features, val_labels = extract_all_features()
        
        # Normalize
        train_features_norm, val_features_norm, feature_mean, feature_std = \
            normalize_features(train_features, val_features)
        
        # Save
        save_features(train_features_norm, train_labels, val_features_norm,
                     val_labels, feature_mean, feature_std)
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*80)
        print("✅ FEATURE EXTRACTION COMPLETE!")
        print("="*80)
        print(f"Total time: {elapsed:.1f}s")
        print(f"\n📊 Summary:")
        print(f"  - 896-dimensional features (256 + 512 + 128)")
        print(f"  - CNN-BiGRU uses DUPLICATION (consistent with detector)")
        print(f"  - Training samples: {len(train_labels)}")
        print(f"  - Validation samples: {len(val_labels)}")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("\n1. Clean features (remove zero-variance):")
        print("   python3 create_cleaned_features.py")
        print("\n2. Retrain fusion model:")
        print("   python3 train_fusion.py")
        print("\n3. Test detector:")
        print("   python3 genre_detector.py Music/Rock/Paint_it_Black.mp3")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()