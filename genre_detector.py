"""
MUSIC GENRE DETECTOR - Production Version with Global Normalization
====================================================================
Detects genre of any music file using trained fusion model.

Usage:
    python3 genre_detector.py <audio_file>
    python3 genre_detector.py Music/Rock/Paint_it_Black.mp3
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import sys
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration - Auto-detect project directory
def get_base_dir():
    """Auto-detect project base directory"""
    # Try common locations
    possible_paths = [
        Path.home() / "music_genre_fusion",
        Path.home() / "Documents" / "music_genre_fusion",
        Path.cwd()
    ]
    
    for path in possible_paths:
        if (path / "models").exists() or (path / "data").exists():
            return path
    
    # Default to home directory location
    return Path.home() / "music_genre_fusion"

BASE_DIR = get_base_dir()
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
FUSION_DIR = DATA_DIR / "fusion_features"

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

SAMPLE_RATE = 22050
WINDOW_DURATION = 3  # Each window is 3 seconds (matches training)
MAX_DURATION = 30    # Analyze up to 30 seconds
N_MELS = 128
N_MFCC = 13
HOP_LENGTH = 512

device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")

# Global normalization parameters
NORM_PARAMS = None


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class CNNBiGRU_Reduced(nn.Module):
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
        features = torch.sum(x * attn_weights, dim=1)
        
        if return_features:
            return features
        
        return self.classifier(features)


class DPAModule(nn.Module):
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
        features = self.feature_proj(x)
        
        if return_features:
            return features
        
        return self.classifier(features)


class ECAModule(nn.Module):
    """Efficient Channel Attention - matches training architecture"""
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
    """ECAS-CNN - matches actual training architecture"""
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


class FusionClassifier(nn.Module):
    """Fusion classifier - matches actual training architecture"""
    def __init__(self, input_dim=711, num_classes=10):
        super(FusionClassifier, self).__init__()
        
        self.input_norm = nn.BatchNorm1d(input_dim, track_running_stats=False)
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 384)
        self.bn2 = nn.BatchNorm1d(384)
        self.drop2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(384, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.input_norm(x)
        
        x = F.relu(self.bn1(self.fc1(x)), inplace=True)
        x = self.drop1(x)
        
        x = F.relu(self.bn2(self.fc2(x)), inplace=True)
        x = self.drop2(x)
        
        x = F.relu(self.bn3(self.fc3(x)), inplace=True)
        x = self.drop3(x)
        
        x = self.fc4(x)
        return x


# ============================================================================
# NORMALIZATION
# ============================================================================

def load_normalization_params():
    """Load global normalization parameters from training"""
    global NORM_PARAMS
    
    if NORM_PARAMS is not None:
        return NORM_PARAMS
    
    NORM_PARAMS = {}
    
    # Load mel-spectrogram normalization
    mel_norm_path = DATA_DIR / "processed" / "melspectrogram_normalized" / "normalization_params.json"
    if mel_norm_path.exists():
        with open(mel_norm_path, 'r') as f:
            params = json.load(f)
        NORM_PARAMS['mel_mean'] = params['mean']
        NORM_PARAMS['mel_std'] = params['std']
        print(f"✓ Mel-spec norm: mean={params['mean']:.2f}, std={params['std']:.2f}")
    else:
        NORM_PARAMS['mel_mean'] = 0
        NORM_PARAMS['mel_std'] = 1
        print("⚠ Mel-spec normalization not found")
    
    # Load MFCC normalization (CRITICAL - was missing!)
    mfcc_norm_path = DATA_DIR / "processed" / "mfcc_normalized" / "normalization_params.json"
    if mfcc_norm_path.exists():
        with open(mfcc_norm_path, 'r') as f:
            params = json.load(f)
        NORM_PARAMS['mfcc_mean'] = params['mean']
        NORM_PARAMS['mfcc_std'] = params['std']
        print(f"✓ MFCC norm: mean={params['mean']:.2f}, std={params['std']:.2f}")
    else:
        NORM_PARAMS['mfcc_mean'] = 0
        NORM_PARAMS['mfcc_std'] = 1
        print("⚠ MFCC normalization not found")
    
    NORM_PARAMS['epsilon'] = 1e-8
    
    return NORM_PARAMS


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def load_full_audio(audio_file):
    """Load full audio file (up to MAX_DURATION seconds)"""
    try:
        audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE, duration=MAX_DURATION)
        return audio, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None


def process_audio_window(audio, sr, start_sample, window_samples):
    """Process a single 3-second window of audio"""
    norm_params = load_normalization_params()
    
    # Extract window
    end_sample = start_sample + window_samples
    window = audio[start_sample:end_sample]
    
    # Pad if needed
    if len(window) < window_samples:
        window = np.pad(window, (0, window_samples - len(window)), mode='constant')
    
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=window, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Apply global normalization
    if norm_params is not None:
        mel_spec_db = (mel_spec_db - norm_params['mel_mean']) / (norm_params['mel_std'] + norm_params['epsilon'])
    else:
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=window, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    )
    
    # Apply GLOBAL normalization for MFCC
    if norm_params is not None:
        mfcc = (mfcc - norm_params['mfcc_mean']) / (norm_params['mfcc_std'] + norm_params['epsilon'])
    else:
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-6)
    
    return mel_spec_db, mfcc


def load_and_process_audio(audio_file):
    """Load audio and extract features with global normalization - DEPRECATED, use load_full_audio"""
    try:
        audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE, duration=WINDOW_DURATION)
        
        target_length = SAMPLE_RATE * WINDOW_DURATION
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None
    
    norm_params = load_normalization_params()
    
    # Extract mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Apply global normalization if available
    if norm_params is not None:
        mel_spec_db = (mel_spec_db - norm_params['mel_mean']) / (norm_params['mel_std'] + norm_params['epsilon'])
    else:
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    )
    
    # Apply GLOBAL normalization for MFCC (must match training!)
    if norm_params is not None:
        mfcc = (mfcc - norm_params['mfcc_mean']) / (norm_params['mfcc_std'] + norm_params['epsilon'])
    else:
        # Fallback to per-feature normalization
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-6)
    
    return mel_spec_db, mfcc


def extract_fusion_features(mel_spec, mfcc, cnn_bigru, dpa, ecas, active_mask, feature_mean, feature_std):
    """Extract 896-dim features, normalize, and apply mask"""
    mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)
    mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        cnn_bigru.eval()
        dpa.eval()
        ecas.eval()
        
        feat1 = cnn_bigru(mel_tensor, return_features=True)
        if feat1.shape[1] == 128:
            feat1 = torch.cat([feat1, feat1], dim=1)
        
        feat2 = dpa(mel_tensor, return_features=True)
        feat3 = ecas(mfcc_tensor, return_features=True)
        
        # Concatenate to 896-dim
        fusion_features_896 = torch.cat([feat1, feat2, feat3], dim=1)
        
        # Z-score normalize using training statistics (CRITICAL!)
        feature_mean_tensor = torch.FloatTensor(feature_mean).to(device)
        feature_std_tensor = torch.FloatTensor(feature_std).to(device)
        fusion_features_896 = (fusion_features_896 - feature_mean_tensor) / (feature_std_tensor + 1e-8)
        
        # Apply mask to get final features
        fusion_features = fusion_features_896[:, active_mask]
    
    return fusion_features


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load all trained models"""
    # CNN-BiGRU - check multiple locations (it was sometimes saved to wrong folder)
    bigru_paths = [
        MODEL_DIR / "cnn_bigru" / "cnn_bigru_hyperdynamic_best.pth",
        MODEL_DIR / "dpa_cnn" / "cnn_bigru_hyperdynamic_best.pth",  # Bug: was saved here
        MODEL_DIR / "cnn_bigru" / "best_model.pth",
    ]
    
    bigru_path = None
    for path in bigru_paths:
        if path.exists():
            bigru_path = path
            break
    
    if bigru_path is None:
        print(f"Error: CNN-BiGRU model not found. Checked:")
        for path in bigru_paths:
            print(f"  - {path}")
        return None, None, None, None, None, None, None
    
    cnn_bigru = CNNBiGRU_Reduced(nc=10).to(device)
    checkpoint = torch.load(bigru_path, map_location=device, weights_only=False)
    cnn_bigru.load_state_dict(checkpoint['model_state_dict'])
    cnn_bigru.eval()
    print(f"✓ CNN-BiGRU loaded from {bigru_path.name}")
    
    # DPA-CNN
    dpa_paths = [
        MODEL_DIR / "dpa_cnn" / "dpa_cnn_best.pth",
        MODEL_DIR / "dpa_cnn" / "dpa_cnn_hyperdynamic_best.pth",
    ]
    
    dpa_path = None
    for path in dpa_paths:
        if path.exists():
            dpa_path = path
            break
    
    if dpa_path is None:
        print(f"Error: DPA-CNN model not found. Checked:")
        for path in dpa_paths:
            print(f"  - {path}")
        return None, None, None, None, None, None, None
    
    dpa = DPACNN_HyperDynamic(num_classes=10).to(device)
    checkpoint = torch.load(dpa_path, map_location=device, weights_only=False)
    dpa.load_state_dict(checkpoint['model_state_dict'])
    dpa.eval()
    print(f"✓ DPA-CNN loaded from {dpa_path.name}")
    
    # ECAS-CNN
    ecas_paths = [
        MODEL_DIR / "ecas_cnn" / "ecas_cnn_hyperdynamic_best.pth",
        MODEL_DIR / "ecas_cnn" / "best_model.pth",
    ]
    
    ecas_path = None
    for path in ecas_paths:
        if path.exists():
            ecas_path = path
            break
    
    if ecas_path is None:
        print(f"Error: ECAS-CNN model not found. Checked:")
        for path in ecas_paths:
            print(f"  - {path}")
        return None, None, None, None, None, None, None
    
    ecas = ECASCNN_HyperDynamic(nc=10).to(device)
    checkpoint = torch.load(ecas_path, map_location=device, weights_only=False)
    ecas.load_state_dict(checkpoint['model_state_dict'])
    ecas.eval()
    print(f"✓ ECAS-CNN loaded from {ecas_path.name}")
    
    # Fusion model
    fusion_paths = [
        MODEL_DIR / "fusion_progressive" / "best_model.pth",
        MODEL_DIR / "fusion" / "best_model.pth",
    ]
    
    fusion_path = None
    for path in fusion_paths:
        if path.exists():
            fusion_path = path
            break
    
    if fusion_path is None:
        print(f"Error: Fusion model not found. Checked:")
        for path in fusion_paths:
            print(f"  - {path}")
        return None, None, None, None, None, None, None
    
    checkpoint = torch.load(fusion_path, map_location=device, weights_only=False)
    input_dim = checkpoint['model_state_dict']['input_norm.weight'].shape[0]
    
    fusion = FusionClassifier(input_dim=input_dim, num_classes=10).to(device)
    fusion.load_state_dict(checkpoint['model_state_dict'])
    fusion.eval()
    print(f"✓ Fusion model loaded from {fusion_path.name} (input: {input_dim}d)")
    
    # Load active feature mask
    mask_paths = [
        FUSION_DIR / "active_feature_mask.npy",
        DATA_DIR / "fusion_features" / "active_feature_mask.npy",
        BASE_DIR / "active_feature_mask.npy",
    ]
    
    mask_path = None
    for path in mask_paths:
        if path.exists():
            mask_path = path
            break
    
    if mask_path is not None:
        active_mask = np.load(mask_path)
        print(f"✓ Feature mask loaded: {active_mask.sum()}/{len(active_mask)} active")
    else:
        if input_dim == 896:
            active_mask = np.ones(896, dtype=bool)
            print(f"⚠ Feature mask not found, using all 896 features")
        else:
            print(f"Error: Feature mask not found and model expects {input_dim} dims")
            print(f"Checked:")
            for path in mask_paths:
                print(f"  - {path}")
            return None, None, None, None, None, None, None
    
    # Load feature normalization params (CRITICAL for correct inference!)
    feature_mean_paths = [
        FUSION_DIR / "feature_mean.npy",
        DATA_DIR / "fusion_features" / "feature_mean.npy",
    ]
    feature_std_paths = [
        FUSION_DIR / "feature_std.npy",
        DATA_DIR / "fusion_features" / "feature_std.npy",
    ]
    
    feature_mean = None
    feature_std = None
    
    for path in feature_mean_paths:
        if path.exists():
            feature_mean = np.load(path)
            break
    
    for path in feature_std_paths:
        if path.exists():
            feature_std = np.load(path)
            break
    
    if feature_mean is not None and feature_std is not None:
        print(f"✓ Feature normalization loaded (896d)")
    else:
        print(f"⚠ Feature normalization not found - predictions may be wrong!")
        feature_mean = np.zeros(896)
        feature_std = np.ones(896)
    
    return cnn_bigru, dpa, ecas, fusion, active_mask, feature_mean, feature_std


# ============================================================================
# DETECTION
# ============================================================================

def detect_genre(audio_file, models, top_k=3):
    """Detect genre by analyzing multiple 3-second windows and averaging predictions"""
    cnn_bigru, dpa, ecas, fusion, active_mask, feature_mean, feature_std = models
    
    audio_path = Path(audio_file)
    if not audio_path.is_absolute():
        if not audio_path.exists():
            audio_path_from_base = BASE_DIR / audio_path
            if audio_path_from_base.exists():
                audio_path = audio_path_from_base
            else:
                audio_path = Path(audio_file).resolve()
    
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        return None
    
    # Load full audio (up to 30 seconds)
    audio, sr = load_full_audio(str(audio_path))
    if audio is None:
        return None
    
    audio_duration = len(audio) / sr
    window_samples = SAMPLE_RATE * WINDOW_DURATION
    
    # Use 80% overlap for better coverage
    hop_samples = window_samples // 5  # 0.6 second hop (80% overlap)
    num_windows = max(1, (len(audio) - window_samples) // hop_samples + 1)
    
    print(f"Analyzing {audio_duration:.1f}s audio in {num_windows} windows (80% overlap)...")
    
    all_probs = []
    
    for i in range(num_windows):
        start_sample = i * hop_samples
        
        # Skip if not enough samples for full window
        if start_sample + window_samples > len(audio):
            break
        
        # Process this window
        mel_spec, mfcc = process_audio_window(audio, sr, start_sample, window_samples)
        
        # Extract features
        fusion_features = extract_fusion_features(
            mel_spec, mfcc, cnn_bigru, dpa, ecas, active_mask, feature_mean, feature_std
        )
        
        # Get predictions for this window
        with torch.no_grad():
            fusion.eval()
            
            # Manual normalization for input_norm (track_running_stats=False fix)
            input_weight = fusion.input_norm.weight.data
            input_bias = fusion.input_norm.bias.data
            
            mean = fusion_features.mean(dim=1, keepdim=True)
            std = fusion_features.std(dim=1, keepdim=True) + 1e-5
            x = (fusion_features - mean) / std
            x = x * input_weight + input_bias
            
            # Pass through rest of network
            x = F.relu(fusion.bn1(fusion.fc1(x)))
            x = fusion.drop1(x)
            x = F.relu(fusion.bn2(fusion.fc2(x)))
            x = fusion.drop2(x)
            x = F.relu(fusion.bn3(fusion.fc3(x)))
            x = fusion.drop3(x)
            outputs = fusion.fc4(x)
            
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    
    # Average probabilities across all windows
    avg_probs = np.mean(all_probs, axis=0).flatten()
    
    # Get top-k predictions
    top_indices = np.argsort(avg_probs)[::-1][:top_k]
    
    predictions = []
    for idx in top_indices:
        predictions.append((GENRES[idx], avg_probs[idx] * 100))
    
    return predictions


def print_results(predictions, audio_file):
    """Print detection results"""
    print("\n" + "="*70)
    print("MUSIC GENRE DETECTION RESULTS")
    print("="*70)
    print(f"File: {Path(audio_file).name}")
    print("-"*70)
    
    if predictions is None:
        print("Detection failed")
        return
    
    print(f"\nPredicted Genre: {predictions[0][0].upper()} ({predictions[0][1]:.1f}%)")
    
    print(f"\nTop {len(predictions)} Predictions:")
    for i, (genre, prob) in enumerate(predictions, 1):
        bar_length = int(prob / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {i}. {genre:10s} {bar} {prob:5.1f}%")
    
    print("="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 genre_detector.py <audio_file>")
        print("\nExamples:")
        print("  python3 genre_detector.py Music/Rock/Paint_it_Black.mp3")
        print("  python3 genre_detector.py ~/Downloads/song.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    print(f"Project directory: {BASE_DIR}")
    print("Loading models...")
    load_normalization_params()
    models = load_models()
    
    if models[0] is None:
        print("\nFailed to load models. Please check:")
        print(f"  Model directory: {MODEL_DIR}")
        print(f"  Models should be in:")
        print(f"    - {MODEL_DIR}/cnn_bigru/cnn_bigru_hyperdynamic_best.pth")
        print(f"    - {MODEL_DIR}/dpa_cnn/dpa_cnn_best.pth")
        print(f"    - {MODEL_DIR}/ecas_cnn/ecas_cnn_hyperdynamic_best.pth")
        print(f"    - {MODEL_DIR}/fusion_progressive/best_model.pth")
        sys.exit(1)
    
    print("Detecting genre...")
    predictions = detect_genre(audio_file, models, top_k=3)
    print_results(predictions, audio_file)


if __name__ == "__main__":
    main()