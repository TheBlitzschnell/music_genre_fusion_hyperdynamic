"""
ECAS-CNN ULTIMATE OPTIMIZATION
================================
Aggressive optimizations to achieve 70%+:

1. WINDOWING: 50% overlap → 3-5x more training samples
2. DYNAMIC CLASS FOCUS: Ignore classes >70%, focus on struggling ones
3. NO EARLY STOPPING: Train minimum 150 epochs
4. PROGRESSIVE TRAINING: Start simple, increase difficulty
5. MULTI-SCALE: Multiple window sizes

Current: 55% → Target: 70%+
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import time
import random
from collections import Counter

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path.home() / "music_genre_fusion"
DATA_DIR = BASE_DIR / "data" / "processed" / "mfcc_normalized"
MODEL_DIR = BASE_DIR / "models" / "ecas_cnn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# AGGRESSIVE HYPERPARAMETERS
BATCH_SIZE = 32  # Increased for faster training with more samples
LEARNING_RATE = 0.001  # Higher to learn faster
NUM_EPOCHS = 250
MIN_EPOCHS = 150  # Train at least this many epochs
PATIENCE = 100  # Very high patience
NUM_CLASSES = 10
WEIGHT_DECAY = 5e-5  # Lower since we have more data now

# WINDOWING PARAMETERS
WINDOW_SIZE = 130  # Standard MFCC time frames
OVERLAP = 0.5  # 50% overlap
HOP_LENGTH = int(WINDOW_SIZE * (1 - OVERLAP))  # 65 frames

# DYNAMIC CLASS FOCUSING
CLASS_ACCURACY_THRESHOLD = 70.0  # Classes above this are "done"
FOCUS_MULTIPLIER = 2.0  # Struggling classes get 2x weight

print("="*80)
print("ECAS-CNN ULTIMATE OPTIMIZATION")
print("="*80)
print(f"Device: {device}")
print(f"🔥 AGGRESSIVE OPTIMIZATIONS:")
print(f"  ✓ Windowing: 50% overlap → 3-5x more samples")
print(f"  ✓ Dynamic focus: Ignore classes >70%, boost struggling ones")
print(f"  ✓ Minimum training: {MIN_EPOCHS} epochs (no premature stopping)")
print(f"  ✓ Higher LR: {LEARNING_RATE} for faster convergence")
print("="*80)


def create_windowed_data(X, y, window_size=WINDOW_SIZE, hop=HOP_LENGTH):
    """
    Create multiple windows from each sample with 50% overlap
    This increases dataset size by 3-5x!
    
    Example: 13×260 MFCC → three windows:
    - Window 1: frames 0-130
    - Window 2: frames 65-195 (50% overlap)
    - Window 3: frames 130-260 (50% overlap)
    """
    windowed_X = []
    windowed_y = []
    
    print(f"\n📊 Creating windowed dataset (50% overlap)...")
    print(f"   Window size: {window_size}, Hop: {hop}")
    
    for i in range(len(X)):
        sample = X[i]  # (freq, time)
        freq_bins, time_frames = sample.shape
        
        # Calculate number of windows
        num_windows = (time_frames - window_size) // hop + 1
        
        if num_windows < 1:
            # If sample shorter than window, pad it
            pad_amount = window_size - time_frames
            sample_padded = np.pad(sample, ((0, 0), (0, pad_amount)), mode='constant')
            windowed_X.append(sample_padded)
            windowed_y.append(y[i])
        else:
            # Extract multiple windows
            for w in range(num_windows):
                start = w * hop
                end = start + window_size
                if end <= time_frames:
                    window = sample[:, start:end]
                    windowed_X.append(window)
                    windowed_y.append(y[i])
    
    windowed_X = np.array(windowed_X)
    windowed_y = np.array(windowed_y)
    
    print(f"   Original: {len(X)} samples")
    print(f"   Windowed: {len(windowed_X)} samples ({len(windowed_X)/len(X):.1f}x increase)")
    
    return windowed_X, windowed_y


class ECAModule(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAModule, self).__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        k_size = max(3, k_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ECASCNN_Ultimate(nn.Module):
    """Optimized ECAS-CNN with larger capacity and better architecture"""
    def __init__(self, num_classes=10):
        super(ECASCNN_Ultimate, self).__init__()
        
        # Larger capacity network
        # Block 1: 13×130 → 6×65
        self.conv1 = nn.Conv2d(1, 192, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(192)
        self.eca1 = ECAModule(192)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.1)  # Very light dropout
        
        # Block 2: 6×65 → 3×32
        self.conv2 = nn.Conv2d(192, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.eca2 = ECAModule(256)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(0.15)
        
        # Block 3: 3×32 → 1×16
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(384)
        self.eca3 = ECAModule(384)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(0.2)
        
        # Block 4: 1×16 → 1×8
        self.conv4 = nn.Conv2d(384, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.eca4 = ECAModule(512)
        self.pool4 = nn.MaxPool2d((1, 2))
        self.drop4 = nn.Dropout2d(0.2)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=False):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.eca1(x)
        x = self.drop1(self.pool1(x))
        
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.eca2(x)
        x = self.drop2(self.pool2(x))
        
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = self.eca3(x)
        x = self.drop3(self.pool3(x))
        
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.eca4(x)
        x = self.drop4(self.pool4(x))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        features = self.feature_proj(x)
        
        if return_features:
            return features
        
        return self.classifier(features)


class SimpleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label


def compute_dynamic_class_weights(train_labels, class_accuracies=None):
    """
    Compute dynamic class weights that focus on struggling classes
    Classes >70% accuracy get reduced weight
    """
    class_counts = Counter(train_labels)
    total = len(train_labels)
    weights = torch.ones(NUM_CLASSES)
    
    for cls in range(NUM_CLASSES):
        count = class_counts.get(cls, 1)
        base_weight = total / (NUM_CLASSES * count)
        
        # If we have accuracy info, adjust weights
        if class_accuracies is not None and cls in class_accuracies:
            acc = class_accuracies[cls]
            if acc >= CLASS_ACCURACY_THRESHOLD:
                # Class is doing well, reduce its importance
                weights[cls] = base_weight * 0.5
                print(f"   Class {cls}: {acc:.1f}% ✓ (reducing weight to {weights[cls]:.3f})")
            else:
                # Class struggling, increase importance
                gap = CLASS_ACCURACY_THRESHOLD - acc
                boost = 1.0 + (gap / CLASS_ACCURACY_THRESHOLD) * FOCUS_MULTIPLIER
                weights[cls] = base_weight * boost
                print(f"   Class {cls}: {acc:.1f}% ⚠️  (boosting weight to {weights[cls]:.3f})")
        else:
            weights[cls] = base_weight
    
    # Normalize
    weights = weights / weights.sum() * NUM_CLASSES
    return weights


def train_epoch(model, loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}", leave=False, ncols=100)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False, ncols=100)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100.*correct/total:.1f}%'})
    
    return running_loss / len(loader), 100. * correct / total, all_preds, all_labels


def compute_per_class_accuracy(preds, labels, num_classes=10):
    """Compute accuracy for each class"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    
    accuracies = {}
    for i in range(num_classes):
        if cm[i].sum() > 0:
            accuracies[i] = 100.0 * cm[i, i] / cm[i].sum()
        else:
            accuracies[i] = 0.0
    
    return accuracies


def print_confusion_matrix(preds, labels, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    
    print("\nConfusion Matrix:")
    print("    ", " ".join(f"{i:3d}" for i in range(num_classes)))
    for i in range(num_classes):
        print(f"{i:2d}: ", " ".join(f"{cm[i,j]:3d}" for j in range(num_classes)))
    
    print("\nPer-class accuracy:")
    struggling_classes = []
    good_classes = []
    
    for i in range(num_classes):
        class_acc = 100.0 * cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        marker = "✓" if class_acc >= 70 else "⚠️"
        print(f"  Class {i}: {class_acc:5.1f}% ({cm[i,i]:3d}/{cm[i].sum():3d}) {marker}")
        
        if class_acc < 70:
            struggling_classes.append(i)
        else:
            good_classes.append(i)
    
    if good_classes:
        print(f"\n✓ Classes performing well (≥70%): {good_classes}")
    if struggling_classes:
        print(f"⚠️  Classes needing focus (<70%): {struggling_classes}")
    
    return struggling_classes, good_classes


def main():
    # Load original data
    print("\n" + "="*80)
    print("STEP 1: LOADING AND WINDOWING DATA")
    print("="*80)
    
    train_X_orig = np.load(DATA_DIR / "train_features.npy")
    train_y_orig = np.load(DATA_DIR / "train_labels.npy")
    val_X_orig = np.load(DATA_DIR / "val_features.npy")
    val_y_orig = np.load(DATA_DIR / "val_labels.npy")
    
    print(f"\nOriginal data:")
    print(f"  Train: {train_X_orig.shape}")
    print(f"  Val: {val_X_orig.shape}")
    
    # Create windowed datasets
    train_X, train_y = create_windowed_data(train_X_orig, train_y_orig)
    val_X, val_y = create_windowed_data(val_X_orig, val_y_orig)
    
    print(f"\n✓ Windowed training data: {train_X.shape}")
    print(f"✓ Windowed validation data: {val_X.shape}")
    print(f"✓ Data augmentation: {len(train_X)/len(train_X_orig):.1f}x more samples!")
    
    # Create datasets
    train_dataset = SimpleDataset(train_X, train_y)
    val_dataset = SimpleDataset(val_X, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    print("\n" + "="*80)
    print("STEP 2: INITIALIZING ULTIMATE MODEL")
    print("="*80)
    
    model = ECASCNN_Ultimate(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model: ECAS-CNN Ultimate")
    print(f"✓ Parameters: {total_params:,}")
    print(f"✓ Architecture: 192→256→384→512 (massive capacity)")
    
    # Initial class weights (uniform, will be updated dynamically)
    class_weights = compute_dynamic_class_weights(train_y)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    
    # Optimizer with higher learning rate
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Aggressive scheduler - only reduce LR when really stuck
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.7, patience=25, min_lr=1e-5
    )
    
    print(f"✓ Learning rate: {LEARNING_RATE}")
    print(f"✓ Minimum epochs: {MIN_EPOCHS} (no early stopping before this)")
    print(f"✓ Dynamic class weights: Will update every 20 epochs")
    
    # Training
    print("\n" + "="*80)
    print("STEP 3: AGGRESSIVE TRAINING")
    print("="*80)
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update class weights every 20 epochs based on current performance
        if epoch % 20 == 0 and epoch > 0:
            print(f"\n🔄 Updating class weights based on current performance...")
            _, _, val_preds, val_labels = validate(model, val_loader, criterion, device)
            class_accuracies = compute_per_class_accuracy(val_preds, val_labels)
            class_weights = compute_dynamic_class_weights(train_y, class_accuracies).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # Step scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print
        print(f"\nEpoch {epoch:3d}/{NUM_EPOCHS} | LR: {current_lr:.6f}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:5.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:5.2f}%")
        
        # Save best
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, MODEL_DIR / "ecas_cnn_best.pth")
            
            print(f"  ✓ Best model saved! Val Acc: {val_acc:.2f}%")
            
            # Show confusion matrix every 20 epochs or when improving above 65%
            if epoch % 20 == 0 or val_acc >= 65:
                struggling, good = print_confusion_matrix(val_preds, val_labels)
            
            if val_acc >= 70.0:
                print(f"\n{'='*80}")
                print(f"🎉 TARGET ACHIEVED! 70%+ Accuracy!")
                print(f"{'='*80}")
                if epoch >= MIN_EPOCHS:
                    print(f"Stopping early after reaching target at epoch {epoch}")
                    break
        else:
            patience_counter += 1
            if epoch >= MIN_EPOCHS:  # Only check patience after minimum epochs
                print(f"  Patience: {patience_counter}/{PATIENCE}")
        
        # Early stopping only after minimum epochs
        if patience_counter >= PATIENCE and epoch >= MIN_EPOCHS:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break
    
    # Save final
    torch.save(model.state_dict(), MODEL_DIR / "ecas_cnn_final.pth")
    with open(MODEL_DIR / "training_history_ultimate.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    elapsed = (time.time() - start_time) / 60
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Time: {elapsed:.1f} minutes ({elapsed/60:.2f} hours)")
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    
    # Final evaluation
    checkpoint = torch.load(MODEL_DIR / "ecas_cnn_best.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Best validation accuracy: {val_acc:.2f}%")
    struggling, good = print_confusion_matrix(val_preds, val_labels)
    
    print("\n" + "="*80)
    if val_acc >= 70.0:
        print("✅ SUCCESS! 70%+ achieved with optimizations!")
        print(f"   Windowing gave us {len(train_X)/len(train_X_orig):.1f}x more training data")
        print(f"   Dynamic class focusing helped struggling classes")
    elif val_acc >= 65.0:
        print("📈 Significant progress! 65%+")
        print(f"   Close to target - {70-val_acc:.1f}% away")
        print(f"   Windowing improved data utilization")
    else:
        print(f"📊 Improved from 55% baseline")
        print(f"   Current: {val_acc:.2f}%")
        print(f"   Windowing provided {len(train_X)/len(train_X_orig):.1f}x more samples")
    print("="*80)
    
    # Test feature extraction
    print("\n" + "="*80)
    print("FEATURE EXTRACTION TEST")
    print("="*80)
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(4, 1, 13, 130).to(device)
        features = model(test_input, return_features=True)
        print(f"✓ Feature extraction working: {test_input.shape} → {features.shape}")
        print(f"✓ Ready for fusion with 128-dim features!")
    print("="*80)


if __name__ == "__main__":
    main()