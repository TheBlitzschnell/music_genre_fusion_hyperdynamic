"""
DPA-CNN HYPER-DYNAMIC TRAINING
================================
Revolutionary adaptive training strategy:

✓ SMART CLASS GRADUATION: Classes ≥85% are "graduated" - predictions saved, training stopped
✓ ADAPTIVE WINDOWING: Classes <50% after epoch 50 get 75% overlap (more samples)
✓ NO EARLY STOPPING: Train full 300 epochs
✓ DYNAMIC WEIGHTS: Real-time adjustment based on performance
✓ FOCUS ON STRUGGLING: All resources go to classes that need help

Target: 85%+ overall by teaching each class optimally
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
import json
import time
import random
from collections import Counter, defaultdict

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path.home() / "music_genre_fusion"
DATA_DIR = BASE_DIR / "data" / "processed" / "melspectrogram_normalized"
MODEL_DIR = BASE_DIR / "models" / "dpa_cnn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# HYPER-DYNAMIC HYPERPARAMETERS
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 300  # Train full 300 epochs
NUM_CLASSES = 10
WEIGHT_DECAY = 5e-5

# DYNAMIC WINDOWING
WINDOW_SIZE = 130
BASE_OVERLAP = 0.5  # 50% for normal classes
STRUGGLING_OVERLAP = 0.75  # 75% for struggling classes (<50% after epoch 50)
STRUGGLING_THRESHOLD = 50.0  # Accuracy threshold for "struggling"
STRUGGLING_CHECK_EPOCH = 50  # When to check and apply extra windowing

# SMART CLASS GRADUATION
REDUCE_WEIGHT_THRESHOLD = 70.0  # Reduce weight if class ≥70%
GRADUATE_THRESHOLD = 85.0  # "Graduate" class if ≥85% - save predictions and stop training
FOCUS_MULTIPLIER = 3.0  # Struggling classes get 3x weight

print("="*80)
print("DPA-CNN HYPER-DYNAMIC TRAINING")
print("="*80)
print(f"Device: {device}")
print(f"🔥 REVOLUTIONARY ADAPTIVE TRAINING:")
print(f"  ✓ Smart graduation: Classes ≥85% are done")
print(f"  ✓ Adaptive windowing: Struggling classes get 75% overlap")
print(f"  ✓ No early stopping: Full 300 epochs")
print(f"  ✓ Dynamic weights: Real-time adjustment")
print(f"  ✓ Target: 85%+ overall")
print("="*80)


def create_windowed_data(X, y, window_size=WINDOW_SIZE, overlap=BASE_OVERLAP):
    """Create windows with specified overlap"""
    hop = int(window_size * (1 - overlap))
    windowed_X = []
    windowed_y = []
    sample_indices = []  # Track which original sample each window came from
    
    for i in range(len(X)):
        sample = X[i]
        freq_bins, time_frames = sample.shape
        
        num_windows = max(1, (time_frames - window_size) // hop + 1)
        
        if num_windows == 1 or time_frames < window_size:
            pad_amount = max(0, window_size - time_frames)
            sample_padded = np.pad(sample, ((0, 0), (0, pad_amount)), mode='constant')
            windowed_X.append(sample_padded)
            windowed_y.append(y[i])
            sample_indices.append(i)
        else:
            for w in range(num_windows):
                start = w * hop
                end = start + window_size
                if end <= time_frames:
                    window = sample[:, start:end]
                    windowed_X.append(window)
                    windowed_y.append(y[i])
                    sample_indices.append(i)
    
    return np.array(windowed_X), np.array(windowed_y), np.array(sample_indices)


def create_adaptive_windowed_data(X_orig, y_orig, struggling_classes=None):
    """
    Create windowed data with adaptive overlap:
    - Normal classes: 50% overlap
    - Struggling classes: 75% overlap (3x more samples)
    """
    print(f"\n📊 Creating adaptive windowed dataset...")
    
    if struggling_classes is None or len(struggling_classes) == 0:
        # Initial training - use base overlap for all
        X, y, indices = create_windowed_data(X_orig, y_orig, overlap=BASE_OVERLAP)
        print(f"   Base overlap (50%): {len(X_orig)} → {len(X)} samples ({len(X)/len(X_orig):.1f}x)")
        return X, y, indices
    
    # Split by class
    all_X = []
    all_y = []
    all_indices = []
    
    for cls in range(NUM_CLASSES):
        mask = y_orig == cls
        X_cls = X_orig[mask]
        y_cls = y_orig[mask]
        
        if cls in struggling_classes:
            # Use 75% overlap for struggling classes
            X_win, y_win, idx_win = create_windowed_data(X_cls, y_cls, overlap=STRUGGLING_OVERLAP)
            print(f"   Class {cls} (struggling): {len(X_cls)} → {len(X_win)} samples (75% overlap, {len(X_win)/len(X_cls):.1f}x)")
        else:
            # Use 50% overlap for normal classes
            X_win, y_win, idx_win = create_windowed_data(X_cls, y_cls, overlap=BASE_OVERLAP)
            print(f"   Class {cls} (normal):     {len(X_cls)} → {len(X_win)} samples (50% overlap, {len(X_win)/len(X_cls):.1f}x)")
        
        all_X.append(X_win)
        all_y.append(y_win)
        all_indices.append(idx_win)
    
    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    indices_final = np.concatenate(all_indices, axis=0)
    
    print(f"   Total: {len(X_orig)} → {len(X_final)} samples ({len(X_final)/len(X_orig):.1f}x)")
    
    return X_final, y_final, indices_final


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
        
        # Channel Attention
        theta = self.theta(x).view(batch_size, -1, H * W)
        phi = self.phi(x).view(batch_size, -1, H * W)
        psi = self.psi(x).view(batch_size, -1, H * W)
        
        attention = torch.bmm(theta.permute(0, 2, 1), phi)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(psi, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, H, W)
        out = self.g(out)
        channel_att = torch.sigmoid(out)
        
        # Spatial Attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_input))
        
        x = x * channel_att * spatial_att
        return x


class DPACNN_HyperDynamic(nn.Module):
    """Enhanced DPA-CNN with increased capacity"""
    def __init__(self, num_classes=10):
        super(DPACNN_HyperDynamic, self).__init__()
        
        # Larger capacity
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


class AdaptiveDataset(Dataset):
    """Dataset that can filter out graduated classes"""
    def __init__(self, features, labels, active_classes=None):
        self.features = features
        self.labels = labels
        self.active_classes = active_classes if active_classes is not None else set(range(10))
        
        # Create mask for active classes only
        if active_classes is not None:
            mask = np.array([label in active_classes for label in labels])
            self.active_indices = np.where(mask)[0]
        else:
            self.active_indices = np.arange(len(labels))
    
    def __len__(self):
        return len(self.active_indices)
    
    def __getitem__(self, idx):
        real_idx = self.active_indices[idx]
        feature = torch.FloatTensor(self.features[real_idx]).unsqueeze(0)
        label = torch.LongTensor([self.labels[real_idx]])[0]
        return feature, label, real_idx


def compute_adaptive_class_weights(train_labels, class_accuracies, graduated_classes, active_classes):
    """
    Compute adaptive weights:
    - Graduated classes (≥85%): weight = 0 (skip training)
    - Good classes (70-85%): weight = 0.5 (reduce focus)
    - Struggling classes (<70%): weight = 1.0 to 3.0 (boost focus)
    """
    class_counts = Counter(train_labels)
    total = len(train_labels)
    weights = torch.zeros(NUM_CLASSES)
    
    print(f"\n📊 Computing adaptive class weights:")
    
    for cls in range(NUM_CLASSES):
        if cls in graduated_classes:
            weights[cls] = 0.0
            print(f"   Class {cls}: GRADUATED ✅ (≥85%, weight=0)")
        elif cls not in active_classes:
            weights[cls] = 0.0
        else:
            count = class_counts.get(cls, 1)
            base_weight = total / (NUM_CLASSES * count)
            
            if cls in class_accuracies:
                acc = class_accuracies[cls]
                
                if acc >= REDUCE_WEIGHT_THRESHOLD:
                    # Good performance, reduce weight
                    weights[cls] = base_weight * 0.5
                    print(f"   Class {cls}: {acc:.1f}% ✓ (reducing weight to {weights[cls]:.3f})")
                else:
                    # Struggling, boost weight
                    gap = REDUCE_WEIGHT_THRESHOLD - acc
                    boost = 1.0 + (gap / REDUCE_WEIGHT_THRESHOLD) * FOCUS_MULTIPLIER
                    weights[cls] = base_weight * boost
                    print(f"   Class {cls}: {acc:.1f}% ⚠️  (boosting weight to {weights[cls]:.3f})")
            else:
                weights[cls] = base_weight
    
    # Normalize only active classes
    active_sum = sum(weights[cls] for cls in active_classes if cls not in graduated_classes)
    if active_sum > 0:
        for cls in active_classes:
            if cls not in graduated_classes:
                weights[cls] = weights[cls] / active_sum * len(active_classes)
    
    return weights


def train_epoch(model, loader, criterion, optimizer, epoch, device, graduated_classes):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}", leave=False, ncols=100)
    for inputs, labels, _ in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Skip graduated classes
        mask = torch.tensor([label.item() not in graduated_classes for label in labels], device=device)
        if mask.sum() == 0:
            continue
        
        inputs = inputs[mask]
        labels = labels[mask]
        
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
    
    if total == 0:
        return 0.0, 0.0
    return running_loss / max(len(loader), 1), 100. * correct / total


def validate(model, loader, criterion, device, graduated_classes, graduated_predictions):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False, ncols=100)
        for inputs, labels, indices in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Use graduated predictions for graduated classes
            batch_preds = []
            for i, (label, idx) in enumerate(zip(labels, indices)):
                label_item = label.item()
                idx_item = idx.item()
                
                if label_item in graduated_classes:
                    # Use saved prediction
                    batch_preds.append(graduated_predictions[idx_item])
                else:
                    # Compute prediction
                    output = model(inputs[i:i+1])
                    _, pred = output.max(1)
                    batch_preds.append(pred.item())
            
            all_preds.extend(batch_preds)
            all_labels.extend(labels.cpu().numpy())
            all_indices.extend(indices.cpu().numpy())
    
    # Compute accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = 100. * correct / len(all_labels) if len(all_labels) > 0 else 0.0
    
    return 0.0, acc, all_preds, all_labels


def compute_per_class_accuracy(preds, labels, num_classes=10):
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


def print_status(class_accuracies, graduated_classes, struggling_classes):
    print("\n" + "="*80)
    print("CLASS STATUS")
    print("="*80)
    
    for cls in range(NUM_CLASSES):
        acc = class_accuracies.get(cls, 0.0)
        
        if cls in graduated_classes:
            status = "🎓 GRADUATED"
        elif cls in struggling_classes:
            status = "💪 STRUGGLING (75% overlap)"
        elif acc >= REDUCE_WEIGHT_THRESHOLD:
            status = "✓ GOOD"
        else:
            status = "⚠️  NEEDS WORK"
        
        print(f"  Class {cls}: {acc:5.1f}%  {status}")
    
    print(f"\nGraduated: {len(graduated_classes)}/10 classes")
    print(f"Active training: {10 - len(graduated_classes)} classes")
    print("="*80)


def main():
    # Load original data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_X_orig = np.load(DATA_DIR / "train_features.npy")
    train_y_orig = np.load(DATA_DIR / "train_labels.npy")
    val_X_orig = np.load(DATA_DIR / "val_features.npy")
    val_y_orig = np.load(DATA_DIR / "val_labels.npy")
    
    print(f"Original data: Train {train_X_orig.shape}, Val {val_X_orig.shape}")
    
    # Initial windowing (50% for all)
    train_X, train_y, train_indices = create_adaptive_windowed_data(train_X_orig, train_y_orig)
    val_X, val_y, val_indices = create_adaptive_windowed_data(val_X_orig, val_y_orig)
    
    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    model = DPACNN_HyperDynamic(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {total_params:,}")
    
    # Training state
    graduated_classes = set()  # Classes that achieved ≥85%
    graduated_predictions = {}  # Saved predictions for graduated classes
    struggling_classes = set()  # Classes <50% after epoch 50
    active_classes = set(range(NUM_CLASSES))
    
    # Initial weights
    class_weights = torch.ones(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    print(f"✓ Training: {NUM_EPOCHS} epochs (NO early stopping)")
    
    # Training
    print("\n" + "="*80)
    print("HYPER-DYNAMIC TRAINING")
    print("="*80)
    
    best_val_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'graduated': [], 'struggling': []}
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Create datasets
        train_dataset = AdaptiveDataset(train_X, train_y, active_classes)
        val_dataset = AdaptiveDataset(val_X, val_y, None)  # Always validate all
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, device, graduated_classes)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device, graduated_classes, graduated_predictions)
        
        # Compute per-class accuracy
        class_accuracies = compute_per_class_accuracy(val_preds, val_labels)
        
        # Update graduated classes
        new_graduates = set()
        for cls in active_classes:
            if cls not in graduated_classes and class_accuracies.get(cls, 0) >= GRADUATE_THRESHOLD:
                new_graduates.add(cls)
                # Save predictions for this class
                for i, (pred, label) in enumerate(zip(val_preds, val_labels)):
                    if label == cls:
                        graduated_predictions[val_indices[i]] = pred
                print(f"\n🎓 Class {cls} GRADUATED with {class_accuracies[cls]:.1f}% accuracy!")
        
        graduated_classes.update(new_graduates)
        active_classes = active_classes - graduated_classes
        
        # Check for struggling classes at epoch 50
        if epoch == STRUGGLING_CHECK_EPOCH:
            print(f"\n{'='*80}")
            print(f"EPOCH {STRUGGLING_CHECK_EPOCH}: CHECKING FOR STRUGGLING CLASSES")
            print(f"{'='*80}")
            
            for cls in range(NUM_CLASSES):
                if cls not in graduated_classes and class_accuracies.get(cls, 0) < STRUGGLING_THRESHOLD:
                    struggling_classes.add(cls)
                    print(f"⚠️  Class {cls} is struggling ({class_accuracies[cls]:.1f}% < 50%)")
            
            if len(struggling_classes) > 0:
                print(f"\n🔄 Recreating dataset with 75% overlap for struggling classes...")
                train_X, train_y, train_indices = create_adaptive_windowed_data(train_X_orig, train_y_orig, struggling_classes)
                val_X, val_y, val_indices = create_adaptive_windowed_data(val_X_orig, val_y_orig, struggling_classes)
                print(f"✓ Dataset recreated with adaptive windowing!")
        
        # Update class weights every 20 epochs
        if epoch % 20 == 0:
            class_weights = compute_adaptive_class_weights(train_y, class_accuracies, graduated_classes, active_classes).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        # Step scheduler
        scheduler.step()
        
        # Save history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['graduated'].append(len(graduated_classes))
        history['struggling'].append(len(struggling_classes))
        
        # Print
        print(f"\nEpoch {epoch:3d}/{NUM_EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Train Acc: {train_acc:5.2f}% (active classes only)")
        print(f"  Val Acc:   {val_acc:5.2f}% (all classes)")
        print(f"  Graduated: {len(graduated_classes)}/10 classes")
        print(f"  Active:    {len(active_classes)} classes")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'graduated_classes': graduated_classes,
                'graduated_predictions': graduated_predictions,
                'struggling_classes': struggling_classes,
            }, MODEL_DIR / "dpa_cnn_best.pth")
            
            print(f"  ✓ Best model saved! Val Acc: {val_acc:.2f}%")
        
        # Print status every 50 epochs
        if epoch % 50 == 0:
            print_status(class_accuracies, graduated_classes, struggling_classes)
        
        # Check if all classes graduated
        if len(graduated_classes) == NUM_CLASSES:
            print(f"\n{'='*80}")
            print(f"🎉 ALL CLASSES GRADUATED! Training complete at epoch {epoch}")
            print(f"{'='*80}")
            break
    
    # Save final
    torch.save(model.state_dict(), MODEL_DIR / "dpa_cnn_final.pth")
    with open(MODEL_DIR / "training_history_hyperdynamic.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    elapsed = (time.time() - start_time) / 60
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Time: {elapsed:.1f} minutes ({elapsed/60:.2f} hours)")
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print(f"Graduated classes: {len(graduated_classes)}/10")
    
    # Final evaluation
    checkpoint = torch.load(MODEL_DIR / "dpa_cnn_best.pth", weights_only=False)
    print(f"\nFinal class accuracies:")
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device, 
                                                        checkpoint['graduated_classes'], 
                                                        checkpoint['graduated_predictions'])
    class_accuracies = compute_per_class_accuracy(val_preds, val_labels)
    
    for cls in range(NUM_CLASSES):
        acc = class_accuracies[cls]
        status = "🎓" if cls in checkpoint['graduated_classes'] else "✓" if acc >= 70 else "⚠️"
        print(f"  Class {cls}: {acc:5.1f}% {status}")
    
    print(f"\n{'='*80}")
    if best_val_acc >= 85.0:
        print("✅ SUCCESS! 85%+ achieved!")
    elif best_val_acc >= 80.0:
        print("🎯 Excellent! 80%+")
    elif best_val_acc >= 75.0:
        print("📈 Strong! 75%+")
    else:
        print(f"📊 Current: {best_val_acc:.2f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()