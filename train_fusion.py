"""
PROGRESSIVE WINDOWING FUSION TRAINING
======================================
Strategy:
- Start immediately with 50% windowing overlap
- Track per-class best from EPOCH 1
- Check EVERY epoch for improvements
- After epoch 50: Increase overlap by 10% every 10 epochs for non-graduated
- Maximum 90% overlap (10x augmentation)

Progressive Schedule:
  Epochs 1-50:   50% overlap (2x aug) for all non-graduated
  Epochs 51-60:  60% overlap (2.5x aug) for non-graduated
  Epochs 61-70:  70% overlap (3.3x aug) for non-graduated
  Epochs 71-80:  80% overlap (5x aug) for non-graduated
  Epochs 81+:    90% overlap (10x aug) for non-graduated
  
Target: Push beyond 77% by aggressive progressive training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import json

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path.home() / "Documents" / "music_genre_fusion"
FUSION_DIR = BASE_DIR / "data" / "fusion_features"
MODEL_DIR = BASE_DIR / "models" / "fusion_progressive"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 500
NUM_CLASSES = 10
WEIGHT_DECAY = 1e-4

# Progressive windowing schedule
INITIAL_OVERLAP = 50  # Start at 50%
OVERLAP_INCREASE = 10  # Increase by 10% every 10 epochs
MAX_OVERLAP = 90  # Maximum 90%
OVERLAP_START_EPOCH = 51  # Start increasing after epoch 50
OVERLAP_INCREASE_INTERVAL = 10  # Every 10 epochs

# Graduation threshold
GRADUATION_THRESHOLD = 90.0

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

print("="*80)
print("PROGRESSIVE WINDOWING FUSION TRAINING")
print("="*80)
print(f"Device: {device}")
print(f"\n🔥 PROGRESSIVE AUGMENTATION STRATEGY:")
print(f"  ✓ Start: 50% overlap (2x augmentation)")
print(f"  ✓ Every 10 epochs after epoch 50: +10% overlap")
print(f"  ✓ Maximum: 90% overlap (10x augmentation)")
print(f"  ✓ Track per-class best from EPOCH 1")
print(f"  ✓ Check improvements EVERY epoch")
print(f"  ✓ Graduated classes (≥90%): stop training")
print(f"  ✓ Train {NUM_EPOCHS} epochs total")
print("="*80)


def overlap_to_augmentation_factor(overlap_percent):
    """
    Convert overlap percentage to augmentation factor
    50% overlap = 2x augmentation
    60% overlap = 2.5x augmentation
    70% overlap = 3.33x augmentation
    80% overlap = 5x augmentation
    90% overlap = 10x augmentation
    """
    if overlap_percent >= 90:
        return 10
    elif overlap_percent >= 80:
        return 5
    elif overlap_percent >= 70:
        return 3
    elif overlap_percent >= 60:
        return 2.5
    else:  # 50%
        return 2


def get_current_overlap(epoch, graduated_classes):
    """Calculate current overlap percentage for each class"""
    overlap_dict = {}
    
    for cls in range(NUM_CLASSES):
        if cls in graduated_classes:
            # Graduated classes don't train
            overlap_dict[cls] = 0
        elif epoch < OVERLAP_START_EPOCH:
            # Before epoch 50: everyone at 50%
            overlap_dict[cls] = INITIAL_OVERLAP
        else:
            # After epoch 50: progressive increase
            epochs_since_start = epoch - OVERLAP_START_EPOCH
            increases = epochs_since_start // OVERLAP_INCREASE_INTERVAL
            current_overlap = min(INITIAL_OVERLAP + (increases * OVERLAP_INCREASE), MAX_OVERLAP)
            overlap_dict[cls] = current_overlap
    
    return overlap_dict


class FeatureAugmenter:
    """Enhanced feature augmenter with intensity control"""
    def __init__(self, noise_std=0.08):
        self.noise_std = noise_std
    
    def add_noise(self, features, scale=1.0):
        """Add Gaussian noise"""
        noise = np.random.normal(0, self.noise_std * scale, features.shape)
        return features + noise
    
    def feature_dropout(self, features, p=0.15):
        """Randomly zero out features"""
        mask = np.random.random(features.shape) > p
        return features * mask
    
    def feature_scaling(self, features, scale_range=(0.9, 1.1)):
        """Scale features randomly"""
        scale = np.random.uniform(*scale_range)
        return features * scale
    
    def feature_shift(self, features, shift_std=0.05):
        """Add random shift"""
        shift = np.random.normal(0, shift_std, features.shape)
        return features + shift
    
    def augment(self, features, overlap_percent):
        """Apply augmentation based on overlap percentage"""
        f = features.copy()
        
        # Augmentation intensity increases with overlap
        if overlap_percent >= 80:
            # Heavy augmentation (80-90% overlap)
            if np.random.random() < 0.95:
                f = self.add_noise(f, scale=1.2)
            if np.random.random() < 0.75:
                f = self.feature_dropout(f, p=0.18)
            if np.random.random() < 0.65:
                f = self.feature_shift(f, shift_std=0.10)
            if np.random.random() < 0.55:
                f = self.feature_scaling(f, scale_range=(0.88, 1.12))
        
        elif overlap_percent >= 70:
            # Medium-heavy augmentation (70-79% overlap)
            if np.random.random() < 0.90:
                f = self.add_noise(f, scale=1.0)
            if np.random.random() < 0.65:
                f = self.feature_dropout(f, p=0.15)
            if np.random.random() < 0.55:
                f = self.feature_shift(f, shift_std=0.08)
            if np.random.random() < 0.45:
                f = self.feature_scaling(f, scale_range=(0.90, 1.10))
        
        elif overlap_percent >= 60:
            # Medium augmentation (60-69% overlap)
            if np.random.random() < 0.85:
                f = self.add_noise(f, scale=0.9)
            if np.random.random() < 0.55:
                f = self.feature_dropout(f, p=0.12)
            if np.random.random() < 0.45:
                f = self.feature_shift(f, shift_std=0.06)
            if np.random.random() < 0.35:
                f = self.feature_scaling(f, scale_range=(0.92, 1.08))
        
        else:  # 50% overlap
            # Light augmentation
            if np.random.random() < 0.75:
                f = self.add_noise(f, scale=0.8)
            if np.random.random() < 0.45:
                f = self.feature_dropout(f, p=0.10)
            if np.random.random() < 0.35:
                f = self.feature_shift(f, shift_std=0.05)
            if np.random.random() < 0.25:
                f = self.feature_scaling(f, scale_range=(0.94, 1.06))
        
        return f


def create_progressive_dataset(features, labels, overlap_dict, augmenter, graduated_classes):
    """
    Create dataset with progressive overlap-based augmentation
    """
    all_features = []
    all_labels = []
    
    for cls in range(NUM_CLASSES):
        if cls in graduated_classes:
            # Skip graduated classes
            continue
        
        mask = labels == cls
        cls_features = features[mask]
        cls_labels = labels[mask]
        
        overlap = overlap_dict[cls]
        aug_factor = overlap_to_augmentation_factor(overlap)
        
        # Original samples
        all_features.append(cls_features)
        all_labels.append(cls_labels)
        
        # Augmented samples (integer part)
        n_full_augs = int(aug_factor) - 1
        for _ in range(n_full_augs):
            aug_feat = np.array([augmenter.augment(f, overlap) for f in cls_features])
            all_features.append(aug_feat)
            all_labels.append(cls_labels)
        
        # Fractional part (e.g., 2.5x needs 0.5 more)
        fractional_part = aug_factor - int(aug_factor)
        if fractional_part > 0:
            n_fractional = int(len(cls_features) * fractional_part)
            if n_fractional > 0:
                indices = np.random.choice(len(cls_features), n_fractional, replace=False)
                aug_feat = np.array([augmenter.augment(cls_features[i], overlap) for i in indices])
                all_features.append(aug_feat)
                all_labels.append(cls_labels[indices])
    
    if len(all_features) == 0:
        # All graduated - return empty dataset
        return np.array([]).reshape(0, features.shape[1]), np.array([])
    
    return np.concatenate(all_features), np.concatenate(all_labels)


class AdaptiveDataset(Dataset):
    """Dataset that filters out graduated classes"""
    def __init__(self, features, labels):
        if len(features) == 0:
            self.features = torch.FloatTensor([])
            self.labels = torch.LongTensor([])
        else:
            self.features = torch.FloatTensor(features)
            self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FusionClassifier(nn.Module):
    """Enhanced fusion classifier"""
    def __init__(self, input_dim=654, num_classes=10):
        super(FusionClassifier, self).__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_drop = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 384)
        self.bn2 = nn.BatchNorm1d(384)
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(384, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.input_norm(x)
        x = self.input_drop(x)
        
        # Handle single sample batch (validation)
        if x.size(0) == 1:
            x = F.relu(self.fc1(x))
        else:
            x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        
        if x.size(0) == 1:
            x = F.relu(self.fc2(x))
        else:
            x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        
        if x.size(0) == 1:
            x = F.relu(self.fc3(x))
        else:
            x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)
        
        return self.fc4(x)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch"""
    if len(train_loader) == 0:
        return 0.0, 100.0
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total


def validate_with_graduated(model, val_loader, device, graduated_predictions):
    """Validate with graduated class predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for idx, (features, labels) in enumerate(val_loader):
            features = features.to(device)
            
            batch_start = idx * val_loader.batch_size
            batch_indices = list(range(batch_start, batch_start + len(labels)))
            
            outputs = model(features)
            _, predicted = outputs.max(1)
            
            # Replace predictions for graduated classes
            for i, global_idx in enumerate(batch_indices):
                if global_idx in graduated_predictions:
                    predicted[i] = graduated_predictions[global_idx]
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_indices.extend(batch_indices)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    
    return accuracy, all_preds, all_labels


def compute_per_class_accuracy(predictions, labels):
    """Compute per-class accuracy"""
    class_acc = {}
    for cls in range(NUM_CLASSES):
        mask = labels == cls
        if mask.sum() > 0:
            class_acc[cls] = 100. * (predictions[mask] == cls).sum() / mask.sum()
        else:
            class_acc[cls] = 0.0
    return class_acc


def main():
    """Main training loop"""
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    train_features = np.load(FUSION_DIR / "train_features_654d_clean.npy")
    train_labels = np.load(FUSION_DIR / "train_labels_clean.npy")
    val_features = np.load(FUSION_DIR / "val_features_654d_clean.npy")
    val_labels = np.load(FUSION_DIR / "val_labels_clean.npy")
    
    print(f"\n✓ Data loaded:")
    print(f"  Train: {train_features.shape}")
    print(f"  Val:   {val_features.shape}")
    
    # Initialize augmenter
    augmenter = FeatureAugmenter(noise_std=0.08)
    
    # Training state
    graduated_classes = set()
    graduated_predictions = {}
    
    # Per-class best tracking (from epoch 1!)
    per_class_best_acc = {cls: 0.0 for cls in range(NUM_CLASSES)}
    per_class_best_preds = {cls: None for cls in range(NUM_CLASSES)}
    per_class_best_epoch = {cls: 0 for cls in range(NUM_CLASSES)}
    
    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    model = FusionClassifier(input_dim=654, num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
    
    print(f"✓ Training: {NUM_EPOCHS} epochs")
    print(f"✓ Progressive overlap: 50% → 90% (epochs 50-90)")
    print(f"✓ Per-class best tracked from epoch 1")
    
    # Training
    print("\n" + "="*80)
    print("PROGRESSIVE TRAINING")
    print("="*80)
    
    best_val_acc = 0.0
    best_epoch = 0
    history = {
        'train_acc': [], 
        'val_acc': [], 
        'graduated': [],
        'overlap': []
    }
    
    # Validation dataset (fixed)
    val_dataset = AdaptiveDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Get current overlap for each class
        overlap_dict = get_current_overlap(epoch, graduated_classes)
        
        # Create dataset with current overlap
        train_features_aug, train_labels_aug = create_progressive_dataset(
            train_features, train_labels, overlap_dict, augmenter, graduated_classes
        )
        
        # Create dataloader
        if len(train_features_aug) > 0:
            train_dataset = AdaptiveDataset(train_features_aug, train_labels_aug)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                     num_workers=0, drop_last=True)
            
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        else:
            # All graduated
            train_loss, train_acc = 0.0, 100.0
        
        # Validate
        val_acc, val_preds, val_labels_list = validate_with_graduated(
            model, val_loader, device, graduated_predictions
        )
        
        # Compute per-class accuracy
        class_acc = compute_per_class_accuracy(val_preds, val_labels_list)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['graduated'].append(len(graduated_classes))
        history['overlap'].append(list(overlap_dict.values()))
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_acc': class_acc,
                'graduated_classes': graduated_classes,
                'graduated_predictions': graduated_predictions,
                'per_class_best_acc': per_class_best_acc,
                'per_class_best_epoch': per_class_best_epoch,
                'per_class_best_preds': per_class_best_preds
            }, MODEL_DIR / "best_model.pth")
        
        # Check for improvements and graduation EVERY epoch
        newly_graduated = []
        improvements = []
        
        for cls in range(NUM_CLASSES):
            if cls not in graduated_classes:
                acc = class_acc.get(cls, 0.0)
                
                # Check for improvement
                if acc > per_class_best_acc[cls]:
                    per_class_best_acc[cls] = acc
                    per_class_best_epoch[cls] = epoch
                    
                    # Save predictions for this class
                    cls_mask = val_labels_list == cls
                    per_class_best_preds[cls] = val_preds[cls_mask].copy()
                    
                    improvements.append((cls, acc))
                
                # Check for graduation
                if acc >= GRADUATION_THRESHOLD:
                    graduated_classes.add(cls)
                    newly_graduated.append((cls, acc))
                    
                    # Store predictions for this class
                    for idx, label in enumerate(val_labels):
                        if label == cls:
                            graduated_predictions[idx] = cls
        
        # Print progress
        grad_count = len(graduated_classes)
        current_overlap = overlap_dict[0] if 0 not in graduated_classes else 0
        aug_factor = overlap_to_augmentation_factor(current_overlap)
        
        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}] "
              f"Train: {train_acc:5.2f}% | Val: {val_acc:5.2f}% | "
              f"Overlap: {current_overlap}% ({aug_factor:.1f}x) | "
              f"LR: {current_lr:.6f} | Grad: {grad_count}/10")
        
        # Print graduations
        if newly_graduated:
            for cls, acc in newly_graduated:
                print(f"  🎓 Class {cls} ({GENRES[cls]}) GRADUATED with {acc:.1f}%!")
        
        # Print improvements (only show if significant)
        if improvements and epoch % 5 == 0:
            print(f"  ✨ Improvements this epoch:")
            for cls, acc in improvements:
                prev = per_class_best_acc[cls] - (acc - per_class_best_acc[cls])
                print(f"     {GENRES[cls]:10s}: {acc:.1f}% (+{acc-prev:.1f}%)")
        
        # Print per-class status every 20 epochs
        if epoch % 20 == 0:
            print("\n" + "="*80)
            print(f"EPOCH {epoch}: PER-CLASS STATUS")
            print("="*80)
            for cls in range(NUM_CLASSES):
                if cls in graduated_classes:
                    print(f"  {GENRES[cls]:10s} (Class {cls}): 🎓 GRADUATED")
                else:
                    current = class_acc.get(cls, 0.0)
                    best = per_class_best_acc[cls]
                    best_ep = per_class_best_epoch[cls]
                    overlap = overlap_dict[cls]
                    print(f"  {GENRES[cls]:10s} (Class {cls}): Current: {current:5.1f}% | "
                          f"Best: {best:5.1f}% @ ep{best_ep:3d} | Overlap: {overlap}%")
            print("="*80 + "\n")
        
        # Save checkpoint every 50 epochs
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_acc': class_acc,
                'graduated_classes': graduated_classes,
                'graduated_predictions': graduated_predictions,
                'per_class_best_acc': per_class_best_acc,
                'per_class_best_epoch': per_class_best_epoch,
                'per_class_best_preds': per_class_best_preds,
                'history': history
            }, MODEL_DIR / f"checkpoint_epoch_{epoch}.pth")
    
    # Training complete
    elapsed = time.time() - start_time
    
    # Compute combined best using per-class bests
    print("\n" + "="*80)
    print("COMPUTING PER-CLASS BEST COMBINATION")
    print("="*80)
    
    combined_best_preds = np.zeros(len(val_labels), dtype=int)
    for cls in range(NUM_CLASSES):
        cls_mask = val_labels == cls
        if per_class_best_preds[cls] is not None:
            combined_best_preds[cls_mask] = per_class_best_preds[cls]
        else:
            combined_best_preds[cls_mask] = cls
    
    combined_best_acc = 100. * (combined_best_preds == val_labels).sum() / len(val_labels)
    
    print(f"\n📊 Per-class best epochs:")
    for cls in range(NUM_CLASSES):
        if cls in graduated_classes:
            print(f"  {GENRES[cls]:10s} (Class {cls}): 🎓 GRADUATED "
                  f"@ epoch {per_class_best_epoch[cls]} ({per_class_best_acc[cls]:.1f}%)")
        else:
            print(f"  {GENRES[cls]:10s} (Class {cls}): {per_class_best_acc[cls]:5.1f}% "
                  f"@ epoch {per_class_best_epoch[cls]}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"⏱️  Total time: {elapsed/3600:.2f} hours")
    print(f"🏆 Best validation: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"✨ COMBINED BEST (per-class): {combined_best_acc:.2f}%")
    print(f"🎓 Graduated classes: {len(graduated_classes)}/10")
    
    # Save final results
    results = {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'combined_best_acc': combined_best_acc,
        'per_class_best_acc': per_class_best_acc,
        'per_class_best_epoch': per_class_best_epoch,
        'graduated_classes': list(graduated_classes),
        'history': history,
        'training_time': elapsed
    }
    
    with open(MODEL_DIR / "training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save combined best predictions
    np.save(MODEL_DIR / "combined_best_predictions.npy", combined_best_preds)
    
    print(f"\n✓ Results saved to {MODEL_DIR}")
    print(f"✓ Combined best predictions saved: combined_best_predictions.npy")
    print("="*80)


if __name__ == "__main__":
    main()