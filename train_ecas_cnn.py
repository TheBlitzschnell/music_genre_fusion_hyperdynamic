"""
ECAS-CNN HYPER-DYNAMIC - FIXED (Correct Indexing)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
import random
from collections import defaultdict

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path.home() / "music_genre_fusion"
DATA_DIR = BASE_DIR / "data" / "processed" / "mfcc"
MODEL_DIR = BASE_DIR / "models" / "ecas_cnn"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
LEARNING_RATE = 0.0003
NUM_EPOCHS = 300
NUM_CLASSES = 10
WEIGHT_DECAY = 1e-4
GRADUATION_THRESHOLD = 85.0
STRUGGLING_THRESHOLD = 50.0
WEIGHT_ADJUSTMENT_EPOCH = 50

print("="*80)
print("ECAS-CNN HYPER-DYNAMIC - PREDICTION SAVING FIXED")
print("="*80)
print(f"Device: {device} | Epochs: {NUM_EPOCHS}")
print("="*80)


class SpecAugment:
    def __init__(self, freq_mask=3, time_mask=20, n_freq=1, n_time=2):
        self.freq_mask, self.time_mask = freq_mask, time_mask
        self.n_freq, self.n_time = n_freq, n_time
    
    def __call__(self, mfcc):
        mfcc = mfcc.copy()
        n_mfcc, time_steps = mfcc.shape
        for _ in range(self.n_freq):
            f, f0 = np.random.randint(0, self.freq_mask), np.random.randint(0, n_mfcc - self.freq_mask)
            mfcc[f0:f0+f, :] = 0
        for _ in range(self.n_time):
            t, t0 = np.random.randint(0, self.time_mask), np.random.randint(0, time_steps - self.time_mask)
            mfcc[:, t0:t0+t] = 0
        return mfcc


def augment_data(X, y, factor=3):
    aug = SpecAugment()
    X_list, y_list = [X], [y]
    for _ in range(factor - 1):
        X_list.append(np.array([aug(s) for s in X]))
        y_list.append(y)
    return np.concatenate(X_list), np.concatenate(y_list)


def create_augmented_data(X, y, strug):
    if not strug:
        return X, y
    all_X, all_y = [], []
    for c in range(NUM_CLASSES):
        mask = y == c
        X_c, y_c = X[mask], y[mask]
        if c in strug:
            X_c, y_c = augment_data(X_c, y_c, 3)
        all_X.append(X_c)
        all_y.append(y_c)
    return np.concatenate(all_X), np.concatenate(all_y)


class ECAModule(nn.Module):
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
        self.feat = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.3),
                                  nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(0.3))
        self.cls = nn.Sequential(nn.Linear(128, 128), nn.ReLU(True), nn.Dropout(0.4),
                                 nn.Linear(128, nc))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m.weight is not None:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.drop1(self.pool(self.eca1(F.relu(self.bn1(self.conv1(x))))))
        x = self.drop2(self.pool(self.eca2(F.relu(self.bn2(self.conv2(x))))))
        x = self.drop3(self.pool(self.eca3(F.relu(self.bn3(self.conv3(x))))))
        x = self.drop4(self.pool4(self.eca4(F.relu(self.bn4(self.conv4(x))))))
        x = self.gpool(x).view(x.size(0), -1)
        return self.cls(self.feat(x))


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def compute_per_class_accuracy(preds, labels):
    acc = defaultdict(lambda: [0, 0])
    for p, l in zip(preds, labels):
        acc[l][1] += 1
        if p == l: acc[l][0] += 1
    return {c: 100.0 * acc[c][0] / acc[c][1] if acc[c][1] > 0 else 0.0 for c in range(NUM_CLASSES)}


def get_weights(cls_acc, grad, strug):
    w = torch.ones(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        if c in grad: w[c] = 0.0
        elif c in strug: w[c] = 2.0
    return w


def train_epoch(model, loader, crit, opt, dev, grad):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for X, y in loader:
        mask = torch.tensor([l.item() not in grad for l in y])
        if not mask.any(): continue
        X, y = X[mask].to(dev), y[mask].to(dev)
        opt.zero_grad()
        out = model(X)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * X.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return (loss_sum / total if total > 0 else 0, 100.0 * correct / total if total > 0 else 0)


def validate_with_saved(model, loader, dev, grad_preds):
    """Use saved predictions for graduated classes"""
    model.eval()
    all_preds, all_labels = [], []
    class_idx = defaultdict(int)  # Per-class counter
    
    with torch.no_grad():
        for X, y in loader:
            X = X.to(dev)
            out = model(X)
            batch_preds = out.argmax(1).cpu().numpy()
            batch_labels = y.numpy()
            
            for i, label in enumerate(batch_labels):
                if label in grad_preds:
                    # Use saved prediction with per-class index
                    batch_preds[i] = grad_preds[label][class_idx[label]]
                class_idx[label] += 1
            
            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)
    
    acc = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    return acc, all_preds, all_labels


def save_preds_for_class(model, loader, dev, cls):
    """Save predictions for a specific class"""
    model.eval()
    preds = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(dev)
            out = model(X)
            batch_preds = out.argmax(1).cpu().numpy()
            batch_labels = y.numpy()
            for pred, label in zip(batch_preds, batch_labels):
                if label == cls:
                    preds.append(pred)
    return np.array(preds)


def main():
    print("\nLoading MFCC...")
    train_X = np.load(DATA_DIR / "train_features.npy")
    train_y = np.load(DATA_DIR / "train_labels.npy")
    val_X = np.load(DATA_DIR / "val_features.npy")
    val_y = np.load(DATA_DIR / "val_labels.npy")
    print(f"Train: {train_X.shape}, Val: {val_X.shape}\n")
    
    model = ECASCNN_HyperDynamic(NUM_CLASSES).to(device)
    grad, strug, grad_preds = set(), set(), {}
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, 'max', 0.7, 25, min_lr=1e-6)
    
    train_loader = DataLoader(SimpleDataset(train_X, train_y), BATCH_SIZE, True, num_workers=0)
    val_loader = DataLoader(SimpleDataset(val_X, val_y), BATCH_SIZE, False, num_workers=0)
    
    best_acc, start = 0.0, time.time()
    print(f"Training {NUM_EPOCHS} epochs...\n{'='*80}")
    
    for ep in range(1, NUM_EPOCHS + 1):
        if ep == WEIGHT_ADJUSTMENT_EPOCH:
            print(f"\n{'='*80}\nEPOCH {WEIGHT_ADJUSTMENT_EPOCH} CHECK\n{'='*80}")
            val_acc, vp, vl = validate_with_saved(model, val_loader, device, grad_preds)
            cls_acc = compute_per_class_accuracy(vp, vl)
            
            for c, a in cls_acc.items():
                if a >= GRADUATION_THRESHOLD and c not in grad:
                    grad_preds[c] = save_preds_for_class(model, val_loader, device, c)
                    grad.add(c)
                    print(f"🎓 Class {c} graduated: {a:.1f}% (saved)")
            
            found = False
            for c, a in cls_acc.items():
                if a <= STRUGGLING_THRESHOLD and c not in strug and c not in grad:
                    strug.add(c)
                    found = True
                    print(f"⚠️  Class {c} struggling: {a:.1f}%")
            
            if found:
                print("\nAugmenting struggling classes...")
                train_X_aug, train_y_aug = create_augmented_data(train_X, train_y, strug)
                val_X_aug, val_y_aug = create_augmented_data(val_X, val_y, strug)
                train_loader = DataLoader(SimpleDataset(train_X_aug, train_y_aug), BATCH_SIZE, True, num_workers=0)
                val_loader = DataLoader(SimpleDataset(val_X_aug, val_y_aug), BATCH_SIZE, False, num_workers=0)
            
            crit = nn.CrossEntropyLoss(weight=get_weights(cls_acc, grad, strug).to(device), label_smoothing=0.1)
            print(f"{'='*80}\n")
        
        if ep % 20 == 0 and ep != WEIGHT_ADJUSTMENT_EPOCH:
            val_acc, vp, vl = validate_with_saved(model, val_loader, device, grad_preds)
            cls_acc = compute_per_class_accuracy(vp, vl)
            for c, a in cls_acc.items():
                if a >= GRADUATION_THRESHOLD and c not in grad:
                    grad_preds[c] = save_preds_for_class(model, val_loader, device, c)
                    grad.add(c)
                    print(f"🎓 Class {c} graduated: {a:.1f}% (saved)")
            crit = nn.CrossEntropyLoss(weight=get_weights(cls_acc, grad, strug).to(device), label_smoothing=0.1)
        
        tr_loss, tr_acc = train_epoch(model, train_loader, crit, opt, device, grad)
        val_acc, vp, vl = validate_with_saved(model, val_loader, device, grad_preds)
        sched.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': opt.state_dict(), 'val_acc': val_acc,
                       'graduated_classes': list(grad), 'struggling_classes': list(strug),
                       'graduated_predictions': grad_preds,
                       'class_accuracies': compute_per_class_accuracy(vp, vl)},
                      MODEL_DIR / 'ecas_cnn_hyperdynamic_best.pth')
        
        if ep % 10 == 0:
            print(f"Epoch [{ep}/{NUM_EPOCHS}] Train: {tr_acc:.2f}% | Val: {val_acc:.2f}% {'⭐' if val_acc == best_acc else ''} | "
                  f"LR: {opt.param_groups[0]['lr']:.6f} | Grad: {len(grad)}/10")
    
    print(f"\n{'='*80}\nCOMPLETE! Best: {best_acc:.2f}% | Time: {(time.time()-start)/3600:.2f}h | Grad: {len(grad)}/10\n{'='*80}")
    
    ckpt = torch.load(MODEL_DIR / 'ecas_cnn_hyperdynamic_best.pth', weights_only=False)
    print("\nFinal per-class:")
    for c, a in ckpt['class_accuracies'].items():
        s = "🎓" if c in grad else "⚠️" if c in strug else "OK"
        print(f"  Class {c}: {a:.2f}% {s}")


if __name__ == "__main__":
    main()