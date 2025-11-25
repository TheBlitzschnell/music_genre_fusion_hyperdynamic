# Music Genre Classification with Hierarchical Feature Fusion

A deep learning system for automatic music genre classification using the GTZAN dataset. This project implements a hierarchical feature fusion approach combining three specialized CNN architectures, achieving **93.33% accuracy** on the DPA-CNN model—exceeding published benchmarks by nearly 2%.

## Highlights

- **Novel "Hyper-Dynamic" Training Methodology**: Adaptive training strategy featuring class graduation, dynamic windowing, and real-time weight adjustment
- **Ensemble of Specialized Architectures**: CNN-BiGRU, DPA-CNN (Dual Parallel Attention), and ECAS-CNN (Efficient Channel Attention with Spatial attention)
- **896-dimensional Feature Fusion**: Concatenated features from all three models for robust genre representation
- **10x-20x Training Speedup**: Achieved through hyper-dynamic training while improving accuracy

## Results

| Model | Accuracy | Feature Dim | Notes |
|-------|----------|-------------|-------|
| CNN-BiGRU | 71.33% | 128 → 256 | Temporal pattern capture |
| ECAS-CNN | 80.67% | 128 | MFCC-based efficient attention |
| **DPA-CNN** | **93.33%** | 512 | Exceeds published 91.4% benchmark |
| Fusion | 73.33% | 654 | Combined feature classification |

## Architecture Overview

```
Audio Input (3-second windows)
         │
         ├──────────────────────────────────────────┐
         │                                          │
    Mel-Spectrogram                               MFCC
    (128×130 frames)                          (20×130 frames)
         │                                          │
         ├────────────┬─────────────┐              │
         │            │             │              │
    CNN-BiGRU    DPA-CNN      ─────────────  ECAS-CNN
    (256-dim)    (512-dim)                   (128-dim)
         │            │                           │
         └────────────┴───────────┬───────────────┘
                                  │
                          Feature Fusion (896-dim)
                                  │
                         Remove Zero-Variance (654-dim)
                                  │
                          Fusion Classifier
                                  │
                           Genre Prediction
```

## Project Structure

```
music_genre_fusion/
├── data/
│   ├── raw/                          # GTZAN dataset (download separately)
│   ├── processed/
│   │   ├── melspectrogram_normalized/  # Mel-spectrogram features
│   │   ├── mfcc/                       # MFCC features
│   │   ├── train_files.txt
│   │   ├── val_files.txt
│   │   └── test_files.txt
│   ├── splits/                       # Train/val/test splits
│   ├── fusion_features/              # Extracted fusion features
│   └── metadata.json
│
├── models/
│   ├── cnn_bigru/                    # CNN-BiGRU checkpoints
│   ├── dpa_cnn/                      # DPA-CNN checkpoints
│   ├── ecas_cnn/                     # ECAS-CNN checkpoints
│   └── fusion_progressive/           # Fusion classifier checkpoints
│
├── prepare_gtzan_data.py             # Dataset preparation & splits
├── notebooks/
│   └── extract_features.py           # Feature extraction from audio
├── train_bigru_cnn.py                # CNN-BiGRU training
├── train_dpa_cnn.py                  # DPA-CNN training (hyper-dynamic)
├── train_ecas_cnn.py                 # ECAS-CNN training
├── extract_model_features.py         # Extract features from trained models
├── create_cleaned_features.py        # Remove zero-variance features
├── verify_model_features.py          # Feature quality verification
├── train_fusion.py                   # Fusion classifier training
└── genre_detector.py                 # Inference script
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- librosa
- numpy
- scikit-learn
- tqdm

```bash
pip install torch torchvision torchaudio
pip install librosa numpy scikit-learn tqdm
```

### Dataset

1. Download the GTZAN dataset from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
2. Place the ZIP file in `data/raw/`
3. Run the preparation script:

```bash
python prepare_gtzan_data.py
```

This creates stratified train/val/test splits (70%/15%/15%).

## Usage

### Training Pipeline

**Step 1: Extract Audio Features**
```bash
python notebooks/extract_features.py
```
Generates mel-spectrograms and MFCC features from audio files.

**Step 2: Train Individual Models**
```bash
python train_dpa_cnn.py      # DPA-CNN (best performer)
python train_ecas_cnn.py     # ECAS-CNN
python train_bigru_cnn.py    # CNN-BiGRU
```

**Step 3: Extract Model Features**
```bash
python extract_model_features.py
```
Extracts 896-dimensional features from trained models.

**Step 4: Clean Features**
```bash
python create_cleaned_features.py
```
Removes zero-variance features (896 → 654 dimensions).

**Step 5: Train Fusion Classifier**
```bash
python train_fusion.py
```

### Inference

Classify a single audio file:
```bash
python genre_detector.py path/to/audio.mp3
```

**Example Output:**
```
====================================================================
GENRE DETECTION RESULTS
====================================================================
File: Paint_it_Black.mp3
Duration: 3:44

Top Predictions:
  1. rock       ████████████████████ 87.3%
  2. metal      ████████            32.1%
  3. blues      ███                 12.4%
====================================================================
```

## Hyper-Dynamic Training Methodology

The key innovation is the **hyper-dynamic training** approach, which achieved 10-20x speedups while improving accuracy:

### Class Graduation
Classes achieving ≥85% accuracy "graduate"—their predictions are saved and training resources shift to struggling classes.

### Adaptive Windowing
- Normal classes: 50% overlap between windows
- Struggling classes (<50% accuracy after epoch 50): 75% overlap for more training samples

### Dynamic Weight Adjustment
Real-time adjustment of class weights based on per-class performance, focusing resources on challenging genres.

### No Early Stopping
Full 300-epoch training ensures all classes have opportunity to improve.

```python
# Hyperparameters
GRADUATION_THRESHOLD = 85.0     # Class graduates at this accuracy
STRUGGLING_THRESHOLD = 50.0     # Below this = struggling
BASE_OVERLAP = 0.5              # Normal windowing
STRUGGLING_OVERLAP = 0.75       # Extra samples for struggling classes
```

## Model Architectures

### DPA-CNN (Dual Parallel Attention)
Combines channel and spatial attention mechanisms:
- 5 convolutional blocks with increasing channels (96→512)
- DPA modules after blocks 2-5
- 512-dimensional feature output

### ECAS-CNN (Efficient Channel Attention + Spatial)
Optimized for MFCC features:
- ECA modules for lightweight channel attention
- 4 convolutional blocks (192→512)
- 128-dimensional feature output

### CNN-BiGRU
Captures temporal patterns in spectrograms:
- 4 CNN layers for local feature extraction
- Bidirectional GRU for temporal modeling
- Attention-weighted aggregation
- 128-dimensional features (duplicated to 256 for fusion)

## Supported Genres

| Genre | Notes |
|-------|-------|
| Blues | Strong DPA-CNN performance |
| Classical | Highest accuracy across all models |
| Country | Moderate confusion with rock |
| Disco | Well-separated |
| Hip-Hop | Distinctive rhythmic patterns |
| Jazz | Good separation |
| Metal | Strong DPA-CNN performance |
| Pop | Some confusion with disco/rock |
| Reggae | Moderate difficulty |
| Rock | Most challenging—broad category |

## Configuration

Key parameters can be adjusted in the training scripts:

```python
# Data
WINDOW_SIZE = 130              # Frames per window (~3 seconds)
BATCH_SIZE = 32

# Training
LEARNING_RATE = 0.001
NUM_EPOCHS = 300
WEIGHT_DECAY = 5e-5

# Hyper-Dynamic
GRADUATION_THRESHOLD = 85.0
STRUGGLING_THRESHOLD = 50.0
```

## Hardware

Developed and tested on:
- **Apple Mac Mini M4** with MPS acceleration
- Training time: ~1-2 minutes per model with hyper-dynamic approach

Also supports:
- CUDA GPUs
- CPU (slower)

## Citation

If you use this work, please cite:

```bibtex
@software{music_genre_fusion,
  title = {Music Genre Classification with Hierarchical Feature Fusion},
  year = {2025},
  description = {Hyper-dynamic training for CNN ensemble genre classification}
}
```

## License

MIT License

## Acknowledgments

- GTZAN dataset by George Tzanetakis
- DPA-CNN architecture inspired by attention mechanism research
- ECAS-CNN based on ECANet efficient channel attention
