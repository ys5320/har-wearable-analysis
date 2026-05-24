# HAR Wearable Sensor Analysis Pipeline

End-to-end deep learning pipeline for Human Activity Recognition (HAR) from raw wearable inertial sensor data, benchmarked on the UCI HAR Dataset (6-class activity classification, subject-independent evaluation).

## Pipeline Overview

| Script | Model | Parameters | Test Accuracy |
|---|---|---|---|
| `01_LSTM_train.py` | Bidirectional LSTM (2 layers) | 533k | 89.3% |
| `02_Transformer_train.py` | Transformer Encoder (2 layers) | 68k | **91.0%** |

## Dataset

UCI Human Activity Recognition Using Smartphones Dataset — 10,299 samples from 30 subjects wearing a smartphone on the waist, recording 9-axis inertial signals (body acceleration x/y/z, gyroscope x/y/z, total acceleration x/y/z) at 50Hz, windowed into 128-timepoint segments.

6 activity classes: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying.

Subject-independent evaluation: training subjects (70%) and test subjects (30%) are completely different people — a harder and more realistic evaluation than cross-validation.

Download: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## Methods

**Input**
- Raw inertial signals only — no hand-engineered features
- Shape: (n_samples, 9 channels, 128 timepoints)
- Normalised per channel using training set statistics

**LSTM**
- 2-layer bidirectional LSTM, hidden size 128 (256 after concatenation)
- Dropout 0.3 between layers and before classifier
- Gradient clipping (max_norm=1.0)
- Final hidden state (last timepoint) → linear classifier

**Transformer**
- Input projection: 9 → 64 (d_model)
- Sinusoidal positional encoding
- 2-layer Transformer encoder: 4 attention heads, feedforward dim 128
- Pre-LayerNorm (norm_first=True) for training stability
- Mean pooling across all timepoints → linear classifier

**Training (both models)**
- Adam optimiser, lr=1e-3, weight_decay=1e-4
- CosineAnnealingLR scheduler over 100 epochs
- Batch size 64, GPU: L40S (Imperial College HPC CX3)

## Results

| Model | Parameters | Test Accuracy |
|---|---|---|
| Chance level | — | 16.7% |
| LSTM (Bidirectional, 2-layer) | 533k | 89.3% |
| Transformer Encoder (2-layer) | **68k** | **91.0%** |

Transformer achieves higher accuracy with 8× fewer parameters than LSTM, demonstrating that self-attention's direct modelling of all pairwise timepoint relationships is more parameter-efficient than sequential processing for wearable sensor timeseries.

Main confusion in both models: Sitting vs Standing — a known HAR challenge as both are stationary activities with similar sensor variance, differing only in subtle gravity vector orientation sensitive to inter-subject postural variation.

## Requirements

```bash
conda create -n eeg python=3.11
pip install numpy scipy matplotlib scikit-learn torch torchvision
```

## Related Project

See [eeg-bci-analysis](https://github.com/ys5320/eeg-bci-analysis) for EEG motor imagery BCI decoding using MNE, CSP+LDA, and EEGNet — complementary pipeline covering electrophysiology signal processing.
