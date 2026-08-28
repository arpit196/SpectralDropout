# Polar Spectral Dropout: Frequency-Modulated Regularization for Out-of-Distribution Robustness

> **A Novel Frequency-Domain Regularization Method for Domain-Invariant Image Classification**

Spectral Dropout is a frequency-domain regularization technique designed to suppress the learning of high-frequency spurious patterns (such as fine textures, background noise, or domain-specific style artifacts). By applying targeted dropout masks in the spectral domain, the model filters out high-frequency domain-dependent noise and focuses on learning robust, low-frequency structural features, leading to significantly improved generalization under domain and distribution shifts.

---

## Key Features

- **Frequency-Domain Regularization:** Penalizes high-frequency noise directly in the spectral domain during training.
- **Improved Generalization:** Prevents overfitting to spurious textures and background artifacts.
- **Robustness to Domain Shift:** Out-of-the-box support for benchmarking against CIFAR-10-C / CIFAR-100-C image corruptions.
- **Modular Pipeline:** Easy-to-use command-line interface for training, evaluation, and hyperparameter tuning.

---

## Directory Structure

```text
SpectralDropout/
├── src/
│   ├── models.py          # VGG16 with Spectral Dropout & Leaky ReLU
│   ├── corruptions.py     # Image corruptions and corruption evaluation
│   ├── frost/             # Asset files required for specific corruption types
│   └── utils.py           # Data loaders, spectral utilities, learning rate schedules
├── weights/               # Directory for model checkpoints (*.weights.h5)
├── logs/                  # Training logs (*.csv, *.json) and metric plots
├── train.py               # Model training script
├── evaluate.py            # Model evaluation script
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
