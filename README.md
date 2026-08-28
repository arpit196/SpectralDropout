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
```

1. To Clone the repository run:

git clone [https://github.com/your-username/SpectralDropout.git](https://github.com/your-username/SpectralDropout.git)
cd SpectralDropout

2. View and install Dependences:

pip install -r requirements.txt

Requires tensorflow >= 2.15, keras for training and opencv, pillow, scipy, skimage, scipy, wandb for generating image corruptions.

3. Training

Run train.py to train the Spectral Dropout model. Training metrics, logs, and loss/accuracy plots are automatically saved to ./logs/, and model weights are saved to ./weights/.
Some different ways to run training are shown below:

```bash
# Run training with default settings (CIFAR-10, 200 epochs, batch size 100)
python train.py

# Run training on CIFAR-100 with custom hyperparameters
python train.py \
    --model_name spectral_vgg16_cifar100 \
    --cifar100 \
    --epochs 150 \
    --batch_size 128 \
    --init_lr 0.05 \
    --dropout_rate 0.25 \
    --patience 20

# Run a fast test run (10 epochs)
python train.py \
    --model_name test_run \
    --epochs 10 \
    --batch_size 64
```

4. Evaluation.

The main evaluation metrics are top-1 accuracy on the clean dataset, CE (Corruption Error) for individual corruptions, as well as the mean Corruption Error (mCE) across corruptions. To evaluate a model, put its .weights.h5 file in the --weights_path argument and set the rate and power parameters for Polar Spectral Dropout. Setting the --eval_corruptions flag will report the mean corruption error (mCE) under different image corruptions (shot noise, Gaussian blur, etc) specified in the ImageNet-C along with the raw accuracy. For instance, to evaluate the accuracy and mCE of a VGG-16 model with Polar Spectral Regularization on CIFAR-100, run the following: 
```bash
# Run basic evaluation on standard test set
python evaluate.py --model_name spectral_vgg16_cifar10

# Benchmark model robustness across all 15 image corruptions and severity levels with specific levels of power and rate
python evaluate.py \
    --weights_path ../cifar100_spectralL.weights.h5   
    --rate 0.2   
    --power 1.6   
    --mag_noise 0.1   
    --eval_corruptions -
    -use_cifar100
```

Setting --eval_corruptions flags also reports the mCE of the model across the 15 corruptions along with raw clean accuracy. We observed that the models trained with Polar Spectral Dropout showed lower mCE than those trained with standard dropout, which means they had greater robustness.  



