import argparse
import json
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.models import VGG16_with_leaky_relu_and_spectral_dropout
from src.utils import load_images, lr_scheduler


def parse_args():
    """Parse command line arguments for model training."""
    parser = argparse.ArgumentParser(description="Train VGG16 with Spectral Dropout")
    parser.add_argument(
        "--model_name",
        type=str,
        default="spectral_dropout_model",
        help="Name identifier for saved weights and log files.",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Total training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Training batch size."
    )
    parser.add_argument(
        "--init_lr", type=float, default=0.1, help="Initial learning rate."
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Spectral dropout rate.",
    )
    parser.add_argument(
        "--cifar100",
        action="store_true",
        help="Use CIFAR-100 instead of CIFAR-10.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (epochs).",
    )
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="./weights",
        help="Directory to save model checkpoints.",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="./logs",
        help="Directory to save training logs and plots.",
    )
    return parser.parse_args()


def plot_and_save_history(history, model_name, logs_dir):
    """Plot training & validation metrics and save figures to disk."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(history.history["categorical_accuracy"], label="Train Acc")
    axes[1].plot(history.history["val_categorical_accuracy"], label="Val Acc")
    axes[1].set_title("Model Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plot_path = os.path.join(logs_dir, f"{model_name}_training_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training curves to: {plot_path}")


def train_model(
    model,
    train_images,
    train_labels,
    test_images,
    test_labels,
    args,
):
    """Compile and train the TensorFlow model with enhanced callbacks and data augmentation."""
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        args.weights_dir, f"{args.model_name}.weights.h5"
    )
    csv_log_path = os.path.join(args.logs_dir, f"{args.model_name}_log.csv")

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_weights_only=True,
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        LearningRateScheduler(lambda epoch: lr_scheduler(epoch)),
        CSVLogger(csv_log_path, append=False),
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # Data augmentation pipeline
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    datagen.fit(train_images)

    # Compile model
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.init_lr, momentum=0.9, nesterov=True
    )
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    # Execute training loop
    history = model.fit(
        datagen.flow(train_images, train_labels, batch_size=args.batch_size),
        epochs=args.epochs,
        validation_data=(test_images, test_labels),
        callbacks=callbacks,
        verbose=1,
    )

    # Save raw history data to JSON
    history_json_path = os.path.join(
        args.logs_dir, f"{args.model_name}_history.json"
    )
    with open(history_json_path, "w") as f:
        json.dump(history.history, f, indent=4)

    # Plot loss and accuracy curves
    plot_and_save_history(history, args.model_name, args.logs_dir)

    return history


if __name__ == "__main__":
    args = parse_args()

    # 1. Load Data
    dataset_name = "CIFAR-100" if args.cifar100 else "CIFAR-10"
    print(f"Loading {dataset_name} dataset...")
    train_img, train_lbl, test_img, test_lbl = load_images(
        cfr100=args.cifar100
    )

    # 2. Initialize Model
    print(
        f"Initializing Spectral VGG model (dropout rate = {args.dropout_rate})..."
    )
    model = VGG16_with_leaky_relu_and_spectral_dropout(rate=args.dropout_rate)

    # 3. Start Training
    print(f"Starting training loop for model: {args.model_name}...")
    history = train_model(
        model=model,
        train_images=train_img,
        train_labels=train_lbl,
        test_images=test_img,
        test_labels=test_lbl,
        args=args,
    )

    print(f"Training completed successfully. Checkpoints saved to {args.weights_dir}")