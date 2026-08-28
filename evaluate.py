import argparse
import numpy as np
import tensorflow as tf

from src.corruptions import test_robustness
from src.models import VGG16_with_leaky_relu_and_spectral_dropout
from src.utils import load_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on clean test sets and dynamic dynamic corruptions."
    )
    # Model Hyperparameters
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to trained model weights (.weights.h5 file).",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=0.2,
        help="Spectral dropout rate.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=1.6,
        help="Power parameter p for frequency-dependent scaling.",
    )
    parser.add_argument(
        "--mag_noise",
        type=float,
        default=0.1,
        help="Magnitude noise standard deviation scaling.",
    )

    # Evaluation Controls
    parser.add_argument(
        "--use_cifar100",
        action="store_true",
        help="Evaluate model using CIFAR-100 instead of CIFAR-10.",
    )
    parser.add_argument(
        "--eval_corruptions",
        action="store_true",
        help="Run dynamic robustness suite using test_robustness in corruptions.py.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Evaluation batch size.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Initialize Architecture
    print(f"Initializing model (rate={args.rate}, power={args.power}, mag_noise={args.mag_noise}), cifar100?({args.use_cifar100})...")
    model = VGG16_with_leaky_relu_and_spectral_dropout(
        rate=args.rate,
        power=args.power,
        mag_noise=args.mag_noise,
        classes=100 if args.use_cifar100 else 10
    )

    model.compile(
        optimizer="sgd",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )

    # 2. Build model dimensions & load weights
    dummy_input = tf.zeros((1, 32, 32, 3))
    model(dummy_input, training=False)
    
    print(f"Loading weights from: {args.weights_path}")
    model.load_weights(args.weights_path)

    # 3. Clean Data Evaluation
    print("Loading test dataset...")
    _, _, test_images, test_labels = load_images(cfr100=args.use_cifar100)

    print("\n--- Evaluating Clean Test Set ---")
    clean_loss, clean_acc = model.evaluate(test_images, test_labels, batch_size=args.batch_size, verbose=1)
    print(f"Clean Test Loss     : {clean_loss:.4f}")
    print(f"Clean Test Accuracy : {clean_acc * 100:.2f}%")

    # 4. Integrated Dynamic Corruptions Benchmarking
    if args.eval_corruptions:
        print("\n" + "=" * 60)
        print("   STARTING INTEGRATED CORRUPTION ROBUSTNESS SUITE")
        print("=" * 60)
        
        # Raw unnormalized test images are passed to test_robustness
        # Note: Ensure test_images are in uint8/raw range [0, 255] as required by corruptions.py
        raw_test_images = (test_images * 255.0).astype(np.uint8) if test_images.max() <= 1.0 else test_images

        corruption_errors = test_robustness(model, raw_test_images)
        mean_corruption_error = float(np.mean(corruption_errors))

        print("\n" + "=" * 60)
        print(f" Mean Corruption Error (mCE) : {mean_corruption_error:.2f}%")
        print("=" * 60)


if __name__ == "__main__":
    main()