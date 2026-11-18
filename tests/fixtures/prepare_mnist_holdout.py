"""One-time data-prep script -- NOT a pytest test, not run as part of the
suite. Fetches real MNIST via scikit-learn (same source/split convention
as mnist_model.ipynb: mnist_784 preserves original ordering, so
X[:60000]/X[60000:] are the standard train/test split) and writes a small
checked-in fixture, tests/fixtures/mnist_holdout.npz, that Phase 7's
tests read directly -- no network/scikit-learn needed after this runs
once.

Run manually: conda run -n ctinyml python tests/fixtures/prepare_mnist_holdout.py
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

OUTPUT = Path(__file__).parent / "mnist_holdout.npz"


def main() -> None:
    # parser="liac-arff" (not "auto") deliberately avoids adding pandas as a
    # dependency just for this one-time script -- "auto" checks for pandas
    # even with as_frame=False.
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X, y = mnist.data, mnist.target.astype(np.int64)
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]

    rng = np.random.default_rng(0)
    calib_idx = rng.choice(len(X_train), size=128, replace=False)
    eval_idx = rng.choice(len(X_test), size=100, replace=False)

    # Input contract: the graph input is a float in [0, 1] with a fixed INT8
    # encoding (scale 1/255, zero_point -128), so a uint8 pixel p is exactly
    # int8 p - 128. `eval_images_int8` stores those wire-ready bytes (what
    # tools/serial_infer.py sends to the board as-is).
    calibration_pixels = X_train[calib_idx].reshape(-1, 1, 28, 28)
    eval_pixels = X_test[eval_idx].reshape(-1, 1, 28, 28)
    calibration_images = (calibration_pixels / 255.0).astype(np.float32)
    eval_images = (eval_pixels / 255.0).astype(np.float32)
    eval_images_int8 = (np.rint(eval_pixels).astype(np.int16) - 128).astype(np.int8)
    eval_labels = y_test[eval_idx]

    np.savez_compressed(
        OUTPUT,
        calibration_images=calibration_images,
        eval_images=eval_images,
        eval_images_int8=eval_images_int8,
        eval_labels=eval_labels,
    )
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")
    print(f"calibration_images: {calibration_images.shape}, eval_images: {eval_images.shape}")


if __name__ == "__main__":
    main()
