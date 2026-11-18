"""Serial test harness: streams real, held-out MNIST images to a
CTinyML-flashed Arduino Mega 2560 over pyserial and compares its
predicted class against the true label.

Protocol (matches arduino/ctinyml_serial_infer/ctinyml_serial_infer.ino),
115200 baud:
  1. Board sends "READY" (newline-terminated) once after setup().
  2. Host sends: 1 sync byte (0xAA) + 784 int8 bytes -- a single MNIST
     image exactly as stored in the fixture's `eval_images_int8` (uint8
     pixel - 128, the compiler's fixed INT8 input encoding), so neither
     host nor board does any quantization work.
  3. Board replies with exactly 1 byte: the predicted class (0-9).

Usage:
    python tools/serial_infer.py --port COM5
    python tools/serial_infer.py --port /dev/ttyACM0 --n 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import serial

REPO_ROOT = Path(__file__).resolve().parent.parent

HOLDOUT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mnist_holdout.npz"

SYNC_BYTE = b"\xaa"
IMG_BYTES = 784


def _wait_for_ready(ser: serial.Serial, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = ser.readline().decode(errors="replace").strip()
        if line == "READY":
            return
        if line:
            print(f"(board) {line}")
    raise TimeoutError(
        "Board never sent READY -- check the port/baud rate, and that the "
        "sketch actually flashed (many boards reset when the port opens; "
        "this already waits through that)."
    )


def predict(ser: serial.Serial, image_int8: np.ndarray) -> int:
    ser.write(SYNC_BYTE + image_int8.tobytes())
    response = ser.read(1)
    if len(response) != 1:
        raise TimeoutError("No response from board (timed out waiting for the predicted class).")
    return int(response[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM5 or /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--n", type=int, default=20, help="Number of held-out images to test")
    parser.add_argument("--timeout", type=float, default=5.0, help="Per-image response timeout (s)")
    args = parser.parse_args()

    data = np.load(HOLDOUT_FIXTURE)
    eval_images = data["eval_images_int8"][: args.n]
    eval_labels = data["eval_labels"][: args.n]

    print(f"Opening {args.port} @ {args.baud} baud...")
    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        print("Waiting for the board to report READY...")
        _wait_for_ready(ser)
        print("Board ready. Sending images...\n")

        correct = 0
        errors = 0
        for i, (image, label) in enumerate(zip(eval_images, eval_labels, strict=True)):
            try:
                pred = predict(ser, image.reshape(-1))
            except TimeoutError as exc:
                print(f"[{i:3d}] true={label}  ERROR: {exc}")
                errors += 1
                continue

            ok = "OK" if pred == label else "MISS"
            correct += int(pred == label)
            print(f"[{i:3d}] true={label}  pred={pred}  {ok}")

        n = len(eval_labels)
        answered = n - errors
        print(f"\n{correct}/{n} correct ({correct / n:.1%})", end="")
        if errors:
            print(f"  ({errors} no-response errors; {correct}/{answered} of answered)")
        else:
            print()


if __name__ == "__main__":
    main()
