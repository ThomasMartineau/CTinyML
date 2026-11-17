"""Serial test harness: streams real, held-out MNIST images to a
CTinyML-flashed Arduino Mega 2560 over pyserial and compares its
predicted class against the true label.

Protocol (matches arduino/ctinyml_serial_infer/ctinyml_serial_infer.ino),
115200 baud:
  1. Board sends "READY" (newline-terminated) once after setup().
  2. Host sends: 1 sync byte (0xAA) + 784 int8 bytes -- a single
     pre-quantized MNIST image (via quantize_input(), the same function
     Phase 7's own validation uses -- the board does zero quantization
     work).
  3. Board replies with exactly 1 byte: the predicted class (0-9).

Usage:
    python tools/serial_infer.py --port COM5
    python tools/serial_infer.py --port /dev/ttyACM0 --n 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import serial

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ctinyml.frontend.load import load_model  # noqa: E402
from ctinyml.passes import run_structural_passes  # noqa: E402
from ctinyml.passes.convert_to_closed_ir import convert  # noqa: E402
from ctinyml.passes.drop_final_requant import apply as drop_final_requant  # noqa: E402
from ctinyml.passes.quantize import quantize, quantize_input  # noqa: E402

MNIST_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mnist" / "mnist_classifier.onnx"
HOLDOUT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "mnist_holdout.npz"

SYNC_BYTE = b"\xaa"
IMG_BYTES = 784


def _quantized_input_tensor():
    """Rebuilds the exact same quantized graph the flashed model.c was
    generated from (same real calibration images), so the input
    tensor's scale/zero_point are guaranteed consistent with whatever
    is actually running on the board.
    """
    graph = convert(run_structural_passes(load_model(MNIST_FIXTURE)))
    drop_final_requant(graph)
    calibration_images = np.load(HOLDOUT_FIXTURE)["calibration_images"][:, np.newaxis]
    quantize(graph, calibration_images)
    return graph.inputs[0]


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


def predict(ser: serial.Serial, quantized_image: np.ndarray) -> int:
    ser.write(SYNC_BYTE + quantized_image.tobytes())
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

    print("Rebuilding the quantized graph (for the input tensor's scale/zero_point)...")
    input_tensor = _quantized_input_tensor()

    data = np.load(HOLDOUT_FIXTURE)
    eval_images = data["eval_images"][:, np.newaxis][: args.n]
    eval_labels = data["eval_labels"][: args.n]

    print(f"Opening {args.port} @ {args.baud} baud...")
    with serial.Serial(args.port, args.baud, timeout=args.timeout) as ser:
        print("Waiting for the board to report READY...")
        _wait_for_ready(ser)
        print("Board ready. Sending images...\n")

        correct = 0
        errors = 0
        for i, (image, label) in enumerate(zip(eval_images, eval_labels, strict=True)):
            quantized = quantize_input(image, input_tensor).reshape(-1)
            try:
                pred = predict(ser, quantized)
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
