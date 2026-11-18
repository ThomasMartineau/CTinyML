"""CLI entrypoint: `python -m ctinyml compile model.onnx --target avr -o build/`.

`--target` is required from day one (only legal value `avr` for now) so
adding a second target later is additive, not a breaking CLI change.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

RUNTIME_DIR = Path(__file__).resolve().parent.parent / "ctinyml" / "targets" / "avr" / "runtime"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ctinyml",
        description="ONNX -> C compiler, AVR-first (Arduino Mega 2560 target).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser(
        "compile", help="Compile an ONNX model to target C source."
    )
    compile_parser.add_argument("model", help="Path to the input ONNX model.")
    compile_parser.add_argument(
        "--target",
        required=True,
        choices=["avr"],
        help="Target backend (only 'avr' is supported today).",
    )
    compile_parser.add_argument(
        "--calibration",
        required=True,
        help="Calibration images (.npz with 'calibration_images', or .npy), floats in [0, 1].",
    )
    compile_parser.add_argument(
        "-o", "--output", required=True, help="Output directory for generated C source."
    )

    return parser


def _load_calibration(path: str) -> np.ndarray:
    """Loads float [0, 1] calibration images; each sample is (1, C, H, W)."""
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "calibration_images" not in loaded.files:
            raise ValueError(f"{path}: .npz must contain a 'calibration_images' array")
        images = loaded["calibration_images"]
    else:
        images = loaded
    if images.ndim == 3:  # (N, H, W) -> (N, C, H, W)
        images = images[:, np.newaxis]
    if images.ndim != 4 or images.shape[0] == 0:
        raise ValueError(f"{path}: expected calibration images of shape (N, H, W) or (N, C, H, W)")
    if images.min() < 0.0 or images.max() > 1.0:
        raise ValueError(
            f"{path}: calibration images must be floats in [0, 1] (got range "
            f"[{images.min():g}, {images.max():g}]); divide raw 0-255 pixels by 255"
        )
    return images[:, np.newaxis]  # per-sample batch axis, as quantize() expects


def _compile(args: argparse.Namespace) -> int:
    from ctinyml.frontend.load import load_model
    from ctinyml.passes import run_structural_passes
    from ctinyml.passes.buffer_planner import plan
    from ctinyml.passes.convert_to_closed_ir import convert
    from ctinyml.passes.drop_final_requant import apply as drop_final_requant
    from ctinyml.passes.quantize import quantize
    from ctinyml.targets.avr.codegen.assemble import MODEL_C_PRELUDE, generate_model_c

    calibration = _load_calibration(args.calibration)
    model = run_structural_passes(load_model(args.model))
    print(f"Loaded {args.model}: {model.graph.num_nodes()} nodes after structural passes")

    graph = convert(model)
    drop_final_requant(graph)
    quantize(graph, calibration)
    buffer_plan = plan(graph)
    model_c = generate_model_c(graph, buffer_plan)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.c").write_text(MODEL_C_PRELUDE + model_c)
    for header in sorted(RUNTIME_DIR.glob("*.h")):
        shutil.copy2(header, out_dir / header.name)

    print(
        f"Buffer plan: 2 x {buffer_plan.buffer_size} B = {buffer_plan.total_sram_bytes} B SRAM"
    )
    if buffer_plan.warning:
        print(f"warning: {buffer_plan.warning}", file=sys.stderr)
    print(f"Wrote {out_dir / 'model.c'} and runtime headers (target={args.target})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "compile":
        from google.protobuf.message import DecodeError

        from ctinyml.frontend.load import FrontendError
        from ctinyml.passes.errors import ClosedIRPassError, StructuralPassError
        from ctinyml.targets.avr.codegen.errors import CodegenError

        try:
            return _compile(args)
        except (
            FrontendError,
            StructuralPassError,
            ClosedIRPassError,
            CodegenError,
            DecodeError,
            ValueError,
            OSError,
        ) as exc:
            print(f"ctinyml: error: {exc}", file=sys.stderr)
            return 1

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
