"""CLI entrypoint: `python -m ctinyml compile model.onnx --target avr -o build/`.

`--target` is required from day one (only legal value `avr` for now) so
adding a second target later is additive, not a breaking CLI change.
"""

import argparse
import sys


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
        "-o", "--output", required=True, help="Output directory for generated C source."
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "compile":
        from ctinyml.frontend.load import load_model

        model = load_model(args.model)
        print(
            f"Loaded {args.model}: {model.graph.num_nodes()} nodes "
            f"(target={args.target}, output={args.output})"
        )
        print(
            "Compilation pipeline beyond the Phase 1 frontend is not implemented yet "
            "(see context/implementation_plan.md, Phases 2-8)."
        )
        return 0

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
