# CTinyML

An ahead-of-time compiler that turns a small ONNX model into dependency-free, int8-quantized C for microcontrollers. The first target is the **Arduino Mega 2560 (ATmega2560)**: 248 KB usable flash, 8 KB SRAM, no FPU, no `malloc`.

The compiler is written in Python and the generated code is plain C. Weights live in `PROGMEM`, activations live in two statically allocated ping-pong buffers, and the generated inference function is a short sequence of calls into a small hand-written kernel library.

> **Status:** research / learning project, pre-1.0. The pipeline works end to end for one model family (see [Supported models](#supported-models)). It is validated on the host and has been run on hardware through a serial harness; see [Status](#status) for exactly what has and hasn't been verified.

## How it works

```
 model.onnx
     │
     ▼
 frontend            load + shape inference, reject unsupported ops / dynamic shapes
     │
     ▼
 structural passes   drop no-op Reshape · fuse Relu into Conv/Gemm
     │               (operate on onnx_ir.Graph)
     ▼
 closed IR           narrow, target-agnostic IR: Conv / Gemm / ArgMax with quantization metadata on tensors
     │
     ▼
 quantize            per-channel symmetric int8 weights, asymmetric int8 activations,
     │               calibrated on real data via a float reference interpreter
     ▼
 buffer planner      liveness analysis → static ping-pong buffers, SRAM budget check (hard-fail at 8 KB)
     │
     ▼
 AVR codegen         PROGMEM weight arrays + model_forward() call sequence  →  model.c
     │
     ▼
 AVR runtime         shared, runtime-parameterized kernels: conv2d_int8, dense_int8, argmax_i32
```

Everything up to and including the buffer planner is target-agnostic. Only `ctinyml/targets/avr/` knows about `PROGMEM`, `pgm_read_*`, or the lack of an FPU, so a second target would be a sibling directory rather than a refactor.

### Design decisions worth knowing

- **Direct convolution**, no im2col. Zero-padding is handled by bounds checks in the loop, so no padded buffer is needed.
- **One shared kernel per op type**, called with different pointers and dimensions, rather than unrolled code per layer. This keeps flash use down.
- **ReLU is fused** into the preceding Conv/Gemm as a `clamp_min0` flag.
- **The graph input is a float in [0, 1] with a fixed INT8 encoding** (scale 1/255, zero point -128), so a uint8 pixel `p` is exactly the int8 value `p - 128`. That encoding is the `/255`, so nothing is divided on the device and the model must not contain a leading `Div`. Any other input scaling (mean/std, a different divisor) is not handled yet and must be folded into the first layer's weights before the model reaches the compiler.
- **`Reshape` is a compile-time alias** when it is a row-major flatten.
- **The final Gemm skips requantization.** `ArgMax` consumes its raw int32 accumulators directly.
- **Requantization is a float multiply** (`acc * requant_mult`, round half away from zero, saturate to `[-128, 127]`). AVR emulates float in software either way, so a fixed-point path would only add a second numeric path to validate.
- **Unsupported graph shapes fail loudly** (a non-fusable `Relu`, a Reshape that needs real data movement, a third simultaneously-live buffer) instead of miscompiling.

## Supported models

The op set is currently `Conv`, `Relu`, `Reshape`, `Gemm`, `ArgMax`, with static shapes and batch size 1. The reference model is a small MNIST classifier:

```
Conv(6,1,3×3, s2, p1) → ReLU → Conv(6,6,3×3, s2, p1) → ReLU → Reshape → Gemm(10×294) → Reshape → ArgMax
```

Its activations peak at 14×14×6 = 1176 bytes, so the two ping-pong buffers fit comfortably in SRAM.

## Repository layout

```
ctinyml/
  frontend/              ONNX loading, shape inference, op-set validation
  passes/                structural passes, closed-IR conversion, quantization, buffer planner
  closed_ir.py           narrow post-fusion IR
  interpreter.py         numpy float reference interpreter (calibration + validation baseline)
  ops.py                 target-agnostic per-op registry (shape derivation, float eval)
  layout.py              shape/stride derivation
  targets/avr/
    codegen/             emitters (weights, conv, gemm, argmax) and model.c assembly
    runtime/             hand-written C kernels: conv2d_direct.h, gemm.h, argmax.h, qmath.h, pgm_read_f32.h
tools/
  cli.py                 CLI entrypoint
  serial_infer.py        host-side harness that streams MNIST images to a flashed board
arduino/
  ctinyml_serial_infer/  Arduino sketch (generated model.c + runtime headers)
tests/                   unit, codegen, runtime (host-compiled C) and end-to-end tests
context/                 implementation plan, per-phase subplans, known_issues.md
mnist_model.ipynb        notebook used to train and export the reference model
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and, for the C runtime tests, a host `gcc`. uv will fetch a suitable Python (≥ 3.11) if you don't have one.

```bash
uv sync --extra dev
uv run pytest
```

`uv sync` installs the project in editable mode from `uv.lock`. The `dev` extra adds pytest, the training stack (torch, lightning, scikit-learn), `pyserial`, and Jupyter. The compiler itself only needs `onnx`, `onnx-ir` and `numpy`, so `uv sync` alone is enough to use it.

### CLI

```bash
uv run python -m ctinyml compile model.onnx --target avr --calibration calibration.npz -o build/
```

`--target` is required from day one so adding targets later is not a breaking change. `--calibration` takes float [0, 1] images (an `.npz` with a `calibration_images` array, or a `.npy`) used to calibrate quantization. The command runs the full pipeline and writes `model.c` plus the AVR runtime headers into the output directory.

### Running on hardware

1. Generate `model.c` and the runtime headers straight into the sketch folder with `-o arduino/ctinyml_serial_infer` (see [arduino/README.md](arduino/README.md) for the exact command).
2. Compile and flash the sketch to a Mega 2560 (Arduino IDE or `arduino-cli`).
3. Stream held-out MNIST digits to the board and compare against the true labels:

```bash
uv run --extra dev python tools/serial_infer.py --port COM5 --n 50
```

The board does no quantization. The host sends one sync byte (`0xAA`) followed by 784 pre-quantized int8 pixels at 115200 baud, and the board replies with a single byte, the predicted class.

## Status

| Phase | Scope | State |
|---|---|---|
| 0–1 | Scaffold, ONNX frontend | Done |
| 2 | Structural passes, closed IR | Done |
| 3 | Float interpreter, quantization | Done |
| 4 | Buffer planner | Done |
| 5 | AVR codegen | Done |
| 6 | AVR runtime kernels | Done, tested on host (x86) |
| 7 | End-to-end validation | Done |
| 8 | CSR sparsity for the Gemm layer | Stretch, not started |

Phase 7 results on 100 real held-out MNIST images (128 real calibration images): the quantized C model's argmax matches the float reference interpreter 99% of the time, against a 95% acceptance gate, and true-label accuracy is 94% (float) / 93% (C), consistent with the notebook's 97.8% given the small sample.

**Caveats:**

- The kernel and end-to-end tests compile the C with host `gcc`. The `#ifdef __AVR__` / `PROGMEM` path is exercised on the board through the serial harness, not in the automated test suite.
- Flash-size verification with `avr-size` against the 248 KB budget has not been recorded here.
- Phase 8 (CSR) would not help the reference model, whose weights are fully dense; it is planned for future pruned models.

## Further reading

- [context/implementation_plan.md](context/implementation_plan.md): the phased plan and locked design decisions
- [context/known_issues.md](context/known_issues.md): bugs and gotchas hit during implementation
- [context/mnist_conv_dense_int8_avr.c](context/mnist_conv_dense_int8_avr.c): the hand-written reference runtime the kernels were ported from

## License

See [LICENSE](LICENSE).
