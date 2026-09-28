# RetroFutureAI

Experimental fixed-point inference targeting a stock-era Amiga 1200.

The goal of this repository is deliberately small: explore what a minimal
inference pipeline can look like on constrained 68k-era hardware, with no
assumption of an FPU, GPU, large runtime, or modern ML framework.

This is **not** intended to be a useful AI system. It is a retro-computing
experiment focused on model representation, integer arithmetic, memory use,
repeatable tests, and eventually running the same inference code on real
Amiga hardware.

## Current status

Early prototype / pre-alpha.

The repository currently contains:

- a tiny integer-only classifier;
- deterministic model weights compiled into the binary;
- a host-side test harness;
- a small command-line demo;
- notes for an eventual Amiga 1200 build and hardware validation.

The reference implementation is intentionally portable C and avoids dynamic
allocation and floating-point inference.

## Why fixed point?

An Amiga 1200 is a very different target from a modern ML workstation.
Keeping inference integer-only makes the experiment easier to reason about:

- predictable memory use;
- no runtime model loader;
- no dependency on floating-point hardware;
- compact weights;
- code that can be tested on a modern host before moving to 68k.

The current prototype uses signed 8-bit inputs and weights with 32-bit
accumulators. That is enough to exercise the basic inference path without
pretending that this is a production model.

## Repository layout

```text
.
├── src/
│   ├── inference.c
│   ├── inference.h
│   └── main.c
├── tests/
│   └── test_inference.c
├── docs/
│   ├── ARCHITECTURE.md
│   └── HARDWARE.md
├── .github/workflows/
│   └── ci.yml
├── Makefile
└── README.md
```

## Build on a modern host

```bash
make
./retrofutureai
make test
```

Tested with GCC/Clang-style C toolchains. The code is deliberately conservative
so it can later be adapted for an Amiga-oriented compiler/toolchain.

## Example

```text
input: [8, 2, -3, 1]
scores: [77, -38, 5]
class: signal
```

The sample model is synthetic. Its purpose is to validate the inference
mechanics, not classification accuracy.

## Target direction

Planned work includes:

- cross-compile the same core for m68k;
- record binary size, inference latency, and memory footprint;
- replace the synthetic weights with an exported tiny trained model;
- add golden-vector tests shared between host and Amiga builds;
- test on real Amiga 1200 hardware;
- investigate whether a small hidden layer is still practical without making
  the runtime unnecessarily complicated.

## Scope

This project is intentionally narrow. Training happens elsewhere; the Amiga
side only needs a compact model representation and an inference routine.

No claims are made about real-world model quality at this stage.

## License

MIT. See [LICENSE](LICENSE).
