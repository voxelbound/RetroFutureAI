# RetroFutureAI

Experimental fixed-point inference targeting Amiga 1200-class hardware.

RetroFutureAI explores how a small neural-network inference runtime can be
implemented for constrained 68k-era systems using predictable integer
arithmetic and minimal runtime dependencies.

The project currently runs and is tested on a modern host. Amiga 1200 is the
target platform; physical Amiga validation comes later.

## Current status

Early prototype / pre-alpha.

The repository currently contains:

- a small portable C inference runtime;
- signed 8-bit dense-layer weights and inputs;
- 32-bit accumulation;
- integer ReLU;
- deterministic requantization from `int32` to `int8`;
- integer argmax;
- deterministic unit tests;
- a synthetic linear-classifier demo;
- GitHub Actions host-side CI.

The existing demo is intentionally simple. It currently uses the reusable
dense and argmax operators to execute a single-layer synthetic classifier.

ReLU and requantization are already implemented and tested for the next step:
multi-layer integer inference.

## Runtime operators

The runtime is built from a few small operations that together form the
inference path.

**Dense** performs the main multiply-and-accumulate step of a fully connected
neural-network layer. Each input value is multiplied by a learned weight, the
results are added together with a bias, and one output score is produced for
each neuron in the layer.

**ReLU** is a simple activation function. Negative values are replaced with
zero, while positive values are left unchanged. This gives the network a
non-linear step between layers instead of reducing the entire model to one
large linear calculation.

**Requantize** converts the larger 32-bit intermediate values produced by a
dense layer back into compact 8-bit values that can be fed into the next
layer. RetroFutureAI currently does this by scaling with a power of two,
truncating toward zero, and clamping values to the signed 8-bit range.

**Argmax** chooses the largest value from the final output scores and returns
its position. In a classifier, that position corresponds to the predicted
class. For example, if four output scores represent four image categories,
argmax selects the category with the highest score.

Together, a small multi-layer classifier can therefore look like:

```text
input
  ↓
Dense       learned weighted combination
  ↓
ReLU        remove negative activations
  ↓
Requantize  reduce values back to int8
  ↓
Dense       produce class scores
  ↓
Argmax      select the predicted class
```

### Why int8 and int32?

Most model inputs and weights are stored as signed 8-bit integers (`int8`),
which keeps the model small.

During a dense calculation many of those values are multiplied and added
together, so the intermediate result needs more numeric range. RetroFutureAI
therefore accumulates those results in signed 32-bit integers (`int32`) before
reducing them back to 8-bit values when another layer follows.

This keeps model storage compact while still allowing intermediate arithmetic
to grow safely.

## Integer inference path

The intended multi-layer integer path is:

```text
int8 input
    ↓
Dense: int8 × int8 → int32
    ↓
ReLU
    ↓
Requantize: int32 → int8
    ↓
Dense
    ↓
Argmax
```

Requantization currently uses division by a power of two with truncation
toward zero followed by saturation to the signed 8-bit range.

This behavior is deliberately explicit and deterministic so that future host,
m68k, and optimized implementations can be checked against the same results.

## Why fixed-point?

An Amiga 1200 is a very different inference target from a modern machine.

Integer inference gives the project:

- predictable memory usage;
- compact model representation;
- no floating-point inference dependency;
- simple portable C code;
- deterministic host/target comparison;
- a practical path toward later 68020-specific optimization.

The goal is not to reproduce a modern ML framework. The runtime stays small
and gains new operations only when a real example requires them.

## Repository layout

```text
.
├── include/
│   └── retrofutureai.h
├── src/
│   ├── inference.c
│   ├── main.c
│   └── runtime/
│       ├── argmax.c
│       ├── dense.c
│       ├── relu.c
│       └── requantize.c
├── tests/
│   ├── test_argmax.c
│   ├── test_dense.c
│   ├── test_inference.c
│   ├── test_relu.c
│   └── test_requantize.c
├── docs/
│   ├── ARCHITECTURE.md
│   └── HARDWARE.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── Makefile
├── LICENSE
└── README.md
```

## Build on a modern host

A C99-compatible compiler and `make` are sufficient.

```bash
make
./retrofutureai
```

Run all tests with:

```bash
make test
```

Clean generated binaries with:

```bash
make clean
```

## Current demo

The current model is a small synthetic linear classifier compiled into the
binary.

Example:

```text
RetroFutureAI fixed-point inference demo
input: [8, 2, -3, 1]
scores: [77, -38, 5]
class: signal
```

The synthetic model exists to validate the runtime mechanics. Its weights were
not trained and it is not intended as a useful classifier.

## Next direction

The next major step is a small real graphics-classification example.

The planned path is:

```text
synthetic 8×8 graphics data
        ↓
train small model on PC
        ↓
quantize model
        ↓
export learned weights
        ↓
run same model through C runtime
        ↓
cross-compile for m68k
        ↓
Amiga validation
```

The first trained network is expected to remain deliberately small, initially
around:

```text
64 → 16 → 4
```

Training remains a PC-side concern. The target only needs the exported model
and the compact integer inference runtime.

## Longer-term direction

Once correctness has been established on Amiga hardware, selected runtime
hotspots can be optimized independently.

Possible future paths include:

- 68020-oriented C kernels;
- alternative data layouts;
- loop unrolling;
- generated model-specific code;
- selected 68020 assembly kernels;
- additional inference examples such as audio classification.

The portable C implementation remains the reference implementation against
which optimized versions can be tested.

## License

MIT. See [LICENSE](LICENSE).
