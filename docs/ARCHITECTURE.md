# Architecture

RetroFutureAI is being developed as a small integer inference runtime for
Amiga 1200-class hardware.

The architecture deliberately separates model training from target inference.

## System overview

The intended end-to-end system is:

```text
Modern PC
────────────────────────────────

training data
     ↓
model training
     ↓
floating-point model
     ↓
quantization
     ↓
integer reference inference
     ↓
model export
     ↓
C-compatible model data

────────────────────────────────

Portable runtime
────────────────────────────────

exported model
     +
integer input
     ↓
RetroFutureAI operators
     ↓
prediction

────────────────────────────────

Execution targets
────────────────────────────────

host C
m68k emulator
physical Amiga 1200
```

The current repository implements the portable C runtime and a small synthetic
host-side classifier.

Training, quantization, and model export will be added later.

## Runtime design

The runtime currently consists of four explicit operators:

```text
rf_dense_i8()
rf_relu_i32()
rf_requantize_i32_to_i8()
rf_argmax_i32()
```

The design is intentionally direct.

There is currently no:

- tensor abstraction;
- computational graph;
- dynamic layer system;
- model interpreter;
- allocator;
- binary model format.

Those abstractions should only be introduced if future examples demonstrate a
real need for them.

## Dense

The dense operator consumes:

```text
int8 input
int8 weights
int32 bias
```

and produces:

```text
int32 output
```

Conceptually:

```text
output[o] =
    bias[o]
    +
    Σ input[i] × weight[o, i]
```

Weights are stored output-row first:

```text
weights[output][input]
```

but are passed to the runtime as a contiguous array.

The current implementation uses portable C loops. This implementation is the
reference for possible future optimized kernels.

## ReLU

ReLU operates directly on signed 32-bit intermediate values:

```text
x < 0  → 0
x ≥ 0  → x
```

Keeping the intermediate representation at `int32` avoids premature loss of
precision immediately after accumulation.

## Requantization

A second dense layer requires compact integer activations rather than the
potentially much larger `int32` accumulator values from the previous layer.

The current requantization operation is:

```text
int32
   ↓
divide by 2^shift
   ↓
truncate toward zero
   ↓
clamp to [-128, 127]
   ↓
int8
```

The portable C implementation deliberately uses semantics that are easy to
define and test exactly.

For example, with `shift = 2`:

```text
  7 →  1
  8 →  2
 -7 → -1
 -8 → -2
```

Truncation toward zero is part of the runtime contract.

A future 68020-specific implementation may replace division with shifts and
sign adjustment, but it must produce exactly the same outputs.

## Argmax

Argmax selects the index containing the largest signed 32-bit value.

For ties:

```text
first maximum wins
```

For an empty input:

```text
-1
```

This behavior is deterministic and covered by tests.

## Current prototype

The current synthetic classifier is a single dense layer:

```text
4 inputs
   ↓
Dense
   ↓
3 class scores
   ↓
Argmax
```

The model definition remains in `src/inference.c`.

The classifier now uses the reusable runtime operators rather than implementing
its own multiply-accumulate and argmax loops.

This is intentionally still a simple model.

The already implemented ReLU and requantization operators will become relevant
when the first multi-layer model is introduced.

## Intended first trained model

The first real model is planned as a tiny graphics classifier.

Initial architecture:

```text
8 × 8 image
   ↓
64 input values
   ↓
Dense 64 → 16
   ↓
ReLU
   ↓
Requantize
   ↓
Dense 16 → 4
   ↓
Argmax
```

Initial classes are expected to represent simple graphical patterns such as:

```text
horizontal
vertical
diagonal
blob
```

The model will be trained on a modern PC and exported for integer inference.

## Integer behavior as a compatibility contract

One of the main architectural goals is exact repeatability across execution
targets.

The intended validation chain is:

```text
Python integer reference
        ==
portable host C
        ==
m68k C
        ==
optimized target kernels
```

Golden test vectors will later capture:

- model input;
- dense accumulator output;
- requantized activations;
- final logits;
- selected class.

This allows target optimization without changing model semantics.

## Optimization strategy

Optimization is deliberately postponed until the complete pipeline works.

The expected progression is:

```text
portable reference C
        ↓
profile
        ↓
optimized portable C
        ↓
68020-specific C
        ↓
selected 68020 assembly kernels
```

Likely optimization candidates include:

- dense inner loops;
- pointer-based traversal;
- loop unrolling;
- weight layout;
- requantization;
- model-specific specialization.

Assembler is intended for measured hotspots rather than general control code.

The reference implementation remains available for correctness testing.

## Memory strategy

The current runtime uses caller-owned buffers and static model data.

The target direction favors:

- static allocation;
- known dimensions;
- compact weights;
- limited temporary storage;
- no mandatory heap allocation.

This keeps memory behavior predictable on constrained systems.

## Model representation

The current synthetic model is compiled directly into the binary.

The first trained model will initially also be exported as C data rather than
loaded from a complex external format.

This avoids introducing:

- model parsers;
- dynamic allocation;
- filesystem dependencies;
- endian-sensitive binary formats

before they are actually useful.

A binary model format can be considered later.

## Design rule

Before adding a new runtime abstraction or operator, ask:

```text
Does a real example require it?
```

If not, it should normally wait.

The goal is a small understandable inference engine, not a miniature general
purpose ML framework.
