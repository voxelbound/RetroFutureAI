# Architecture notes

## Design goal

Keep the first Amiga-facing inference runtime small enough that every byte and
operation can be understood.

The current path is:

```text
input vector
    ↓
signed 8-bit features
    ↓
integer multiply / accumulate
    ↓
32-bit class scores
    ↓
argmax
    ↓
class id
```

There is no model loader, heap allocation, tensor runtime, or floating-point
dependency.

## Why start with a linear model?

A linear classifier is intentionally unambitious. It provides a useful first
milestone because it lets the project validate:

1. model representation;
2. deterministic inference;
3. host-vs-target test vectors;
4. compiler behaviour on 68k;
5. timing and binary-size measurement.

Once that path is stable, a tiny hidden layer can be added without changing
the basic testing strategy.

## Model export direction

A future exporter can turn a tiny model trained on a modern machine into a C
header containing quantized weights and biases.

That keeps training and inference clearly separated:

```text
modern machine:
training → quantization → exported weights

Amiga:
compiled weights → integer inference
```

## Non-goals

For now the project deliberately does not attempt:

- on-device training;
- large language models;
- image generation;
- dynamic model formats;
- framework compatibility;
- claims of practical AI performance.
