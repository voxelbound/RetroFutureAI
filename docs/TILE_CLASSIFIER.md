# Tile Classifier

The tile classifier is the first trained model used by RetroFutureAI.

Its purpose is not to solve a difficult machine-learning problem. Instead, it
provides a small and fully controlled model that can be trained on a modern PC,
quantized to integer arithmetic, exported to C, and eventually executed on
Amiga 1200-class hardware.

## Dataset

The dataset consists of synthetic 8×8 grayscale tiles.

Each tile contains 64 pixel values in the range:

```text
0..127
```

The dataset generator creates four balanced classes:

```text
horizontal
vertical
diagonal
blob
```

The generated dataset is split into:

```text
training:   4000 samples
validation: 1000 samples
test:       1000 samples
```

The default generator seed is:

```text
1200
```

Using the same generator version and seed produces the same dataset again.

The generated dataset is stored locally in:

```text
data/generated/tile_dataset.npz
```

Generated data is intentionally excluded from Git. The repository stores the
generator rather than the generated dataset.

The generator is:

```text
tools/generate_tile_dataset.py
```

## Example classes

The patterns are intentionally simple.

A horizontal sample may look like:

```text
........
........
........
........
.######.
........
........
........
```

A vertical sample may look like:

```text
.##.....
.##.....
.##.....
.##.....
.##.....
.##.....
.##.....
.##.....
```

A diagonal sample may look like:

```text
........
.......#
......#.
.....#..
....#...
...#....
..#.....
.#......
```

A blob sample may look like:

```text
........
........
........
....##..
...###..
..+###..
........
........
```

The exact samples vary, but the generator introduces only modest noise and
variation. The first model is deliberately easy to train so that later work can
focus on inference correctness and target portability.

## Model architecture

The first trained model is a small fully connected neural network:

```text
8×8 input tile
      ↓
64 input values
      ↓
Dense 64 → 16
      ↓
ReLU
      ↓
Dense 16 → 4
      ↓
class scores
```

The four final outputs correspond to:

```text
0 = horizontal
1 = vertical
2 = diagonal
3 = blob
```

During training, the class scores are passed to cross-entropy loss.

During inference, the predicted class is the output with the largest score.

## Training

Training is performed on a modern host using PyTorch.

The training script is:

```text
tools/train_tile_classifier.py
```

The default training configuration is:

```text
epochs:        30
batch size:    64
learning rate: 0.001
seed:          1200
optimizer:     Adam
loss:          cross entropy
```

Input pixels are stored in the dataset as integers in the range `0..127`.

For floating-point training they are converted to `float32` and divided by:

```text
127.0
```

This gives the training model inputs approximately in the range:

```text
0.0 .. 1.0
```

The model is small enough to train quickly on CPU, although the training script
can also use CUDA when available.

## Reproducing the training run

Create the dataset:

```bash
python tools/generate_tile_dataset.py
```

Then train the model:

```bash
python tools/train_tile_classifier.py
```

The training script reports validation accuracy after every epoch and prints
final train, validation, and test metrics.

The initial reference run with seed `1200` reached:

```text
train accuracy:      100.00%
validation accuracy: 100.00%
test accuracy:       100.00%
```

These numbers should not be interpreted as evidence that the classifier solves
a difficult vision task. The dataset is intentionally simple and synthetic.

The important result is that the model learns the generated classes reliably
and gives us stable learned weights for the next stages of the project.

## Floating-point checkpoint

After training, the script writes a PyTorch checkpoint to:

```text
models/generated/tile_classifier.pt
```

The checkpoint contains:

- learned model parameters;
- model dimensions;
- class names;
- input normalization information;
- training configuration;
- final train, validation, and test metrics.

The generated checkpoint is intentionally excluded from Git.

Like the generated dataset, it can be recreated from the scripts and fixed
training seed.

## Why train in floating point?

RetroFutureAI targets integer inference, but training and inference are
different concerns.

Training benefits from:

- automatic differentiation;
- floating-point optimization;
- mature ML tooling;
- easy experimentation;
- fast host hardware.

There is little value in reproducing the training process on an Amiga.

Instead, the project follows this separation:

```text
Modern PC
─────────

synthetic dataset
      ↓
floating-point training
      ↓
trained float model

Integer conversion
──────────────────

trained float model
      ↓
quantization
      ↓
integer reference model
      ↓
exported C model data

Target inference
────────────────

integer input
      ↓
RetroFutureAI C runtime
      ↓
prediction
```

The Amiga therefore only needs the final model representation and the inference
runtime.

## Relationship to the runtime

The current C runtime already contains the operators needed for this network:

```text
rf_dense_i8()
rf_relu_i32()
rf_requantize_i32_to_i8()
rf_argmax_i32()
```

The floating-point training model does not yet use those operators directly.

Later blocks will convert the trained model into an integer representation that
matches the runtime semantics.

The intended integer execution path is:

```text
int8 input
    ↓
Dense 64 → 16
    ↓
int32 accumulators
    ↓
ReLU
    ↓
Requantize to int8
    ↓
Dense 16 → 4
    ↓
int32 class scores
    ↓
Argmax
```

## Next steps

The next stages are deliberately separated from training.

They will introduce:

```text
Python integer reference inference
        ↓
quantization
        ↓
model export
        ↓
C integration
        ↓
golden-vector equivalence tests
```

This separation makes it possible to compare every representation of the model
and detect semantic differences before moving to m68k and Amiga-specific code.
