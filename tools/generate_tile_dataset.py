#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

IMAGE_SIZE = 8
CLASS_NAMES = np.array(["horizontal", "vertical", "diagonal", "blob"])


def _base_image(rng):
    return rng.integers(0, 9, size=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)


def _paint(image, y, x, value):
    if 0 <= y < IMAGE_SIZE and 0 <= x < IMAGE_SIZE:
        image[y, x] = np.uint8(max(int(image[y, x]), int(value)))


def _noise(image, rng):
    for _ in range(int(rng.integers(0, 4))):
        _paint(
            image,
            int(rng.integers(0, IMAGE_SIZE)),
            int(rng.integers(0, IMAGE_SIZE)),
            int(rng.integers(12, 45)),
        )


def _horizontal(rng):
    image = _base_image(rng)
    y = int(rng.integers(1, IMAGE_SIZE - 1))
    thickness = int(rng.integers(1, 3))
    start = int(rng.integers(0, 2))
    end = int(rng.integers(IMAGE_SIZE - 1, IMAGE_SIZE + 1))
    intensity = int(rng.integers(90, 128))

    for dy in range(thickness):
        row = min(y + dy, IMAGE_SIZE - 1)
        for x in range(start, min(end, IMAGE_SIZE)):
            _paint(image, row, x, intensity)

    _noise(image, rng)
    return image


def _vertical(rng):
    image = _base_image(rng)
    x = int(rng.integers(1, IMAGE_SIZE - 1))
    thickness = int(rng.integers(1, 3))
    start = int(rng.integers(0, 2))
    end = int(rng.integers(IMAGE_SIZE - 1, IMAGE_SIZE + 1))
    intensity = int(rng.integers(90, 128))

    for dx in range(thickness):
        col = min(x + dx, IMAGE_SIZE - 1)
        for y in range(start, min(end, IMAGE_SIZE)):
            _paint(image, y, col, intensity)

    _noise(image, rng)
    return image


def _diagonal(rng):
    image = _base_image(rng)
    descending = bool(rng.integers(0, 2))
    offset = int(rng.integers(-1, 2))
    thickness = int(rng.integers(1, 3))
    intensity = int(rng.integers(90, 128))

    for x in range(IMAGE_SIZE):
        y = (IMAGE_SIZE - 1 - x if descending else x) + offset
        for t in range(thickness):
            _paint(image, y + t, x, intensity)

    _noise(image, rng)
    return image


def _blob(rng):
    image = _base_image(rng)
    cy = float(rng.uniform(2.0, 5.0))
    cx = float(rng.uniform(2.0, 5.0))
    radius = float(rng.uniform(1.3, 2.4))
    intensity = int(rng.integers(90, 128))

    for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
            if (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2:
                jitter = int(rng.integers(-8, 9))
                _paint(image, y, x, max(55, min(127, intensity + jitter)))

    _noise(image, rng)
    return image


GENERATORS = (_horizontal, _vertical, _diagonal, _blob)


def make_split(rng, samples_per_class):
    images = []
    labels = []

    for label, generator in enumerate(GENERATORS):
        for _ in range(samples_per_class):
            images.append(generator(rng))
            labels.append(label)

    x = np.stack(images).reshape(-1, IMAGE_SIZE * IMAGE_SIZE)
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(y))
    return x[order], y[order]


def validate_split(name, x, y, samples_per_class):
    expected = samples_per_class * len(CLASS_NAMES)
    assert x.shape == (expected, 64)
    assert y.shape == (expected,)
    assert x.dtype == np.uint8
    assert y.dtype == np.int64
    assert int(x.min()) >= 0
    assert int(x.max()) <= 127
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    assert np.all(counts == samples_per_class), f"{name}: {counts.tolist()}"


def render_ascii(flat_image):
    image = flat_image.reshape(IMAGE_SIZE, IMAGE_SIZE)
    lines = []
    for row in image:
        lines.append("".join("#" if v >= 75 else "+" if v >= 30 else "." for v in row))
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic 8x8 tile data.")
    parser.add_argument("--output", type=Path, default=Path("data/generated/tile_dataset.npz"))
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--val-per-class", type=int, default=250)
    parser.add_argument("--test-per-class", type=int, default=250)
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.train_per_class, args.val_per_class, args.test_per_class) <= 0:
        raise SystemExit("All split sizes must be greater than zero.")

    rng = np.random.default_rng(args.seed)

    train_x, train_y = make_split(rng, args.train_per_class)
    val_x, val_y = make_split(rng, args.val_per_class)
    test_x, test_y = make_split(rng, args.test_per_class)

    validate_split("train", train_x, train_y, args.train_per_class)
    validate_split("validation", val_x, val_y, args.val_per_class)
    validate_split("test", test_x, test_y, args.test_per_class)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
        class_names=CLASS_NAMES,
        image_size=np.int64(IMAGE_SIZE),
        seed=np.int64(args.seed),
    )

    print(f"Wrote {args.output}")
    print(f"Samples: train={len(train_y)}, validation={len(val_y)}, test={len(test_y)}")
    print("Shape: 8x8 (64 values), range 0..127")
    print("Classes: " + ", ".join(CLASS_NAMES.tolist()))
    print(f"Seed: {args.seed}")

    if not args.no_preview:
        print("\nSample preview:")
        for class_id, class_name in enumerate(CLASS_NAMES):
            idx = int(np.flatnonzero(train_y == class_id)[0])
            print(f"\n[{class_name}]")
            print(render_ascii(train_x[idx]))


if __name__ == "__main__":
    main()
