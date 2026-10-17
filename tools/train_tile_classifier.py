#!/usr/bin/env python3
"""Train the first RetroFutureAI tile classifier.

Architecture:
    64 -> 16 -> ReLU -> 4

Training is host-side only. The saved floating-point checkpoint is intended
for later quantization and export, not for direct execution on the Amiga.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


INPUT_SIZE = 64
HIDDEN_SIZE = 16
OUTPUT_SIZE = 4


class TileClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RetroFutureAI 8x8 tile classifier."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/generated/tile_dataset.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/generated/tile_classifier.pt"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested, but CUDA is not available.")
        return torch.device("cuda")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(path: Path):
    if not path.exists():
        raise SystemExit(
            f"Dataset not found: {path}\n"
            "Run: python tools/generate_tile_dataset.py"
        )

    with np.load(path) as data:
        required = {
            "train_x",
            "train_y",
            "val_x",
            "val_y",
            "test_x",
            "test_y",
            "class_names",
        }
        missing = required.difference(data.files)
        if missing:
            raise SystemExit(
                "Dataset is missing required arrays: "
                + ", ".join(sorted(missing))
            )

        arrays = {
            "train_x": data["train_x"].copy(),
            "train_y": data["train_y"].copy(),
            "val_x": data["val_x"].copy(),
            "val_y": data["val_y"].copy(),
            "test_x": data["test_x"].copy(),
            "test_y": data["test_y"].copy(),
        }
        class_names = [str(name) for name in data["class_names"]]

    for split in ("train", "val", "test"):
        x = arrays[f"{split}_x"]
        y = arrays[f"{split}_y"]

        if x.ndim != 2 or x.shape[1] != INPUT_SIZE:
            raise SystemExit(
                f"{split}_x has shape {x.shape}; expected (N, {INPUT_SIZE})."
            )

        if y.ndim != 1 or len(y) != len(x):
            raise SystemExit(
                f"{split}_y has shape {y.shape}; expected one label per sample."
            )

    return arrays, class_names


def to_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    x_tensor = torch.from_numpy(x.astype(np.float32) / 127.0)
    y_tensor = torch.from_numpy(y.astype(np.int64))
    return TensorDataset(x_tensor, y_tensor)


def make_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    count = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        total_loss += float(criterion(logits, y).item())
        correct += int((logits.argmax(dim=1) == y).sum().item())
        count += int(y.numel())

    return total_loss / count, correct / count


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise SystemExit("--epochs must be greater than zero.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero.")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be greater than zero.")

    set_deterministic(args.seed)
    device = choose_device(args.device)
    arrays, class_names = load_dataset(args.dataset)

    train_dataset = to_tensor_dataset(arrays["train_x"], arrays["train_y"])
    val_dataset = to_tensor_dataset(arrays["val_x"], arrays["val_y"])
    test_dataset = to_tensor_dataset(arrays["test_x"], arrays["test_y"])

    train_loader = make_loader(train_dataset, args.batch_size, True, args.seed)
    train_eval_loader = make_loader(
        train_dataset, args.batch_size, False, args.seed
    )
    val_loader = make_loader(val_dataset, args.batch_size, False, args.seed)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.seed)

    model = TileClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("RetroFutureAI tile classifier training")
    print(f"Architecture: {INPUT_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(
        f"Samples: train={len(train_dataset)}, "
        f"validation={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )
    print()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_count = int(y.numel())
            running_loss += float(loss.item()) * batch_count
            sample_count += batch_count

        train_loss = running_loss / sample_count
        val_loss, val_accuracy = evaluate(model, val_loader, device)

        print(
            f"epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy * 100:.2f}%"
        )

    train_loss, train_accuracy = evaluate(model, train_eval_loader, device)
    val_loss, val_accuracy = evaluate(model, val_loader, device)
    test_loss, test_accuracy = evaluate(model, test_loader, device)

    print()
    print("Final metrics")
    print(
        f"train:      loss={train_loss:.4f} "
        f"accuracy={train_accuracy * 100:.2f}%"
    )
    print(
        f"validation: loss={val_loss:.4f} "
        f"accuracy={val_accuracy * 100:.2f}%"
    )
    print(
        f"test:       loss={test_loss:.4f} "
        f"accuracy={test_accuracy * 100:.2f}%"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "architecture": {
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "output_size": OUTPUT_SIZE,
        },
        "class_names": class_names,
        "normalization": {
            "input_min": 0,
            "input_max": 127,
            "training_scale_divisor": 127.0,
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "metrics": {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        },
    }

    torch.save(checkpoint, args.output)
    print(f"\nWrote checkpoint: {args.output}")


if __name__ == "__main__":
    main()
