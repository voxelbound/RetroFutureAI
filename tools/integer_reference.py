#!/usr/bin/env python3
"""Reference implementation of RetroFutureAI integer inference.

This module mirrors the intended semantics of the portable C runtime. It is
not optimized for speed. Its purpose is to provide a readable and explicit
reference that later implementations can be compared against.

The long-term validation chain is:

    Python integer reference
            ==
    portable C runtime
            ==
    m68k C
            ==
    optimized target kernels

Keeping this implementation simple makes subtle differences in rounding,
saturation, memory layout, and tie-breaking easier to detect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


INT8_MIN = -128
INT8_MAX = 127
INT32_MIN = -2_147_483_648
INT32_MAX = 2_147_483_647


def _as_i8_vector(values, name: str) -> np.ndarray:
    """Convert input data to a one-dimensional int8 NumPy array.

    This helper keeps shape validation in one place so the public reference
    functions can focus on the actual inference semantics.

    Args:
        values: Array-like input data.
        name: Human-readable parameter name used in error messages.

    Returns:
        One-dimensional NumPy array with dtype int8.

    Raises:
        ValueError: If the resulting array is not one-dimensional.
    """
    array = np.asarray(values, dtype=np.int8)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int8 vector.")

    return array


def _as_i32_vector(values, name: str) -> np.ndarray:
    """Convert input data to a one-dimensional int32 NumPy array.

    Args:
        values: Array-like input data.
        name: Human-readable parameter name used in error messages.

    Returns:
        One-dimensional NumPy array with dtype int32.

    Raises:
        ValueError: If the resulting array is not one-dimensional.
    """
    array = np.asarray(values, dtype=np.int32)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional int32 vector.")

    return array


def dense_i8(
    input_values,
    weights,
    bias,
) -> np.ndarray:
    """Compute a fully connected layer using int8 inputs and weights.

    This mirrors ``rf_dense_i8()`` in the C runtime.

    Each output is calculated as:

        bias[o] + sum(input[i] * weight[o, i])

    Inputs and weights are signed 8-bit integers, while products are
    accumulated in signed 32-bit integers.

    The weight matrix uses the same logical layout as the C runtime:

        weights[output][input]

    Args:
        input_values: One-dimensional int8 input vector.
        weights: Two-dimensional int8 matrix with shape
            ``(output_size, input_size)``.
        bias: One-dimensional int32 bias vector with one value per output.

    Returns:
        One-dimensional int32 vector containing one accumulator per output.

    Raises:
        ValueError: If the input, weight, or bias dimensions do not match.
        OverflowError: If an accumulator exceeds signed 32-bit range.
    """
    x = _as_i8_vector(input_values, "input_values")
    w = np.asarray(weights, dtype=np.int8)
    b = _as_i32_vector(bias, "bias")

    if w.ndim != 2:
        raise ValueError("weights must be a two-dimensional int8 matrix.")

    output_size, input_size = w.shape

    if x.shape[0] != input_size:
        raise ValueError(
            f"input size mismatch: got {x.shape[0]}, expected {input_size}."
        )

    if b.shape[0] != output_size:
        raise ValueError(
            f"bias size mismatch: got {b.shape[0]}, expected {output_size}."
        )

    output = np.empty(output_size, dtype=np.int32)

    # Keep the arithmetic as explicit Python loops rather than using NumPy
    # matrix multiplication. That makes the implementation resemble the C
    # reference and avoids hiding integer-conversion details inside NumPy.
    for o in range(output_size):
        acc = int(b[o])

        for i in range(input_size):
            acc += int(x[i]) * int(w[o, i])

        # The C runtime stores the accumulator in int32. Raising here makes an
        # overflow visible instead of silently wrapping and masking a problem.
        if acc < INT32_MIN or acc > INT32_MAX:
            raise OverflowError(
                "dense accumulator exceeded signed 32-bit range."
            )

        output[o] = np.int32(acc)

    return output


def relu_i32(values) -> np.ndarray:
    """Apply ReLU to signed 32-bit accumulator values.

    ReLU is defined as:

        ReLU(x) = max(0, x)

    The operation returns a new array rather than modifying the caller's input.
    This differs from the C function's in-place API, but the numeric result is
    intentionally identical.

    Args:
        values: One-dimensional int32 vector.

    Returns:
        New int32 vector where negative values have been replaced by zero.
    """
    x = _as_i32_vector(values, "values")
    output = x.copy()
    output[output < 0] = 0
    return output


def _truncate_toward_zero(value: int, divisor: int) -> int:
    """Divide two positive-divisor integers using truncation toward zero.

    Python's ``//`` operator rounds negative results toward negative infinity,
    while signed integer division in C99 truncates toward zero. This helper
    makes Python reproduce the C behavior explicitly.

    Example:

        7 / 4  ->  1
       -7 / 4  -> -1

    Args:
        value: Signed integer numerator.
        divisor: Positive integer divisor.

    Returns:
        Integer quotient truncated toward zero.
    """
    if value >= 0:
        return value // divisor

    return -(abs(value) // divisor)


def requantize_i32_to_i8(values, shift: int) -> np.ndarray:
    """Convert int32 accumulators to compact int8 activations.

    This mirrors ``rf_requantize_i32_to_i8()`` in the current C runtime.

    For normal shifts, each value is divided by ``2**shift``. Division uses
    truncation toward zero. The result is then saturated to the signed int8
    range:

        -128 .. 127

    Saturation is important because a plain narrowing conversion could wrap
    values around instead of clipping them.

    The ``shift >= 31`` behavior intentionally mirrors the current C runtime,
    which uses ``INT32_MAX`` as the divisor. This is an implementation detail
    we may tighten later, but the Python reference must match the C semantics
    that exist today.

    Args:
        values: One-dimensional int32 vector.
        shift: Non-negative power-of-two scaling exponent.

    Returns:
        One-dimensional int8 vector.

    Raises:
        ValueError: If shift is negative.
    """
    x = _as_i32_vector(values, "values")

    if shift < 0:
        raise ValueError("shift must be non-negative.")

    if shift == 0:
        divisor = 1
    elif shift >= 31:
        divisor = INT32_MAX
    else:
        divisor = 1 << shift

    output = np.empty(x.shape[0], dtype=np.int8)

    for i, raw in enumerate(x):
        value = int(raw)

        if shift != 0:
            value = _truncate_toward_zero(value, divisor)

        # Saturate before converting to int8. This mirrors the explicit clamp
        # in the portable C runtime and prevents wraparound.
        value = max(INT8_MIN, min(INT8_MAX, value))
        output[i] = np.int8(value)

    return output


def argmax_i32(values) -> int:
    """Return the index of the largest signed 32-bit value.

    This mirrors ``rf_argmax_i32()``.

    Tie-breaking is intentionally deterministic: the first maximum wins.
    An empty vector returns ``-1``.

    Args:
        values: One-dimensional int32 vector.

    Returns:
        Index of the first maximum value, or ``-1`` for an empty vector.
    """
    x = _as_i32_vector(values, "values")

    if x.shape[0] == 0:
        return -1

    best = 0

    # Use '>' rather than '>=' so a later equal value does not replace the
    # first maximum. This matches the C implementation exactly.
    for i in range(1, x.shape[0]):
        if int(x[i]) > int(x[best]):
            best = i

    return best


@dataclass(frozen=True)
class IntegerInferenceResult:
    """Intermediate and final values from a two-layer integer inference run.

    Keeping the intermediate tensors is useful for future golden-vector tests.
    If Python and C disagree, we can compare each stage independently instead
    of only seeing that the final class differs.
    """

    hidden_accumulator: np.ndarray
    hidden_relu: np.ndarray
    hidden_i8: np.ndarray
    logits: np.ndarray
    best_class: int


def infer_two_layer(
    input_values,
    weights1,
    bias1,
    requant_shift: int,
    weights2,
    bias2,
) -> IntegerInferenceResult:
    """Run the intended two-layer RetroFutureAI integer inference path.

    The sequence is:

        int8 input
            ↓
        Dense
            ↓
        int32 accumulator
            ↓
        ReLU
            ↓
        Requantize
            ↓
        int8 hidden activation
            ↓
        Dense
            ↓
        int32 class scores
            ↓
        Argmax

    This function is intentionally generic enough to test the future
    ``64 -> 16 -> 4`` tile classifier without hard-coding those dimensions.

    Args:
        input_values: int8 input vector.
        weights1: First-layer int8 weight matrix.
        bias1: First-layer int32 bias vector.
        requant_shift: Scaling exponent between the two dense layers.
        weights2: Second-layer int8 weight matrix.
        bias2: Second-layer int32 bias vector.

    Returns:
        IntegerInferenceResult containing all important intermediate values.
    """
    hidden_accumulator = dense_i8(input_values, weights1, bias1)
    hidden_relu = relu_i32(hidden_accumulator)
    hidden_i8 = requantize_i32_to_i8(hidden_relu, requant_shift)
    logits = dense_i8(hidden_i8, weights2, bias2)
    best_class = argmax_i32(logits)

    return IntegerInferenceResult(
        hidden_accumulator=hidden_accumulator,
        hidden_relu=hidden_relu,
        hidden_i8=hidden_i8,
        logits=logits,
        best_class=best_class,
    )


def _test_dense_demo() -> None:
    """Check dense arithmetic against the existing synthetic C demo."""
    demo_input = np.array([8, 2, -3, 1], dtype=np.int8)
    demo_weights = np.array(
        [
            [8, 3, -2, 1],
            [-4, 6, 5, -3],
            [1, -2, 2, 7],
        ],
        dtype=np.int8,
    )
    demo_bias = np.array([0, 0, 0], dtype=np.int32)

    scores = dense_i8(demo_input, demo_weights, demo_bias)

    np.testing.assert_array_equal(
        scores,
        np.array([77, -38, 5], dtype=np.int32),
    )
    assert argmax_i32(scores) == 0


def _test_relu() -> None:
    """Check negative, zero, and positive ReLU behavior."""
    np.testing.assert_array_equal(
        relu_i32(np.array([-5, 0, 7], dtype=np.int32)),
        np.array([0, 0, 7], dtype=np.int32),
    )


def _test_requantize() -> None:
    """Check scaling, truncation toward zero, and int8 saturation."""
    np.testing.assert_array_equal(
        requantize_i32_to_i8(
            np.array([7, 8, -7, -8, 1000, -1000], dtype=np.int32),
            shift=2,
        ),
        np.array([1, 2, -1, -2, 127, -128], dtype=np.int8),
    )


def _test_argmax() -> None:
    """Check deterministic tie handling and the empty-input case."""
    assert argmax_i32(np.array([4, 9, 9, 2], dtype=np.int32)) == 1
    assert argmax_i32(np.array([], dtype=np.int32)) == -1


def _test_two_layer_inference() -> None:
    """Check a complete small two-layer integer inference path."""
    x = np.array([2, -1], dtype=np.int8)

    weights1 = np.array(
        [
            [3, 2],
            [-4, 1],
        ],
        dtype=np.int8,
    )
    bias1 = np.array([0, 1], dtype=np.int32)

    weights2 = np.array(
        [
            [2, -1],
            [-1, 3],
        ],
        dtype=np.int8,
    )
    bias2 = np.array([0, 0], dtype=np.int32)

    result = infer_two_layer(
        x,
        weights1,
        bias1,
        requant_shift=1,
        weights2=weights2,
        bias2=bias2,
    )

    np.testing.assert_array_equal(
        result.hidden_accumulator,
        np.array([4, -8], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.hidden_relu,
        np.array([4, 0], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        result.hidden_i8,
        np.array([2, 0], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        result.logits,
        np.array([4, -2], dtype=np.int32),
    )
    assert result.best_class == 0


def _self_test() -> None:
    """Run the built-in integer-reference sanity checks."""
    _test_dense_demo()
    _test_relu()
    _test_requantize()
    _test_argmax()
    _test_two_layer_inference()

    print("all integer reference tests passed")


if __name__ == "__main__":
    _self_test()
