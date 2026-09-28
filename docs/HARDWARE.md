# Hardware notes

Primary target: Amiga 1200-class hardware.

The first hardware milestone is simply to run the same deterministic inference
vectors used by the host test harness and confirm identical scores.

## Practical constraints

The experiment assumes a constrained environment:

- 68k CPU;
- no GPU or accelerator requirement;
- small memory budget;
- integer-first arithmetic;
- minimal runtime dependencies.

Fast RAM or accelerators may be explored later, but the initial design should
not require them.

## Current hardware status

The target machine has recently been recapped as preventative maintenance.
Hands-on testing is temporarily inconvenient because a keyboard key was damaged
during transport and now behaves as if it is continuously pressed.

For that reason, the repository currently emphasizes host-side reproducibility
and a clean path toward later real-hardware validation.

## Measurements to capture later

When the target is usable again, record:

- compiler and optimization flags;
- executable size;
- free memory before and after launch;
- inference time over repeated runs;
- results for all golden test vectors;
- exact hardware configuration.
