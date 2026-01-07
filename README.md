# RetroFutureAI
Running AI inference on an Amiga 1200 - initial commit will be published once the implementation is stable and runnable.

## TODO

- [ ] Define the inference target (what model, what input/output, what “done” means)
- [X] Confirm execution environment (real A1200 vs emulator; AmigaOS vs other)
- [ ] Choose model format + conversion path (train/build elsewhere, then export to Amiga)
- [ ] Implement or port a minimal inference runtime (fixed-point where needed)
- [ ] Create a repeatable test harness with golden vectors (reference vs Amiga output)
- [ ] Benchmark latency and memory; optimize the hot paths
- [ ] Package a small demo + build instructions; decide license before first release
