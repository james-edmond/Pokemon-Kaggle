# GPU training venv spike results

Date: 2026-07-04

## Environment

- Machine: Windows 10, GPU = NVIDIA GeForce GTX 1060 6GB (Pascal, sm_61)
- Base Python: 3.14 at `C:\Python314` (untouched)
- Training venv interpreter: Python 3.12.10, installed via
  `winget install -e --id Python.Python.3.12` (no 3.11/3.12/3.13 was present
  beforehand; winget install succeeded on the first attempt)
- venv location: `venv-train\` (gitignored)

## Torch decision ladder

Ladder rung 1 succeeded immediately — no need to try rungs 2 or 3.

| Rung | Build | Result |
|---|---|---|
| 1 | `torch==2.5.1+cu121` (index `https://download.pytorch.org/whl/cu121`) | **SUCCESS** — arch check printed `OK` |
| 2 | `torch==2.4.1+cu121` | not attempted (rung 1 succeeded) |
| 3 | `torch==2.3.1+cu118` | not attempted (rung 1 succeeded) |
| CPU fallback | `torch` (cpu index) | not needed |

Arch-check command and output:

```
venv-train\Scripts\python -c "import torch; assert torch.cuda.is_available(), 'no cuda'; assert 'sm_61' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list(); assert torch.cuda.get_device_capability(0) == (6, 1), torch.cuda.get_device_capability(0); print('OK')"
```
```
OK
```

Full detail:

```
torch version: 2.5.1+cu121
cuda available: True
arch list: ['sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
device capability: (6, 1)
device name: NVIDIA GeForce GTX 1060 6GB
cuda version (torch built with): 12.1
```

**Chosen torch build: `torch==2.5.1+cu121`. Arch list includes `sm_61`; device
capability (6, 1) confirmed for the GTX 1060.**

## Other venv packages

- numpy 2.5.1
- pytest 9.1.1
- matplotlib 3.11.0
- `ptcg` installed editable via `pip install -e . --no-deps` (no-deps used so
  pip does not try to satisfy `pyproject.toml`'s `torch>=2.4` pin by
  reinstalling a different torch build over the ladder-chosen one)

## Spike script output (`benchmarks/spike_gpu.py`)

```
torch 2.5.1+cu121 | device cuda | arch ['sm_50', 'sm_60', 'sm_61', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
matmul+backward ok
student encoder fwd+bwd b256: 707 ms
```

(A benign `UserWarning: enable_nested_tensor is True, but self.use_nested_tensor
is False because encoder_layer.norm_first was True` is emitted by
`torch.nn.TransformerEncoder`; this is the same warning already suppressed via
`filterwarnings` in `pyproject.toml`'s pytest config, so it's expected and not
a compat issue.)

## Phase-1 suite on the venv torch

Run: `venv-train\Scripts\python -m pytest tests/ -q` (single process, no
`-n`/xdist — the engine allows one battle per process)

```
37 passed in 30.23s
```

All 37 tests pass under torch 2.5.1+cu121 / Python 3.12.10, confirming the
phase-1 `ptcg` package is compatible with the older torch pinned by the
Pascal-capable ladder. (The brief's expected count of "37 passed" matches
exactly; there is no discrepancy to flag.)

## Decision

- `device: cuda`
- torch build: `torch==2.5.1+cu121`
- No CPU fallback needed.

## Concerns

- None blocking. The GTX 1060's compute capability (6, 1) is on the edge of
  what modern torch wheels support without warnings from other library
  versions (e.g. some torchvision/xformers builds drop sm_6x); if later tasks
  add more GPU-side dependencies, re-run the arch check after each one.
- The 707 ms/iteration encoder fwd+bwd timing at batch 256 is a first-pass
  spike number (2 warmup iterations only, no CUDA graph capture, default
  kernel selection); treat it as a rough baseline for later training-loop
  batch-size/throughput decisions, not a tuned number.
