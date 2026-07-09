# Scripts

All scripts use `uv` + `accelerate launch` and accept extra Hydra overrides at the end.

## Available scripts

- `train_cpu.sh`
  - CPU single-process training.
  - Usage: `./scripts/train_cpu.sh [hydra_overrides...]`
  - Example: `./scripts/train_cpu.sh mode=standard game=breakout`

- `train_cpu_debug.sh`
  - Fast local CPU debug profile (`mode=cpu_debug`).
  - Usage: `./scripts/train_cpu_debug.sh [hydra_overrides...]`
  - Example: `./scripts/train_cpu_debug.sh game=pong`

- `smoke_cpu.sh`
  - Very short CPU smoke test (1 epoch, tiny schedule).
  - Usage: `./scripts/smoke_cpu.sh [hydra_overrides...]`
  - Example: `./scripts/smoke_cpu.sh replay.path=./outputs/smoke/replay.h5`

- `train_mps.sh`
  - Apple Silicon MPS training (single process).
  - Usage: `./scripts/train_mps.sh [hydra_overrides...]`
  - Example: `./scripts/train_mps.sh game=breakout`

- `train_cuda_single.sh`
  - Single NVIDIA GPU training.
  - Usage: `./scripts/train_cuda_single.sh [hydra_overrides...]`
  - Example: `./scripts/train_cuda_single.sh trainer.total_epochs=200`

- `train_multi_gpu.sh`
  - Multi-GPU CUDA training.
  - Usage: `./scripts/train_multi_gpu.sh [num_processes] [hydra_overrides...]`
  - Default `num_processes`: `8`
  - Example: `./scripts/train_multi_gpu.sh 4 game=breakout trainer.total_epochs=200`

- `smoke_multi_gpu.sh`
  - Short multi-GPU CUDA smoke test.
  - Usage: `./scripts/smoke_multi_gpu.sh [num_processes] [hydra_overrides...]`
  - Default `num_processes`: `2`
  - Example: `./scripts/smoke_multi_gpu.sh 2`

- `train_cpu_multi_process.sh`
  - Accelerate CPU multi-process launch for MPI-capable environments (typically Linux).
  - Usage: `./scripts/train_cpu_multi_process.sh <mpirun_hostfile> [hydra_overrides...]`
  - Example: `./scripts/train_cpu_multi_process.sh ./hosts.txt`

## Notes

- On macOS, Accelerate CPU mode is often single-process.
- For true multi-rank local runs on macOS, prefer CUDA multi-GPU on a Linux/CUDA machine.
