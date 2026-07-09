# LeWM-Atari100K

## Training

This project now supports distributed multi-GPU training with Hugging Face Accelerate.

### 1) Install dependencies

```bash
uv sync
```

### 2) Use one of the provided Accelerate profiles

You can use the prebuilt config files in `accelerate_configs/`:

- `cpu.yaml`: CPU-only (single process)
- `cuda_single_gpu.yaml`: one NVIDIA CUDA GPU
- `mps.yaml`: Apple Silicon MPS (single process)
- `multi_gpu.yaml`: multi-GPU on one node (CUDA/NCCL environments)

Optional: generate your own config interactively.

```bash
uv run accelerate config
```

### 3) Launch training

You can run all common launch modes via scripts in `scripts/`.
For detailed per-script help and examples, see `scripts/README.md`.

Make scripts executable (already done in this repository):

```bash
chmod +x scripts/*.sh
```

CPU-only (single process):

```bash
./scripts/train_cpu.sh
```

Fast CPU debug mode (short schedule for quick iteration):

```bash
./scripts/train_cpu_debug.sh
```

Standard full schedule with CPU profile:

```bash
./scripts/train_cpu.sh mode=standard
```

CPU-only quick smoke test (single epoch/short collection):

```bash
./scripts/smoke_cpu.sh
```

CUDA single GPU:

```bash
./scripts/train_cuda_single.sh
```

Apple Silicon MPS:

```bash
./scripts/train_mps.sh
```

Multi-GPU (uses `accelerate_configs/multi_gpu.yaml`):

```bash
./scripts/train_multi_gpu.sh
```

For a different GPU count, either edit `num_processes` in the YAML file or override it at launch:

```bash
./scripts/train_multi_gpu.sh 8
```

Multi-GPU quick smoke test:

```bash
./scripts/smoke_multi_gpu.sh 2
```

CPU multi-process (Accelerate, Linux/MPI environments):

```bash
./scripts/train_cpu_multi_process.sh /path/to/hostfile
```

Notes:

- Treat "GPU" in this repository as CUDA unless MPS is explicitly named.
- On macOS/Apple Silicon, prefer `mps.yaml`.
- On macOS, Accelerate CPU runs are typically single-process. Use CUDA multi-GPU for true multi-rank local runs.
- For CPU multi-process with Accelerate, use an MPI-capable environment (typically Linux).

Hydra overrides still work when appended to the command, for example:

```bash
./scripts/train_multi_gpu.sh 4 game=breakout trainer.total_epochs=200
```