# LeWM-Atari100K

## Training

This project now supports distributed multi-GPU training with Hugging Face Accelerate.

### 1) Install dependencies

```bash
uv sync
```

### 2) Use one of the provided Accelerate profiles

You can use the prebuilt config files in `accelerate_configs/`:

- `cpu.yaml`: CPU-only debug mode
- `single_gpu.yaml`: one GPU
- `multi_gpu.yaml`: multi-GPU on one node (default `num_processes: 4`)

Optional: generate your own config interactively.

```bash
uv run accelerate config
```

### 3) Launch training

CPU-only (local debugging, no GPU required):

```bash
uv run accelerate launch --config_file accelerate_configs/cpu.yaml train.py trainer.precision=no
```

Fast CPU debug mode (short schedule for quick iteration):

```bash
uv run accelerate launch --config_file accelerate_configs/cpu.yaml train.py mode=cpu_debug
```

Standard full schedule with CPU profile:

```bash
uv run accelerate launch --config_file accelerate_configs/cpu.yaml train.py mode=standard
```

Single GPU:

```bash
uv run accelerate launch --config_file accelerate_configs/single_gpu.yaml train.py
```

Multi-GPU (uses `accelerate_configs/multi_gpu.yaml`):

```bash
uv run accelerate launch --config_file accelerate_configs/multi_gpu.yaml train.py
```

For a different GPU count, either edit `num_processes` in the YAML file or override it at launch:

```bash
uv run accelerate launch --config_file accelerate_configs/multi_gpu.yaml --num_processes 8 train.py
```

Hydra overrides still work when appended to the command, for example:

```bash
uv run accelerate launch --num_processes 4 train.py game=breakout trainer.total_epochs=200
```