#!/usr/bin/env bash
set -euo pipefail

uv run accelerate launch --config_file accelerate_configs/cuda_single_gpu.yaml train.py "$@"
