#!/usr/bin/env bash
set -euo pipefail

NUM_PROCESSES="${1:-8}"
shift || true

uv run accelerate launch --config_file accelerate_configs/multi_gpu.yaml --num_processes "${NUM_PROCESSES}" train.py "$@"
