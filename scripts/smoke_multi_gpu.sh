#!/usr/bin/env bash
set -euo pipefail

NUM_PROCESSES="${1:-2}"
shift || true

uv run accelerate launch --config_file accelerate_configs/multi_gpu.yaml --num_processes "${NUM_PROCESSES}" train.py \
  game=pong \
  trainer.total_epochs=1 \
  collection_schedule.collection_limit=40 \
  collection_schedule.collection_per_epoch=20 \
  "$@"
