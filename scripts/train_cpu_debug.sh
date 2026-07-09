#!/usr/bin/env bash
set -euo pipefail

uv run accelerate launch --config_file accelerate_configs/cpu.yaml train.py mode=cpu_debug "$@"
