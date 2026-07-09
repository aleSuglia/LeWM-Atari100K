#!/usr/bin/env bash
set -euo pipefail

uv run accelerate launch --config_file accelerate_configs/mps.yaml train.py "$@"
