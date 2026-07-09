#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <mpirun_hostfile> [train.py overrides ...]"
  exit 1
fi

HOSTFILE="$1"
shift

uv run accelerate launch --cpu --num_processes 2 --mpirun_hostfile "${HOSTFILE}" train.py "$@"
