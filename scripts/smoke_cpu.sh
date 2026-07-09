#!/usr/bin/env bash
set -euo pipefail

uv run accelerate launch --config_file accelerate_configs/cpu.yaml train.py \
  game=pong \
  trainer.total_epochs=1 trainer.precision=no \
  collection_schedule.collection_limit=40 \
  collection_schedule.collection_per_epoch=40 \
  collection_schedule.random_collection_epochs=1 \
  wm_schedule.start_epoch=1 wm_schedule.regular_epochs=1 \
  wm_schedule.periodic_start_epoch=99999 wm_schedule.stop_epoch=1 \
  wm_schedule.steps_per_epoch=1 \
  loader.batch_size=4 loader.num_workers=0 \
  loader.persistent_workers=false loader.pin_memory=false loader.prefetch_factor=null \
  dataset.seq_len=8 dataset.num_batches=1 \
  agent_trainer.agent_start_epoch=999999 \
  sanity_eval.every_x_epoch=999999 \
  eval.episodes=1 eval.per_episode_limit=100 \
  "$@"
