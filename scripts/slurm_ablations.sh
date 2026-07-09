#!/usr/bin/env bash
# =============================================================================
# LeWM World-Model Ablations — SLURM array job
#
# Submits 14 independent runs (one GPU each) covering the key axes of the
# world-model design: SIGReg regularisation strength, individual loss terms,
# predictor history length, predictor depth, embedding dimension, and encoder
# patch size.
#
# Usage:
#   sbatch scripts/slurm_ablations.sh              # all 14 ablations
#   sbatch --array=0-3 scripts/slurm_ablations.sh  # subset
#
# Tune --partition, --time, --mem, and the WANDB_* variables below before
# submitting.
# =============================================================================

#SBATCH --job-name=lewm_abl
#SBATCH --array=0-13
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=YOUR_PARTITION       # <-- set your cluster partition
#SBATCH --output=slurm_logs/abl_%A_%a.out
#SBATCH --error=slurm_logs/abl_%A_%a.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Optional: W&B logging.  Set WANDB_PROJECT and WANDB_ENTITY to enable,
# or leave blank to keep W&B disabled (the default in log.yaml).
# ---------------------------------------------------------------------------
WANDB_PROJECT=""   # e.g. "lewm-ablations"
WANDB_ENTITY=""    # e.g. "your-team"

# ---------------------------------------------------------------------------
# Ablation definitions.
# NAMES   — short tag used for checkpoint dirs and eval output files.
# LABELS  — human-readable description printed in the SLURM log.
# OVERRIDES — space-separated Hydra CLI overrides (empty = baseline).
# ---------------------------------------------------------------------------
NAMES=(
    "baseline"          # 0
    "no_sigreg"         # 1
    "sigreg_low"        # 2
    "sigreg_high"       # 3
    "pred_loss_only"    # 4
    "no_rew"            # 5
    "rew_high"          # 6
    "no_don"            # 7
    "hist_3"            # 8
    "hist_8"            # 9
    "pred_shallow"      # 10
    "pred_deep"         # 11
    "embed_512"         # 12
    "patch_8"           # 13
)

LABELS=(
    "Baseline (all defaults)"
    "SIGReg disabled (sigreg_loss=0)"
    "SIGReg low weight (sigreg_loss=0.1)"
    "SIGReg high weight (sigreg_loss=1.0)"
    "Prediction loss only (sigreg/rew/don=0)"
    "No reward loss (rew_loss=0)"
    "Higher reward loss weight (rew_loss=0.1)"
    "No done loss (don_loss=0)"
    "Short history (history_size=3)"
    "Long history (history_size=8)"
    "Shallow predictor (depth=3)"
    "Deep predictor (depth=9)"
    "Small embedding (embed_dim=512)"
    "Large patch size (patch_size=8, 64 tokens)"
)

OVERRIDES=(
    ""
    "loss_weights.sigreg_loss=0.0"
    "loss_weights.sigreg_loss=0.1"
    "loss_weights.sigreg_loss=1.0"
    "loss_weights.sigreg_loss=0.0 loss_weights.rew_loss=0.0 loss_weights.don_loss=0.0"
    "loss_weights.rew_loss=0.0"
    "loss_weights.rew_loss=0.1"
    "loss_weights.don_loss=0.0"
    "history_size=3"
    "history_size=8"
    "model.predictor.depth=3"
    "model.predictor.depth=9"
    "embed_dim=512"
    "model.encoder.patch_size=8"
)

# ---------------------------------------------------------------------------
# Pick this task's ablation.
# ---------------------------------------------------------------------------
IDX=${SLURM_ARRAY_TASK_ID}
NAME=${NAMES[$IDX]}
LABEL=${LABELS[$IDX]}
EXTRA=${OVERRIDES[$IDX]}

echo "============================================================"
echo "SLURM array task : ${IDX}"
echo "Ablation name    : ${NAME}"
echo "Description      : ${LABEL}"
echo "Extra overrides  : ${EXTRA:-<none>}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Ensure log and output dirs exist.
# ---------------------------------------------------------------------------
mkdir -p slurm_logs
mkdir -p "checkpoints/pong_${NAME}/world_model"
mkdir -p "checkpoints/pong_${NAME}/agent"
mkdir -p eval

# ---------------------------------------------------------------------------
# Build W&B overrides (no-op if project is unset).
# ---------------------------------------------------------------------------
WANDB_OVERRIDES=""
if [[ -n "${WANDB_PROJECT}" ]]; then
    WANDB_OVERRIDES="local.wandb.enabled=true local.wandb.project=${WANDB_PROJECT}"
    if [[ -n "${WANDB_ENTITY}" ]]; then
        WANDB_OVERRIDES="${WANDB_OVERRIDES} local.wandb.entity=${WANDB_ENTITY}"
    fi
fi

# ---------------------------------------------------------------------------
# Launch training.
# Each run gets its own Hydra output dir, checkpoint path, and eval file so
# concurrent array tasks never clobber each other.
# ---------------------------------------------------------------------------
uv run accelerate launch \
    --config_file accelerate_configs/cuda_single_gpu.yaml \
    train.py \
    game=pong \
    "checkpointing.wm_path=checkpoints/pong_${NAME}/world_model" \
    "checkpointing.agent_path=checkpoints/pong_${NAME}/agent" \
    "eval.output_path=eval/pong_${NAME}.json" \
    "hydra.run.dir=outputs/ablations/${NAME}" \
    ${WANDB_OVERRIDES} \
    ${EXTRA}

echo "Ablation '${NAME}' finished."
