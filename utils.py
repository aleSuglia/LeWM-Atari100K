from pathlib import Path

import torch
from stable_baselines3 import PPO, DQN, A2C

from omegaconf import OmegaConf
import wandb
from wandb.integration.sb3 import WandbCallback

def get_agent(cfg, env, tb_logs, device):
    """
    Get agent based on config
    """
    valid_algs = {
        'ppo': PPO,
        'dqn': DQN,
        'a2c': A2C,
    }

    if cfg.algorithm not in valid_algs:
        raise KeyError(f"Unknown policy algorithm '{cfg.algorithm}'. Expected one of {sorted(valid_algs)}")

    alg_cls = valid_algs[cfg.algorithm]
    agent = alg_cls(
        policy=cfg.policy,
        env=env,
        tensorboard_log=tb_logs,
        device=device,
    )

    return agent

def build_optimizer(parameters, optimizer_cfg):
    """
    """
    optim_type = str(getattr(optimizer_cfg, "type", "AdamW")).lower()
    lr = float(getattr(optimizer_cfg, "lr", 3e-4))
    weight_decay = float(getattr(optimizer_cfg, "weight_decay", 1e-3))

    optimizers = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }

    if optim_type not in optimizers:
        raise KeyError(f"Unknown optimizer '{optim_type}'. Expected one of {sorted(optimizers)}")

    return optimizers[optim_type](
        parameters,
        lr=lr,
        weight_decay=weight_decay,
    )

def try_wandb_init(cfg):
    """
    """
    if cfg.local.wandb.enabled:
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        run =  wandb.init(
            project=cfg.local.wandb.project,
            entity=cfg.local.wandb.entity,
            config=config_dict,
            sync_tensorboard=True,
        )

        agent_wandb_callback = WandbCallback(log='all', verbose=1)

        return run, agent_wandb_callback
    else:
        return None, None

def log_wandb(run, metrics, global_epoch):
    """
    """
    if run is not None and metrics:
        run.log(metrics, step=global_epoch)