import torch
from stable_baselines3 import PPO, DQN, A2C


def get_agent(cfg, env, device):
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
        device=device,
    )

    return agent

def _build_optimizer(parameters, optimizer_cfg):
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

