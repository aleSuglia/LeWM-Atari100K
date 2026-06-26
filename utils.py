import torch

from omegaconf import OmegaConf
import wandb

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
        )

        return run
    else:
        return None

def log_wandb(run, metrics, global_epoch):
    """
    """
    if run is not None and metrics:
        run.log(metrics, step=global_epoch)