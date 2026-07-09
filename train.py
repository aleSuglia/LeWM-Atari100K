import json
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

import stable_pretraining
from lewm.imagination import ImaginationEnv
from lewm.modules import DistributedSIGReg
from utils import build_optimizer, log_wandb, try_wandb_init

# Keep module import for side effects (model/registry registration).
_ = stable_pretraining


def _normalize_mixed_precision(precision):
    value = str(precision).lower()
    if value in {"bf16", "bf16-mixed"}:
        return "bf16"
    if value in {"fp16", "16", "16-mixed"}:
        return "fp16"
    return "no"


@torch.no_grad()
def collect_real_interactions(
    num_interactions,
    is_random,
    obs,
    agent,
    memory,
    world_model,
    env,
    writer,
    device,
):
    """
    Collect given number of real interactions.
    """
    agent.eval()
    world_model.eval()

    agent.set_memory(memory)

    for _ in range(num_interactions):
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)

        emb = world_model.encode(obs_tensor).squeeze(1)
        if not is_random:
            action, memory = agent.act(
                emb,
                deterministic=False,
            )
        else:
            action = random.randint(0, env.num_actions - 1)

        next_obs, reward, terminated, truncated, _ = env.step(int(action))

        done = int(terminated or truncated)
        writer.append(
            obs_tensor.cpu().numpy(),
            action,
            reward,
            done,
        )

        obs = next_obs

        if terminated or truncated:
            obs, _ = env.reset()
            memory = agent.reset(1)

    writer.flush()
    agent.clear()

    return obs, memory


def train_world_model(
    world_model,
    dataset,
    loader_cfg,
    trainer_cfg,
    loss_weights,
    history_size,
    optimizer,
    accelerator,
):
    """
    Train world model for one pass over refreshed replay data.
    """
    dataset.refresh()

    world_model.train()

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, drop_last=True, **loader_cfg
    )
    dataloader = accelerator.prepare(dataloader)

    global_loss_tracker = None
    num_total_batches = 0

    for batch in dataloader:
        observations = batch["obs"].float()
        actions = batch["action"].long()
        rewards = batch["reward"].float()
        dones = batch["done"].float()

        optimizer.zero_grad()

        with accelerator.autocast():
            losses = world_model.loss(
                observations,
                actions,
                rewards,
                dones,
                loss_weights,
                history_size,
            )
            total_loss = losses["total_loss"]

        accelerator.backward(total_loss)

        clip_val = trainer_cfg.gradient_clip_val
        if clip_val is not None and clip_val > 0:
            accelerator.clip_grad_norm_(world_model.parameters(), clip_val)

        optimizer.step()

        reduced_losses = {
            name: accelerator.gather_for_metrics(value.detach()).mean()
            for name, value in losses.items()
        }
        detached_losses = {
            name: float(value.cpu()) for name, value in reduced_losses.items()
        }

        if global_loss_tracker is None:
            global_loss_tracker = {name: 0.0 for name in detached_losses}

        for name, value in detached_losses.items():
            global_loss_tracker[name] += value

        num_total_batches += 1

    if num_total_batches == 0 or global_loss_tracker is None:
        return {}

    return {
        name: value / num_total_batches for name, value in global_loss_tracker.items()
    }


def train_agent(
    agent,
    optimizer,
    trainer_cfg,
    imagination_env,
    accelerator,
):
    """
    Train policy on imagined trajectories for one epoch.
    """
    agent.train()
    imagination_env.world_model.eval()

    metric_tracker = None
    num_steps = 0

    for _ in range(trainer_cfg.steps_per_epoch):
        rollout = agent.imagine(imagination_env)
        loss, metrics = agent.loss(rollout, trainer_cfg)

        optimizer.zero_grad()
        accelerator.backward(loss)

        clip_val = trainer_cfg.gradient_clip_val
        if clip_val is not None and clip_val > 0:
            accelerator.clip_grad_norm_(agent.parameters(), clip_val)

        optimizer.step()
        agent.clear()

        reduced_metrics = {
            name: accelerator.gather_for_metrics(value.detach()).mean()
            for name, value in metrics.items()
        }
        detached_metrics = {
            name: float(value.cpu())
            for name, value in reduced_metrics.items()
        }

        if metric_tracker is None:
            metric_tracker = {
                name: 0.0 for name in detached_metrics
            }

        for name, value in detached_metrics.items():
            metric_tracker[name] += value

        num_steps += 1

    agent.clear()

    if num_steps == 0 or metric_tracker is None:
        return {}

    return {
        name: value / num_steps
        for name, value in metric_tracker.items()
    }


def eval_agent(
    episodes,
    per_episode_limit,
    agent,
    world_model,
    env_cfg,
    device,
    seed,
    at_end=False,
    eval_path=None,
):
    """
    Evaluate policy in real environment.
    """
    world_model.eval()
    agent.eval()

    reward_history = []
    length_history = []

    eval_env = hydra.utils.instantiate(env_cfg)

    for ep_idx in range(episodes):
        _ = agent.reset(1)
        obs, _ = eval_env.reset(seed + ep_idx)

        done = False
        total_return = 0.0
        length = 0

        while not done:
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = world_model.encode(obs_tensor).squeeze(1)
                action, _ = agent.act(
                    emb,
                    deterministic=True,
                )

            obs, reward, terminated, truncated, _ = eval_env.step(int(action))

            done = terminated or truncated

            total_return += reward
            length += 1

            if length >= per_episode_limit:
                done = True

        reward_history.append(total_return)
        length_history.append(length)

    agent.clear()
    eval_env.close()
    result = {
        "episodes": episodes,
        "ep_rewards": reward_history,
        "ep_lengths": length_history,
        "mean_reward": np.mean(reward_history),
        "std_reward": np.std(reward_history),
        "mean_length": np.mean(length_history),
    }

    if at_end:
        if eval_path is None:
            raise ValueError("eval_path must be provided when at_end=True")
        output_path = Path(eval_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


@hydra.main(version_base=None, config_path="./config", config_name="config")
def run(cfg):
    accelerator = Accelerator(
        mixed_precision=_normalize_mixed_precision(cfg.trainer.precision)
    )
    device = accelerator.device

    # Keep Hydra device fields aligned with Accelerate runtime selection.
    OmegaConf.update(cfg, "device", str(device), merge=False)
    OmegaConf.update(cfg, "agent.device", str(device), merge=False)

    # On CPU/MPS, keep HDF5 reads in-process to avoid file-lock errors from
    # DataLoader worker subprocesses opening the replay file concurrently.
    if device.type in {"cpu", "mps"}:
        OmegaConf.update(cfg, "loader.num_workers", 0, merge=False)
        OmegaConf.update(cfg, "loader.persistent_workers", False, merge=False)
        OmegaConf.update(cfg, "loader.pin_memory", False, merge=False)
        OmegaConf.update(cfg, "loader.prefetch_factor", None, merge=False)

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed + accelerator.process_index)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed + accelerator.process_index)
        torch.cuda.manual_seed_all(cfg.seed + accelerator.process_index)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Atari Env
    atari_env = hydra.utils.instantiate(cfg.env)
    num_actions = atari_env.num_actions

    OmegaConf.update(
        cfg,
        "model.action_encoder.num_actions",
        int(num_actions),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "agent.num_actions",
        int(num_actions),
        merge=False,
    )

    # W&B
    wandb_run = try_wandb_init(cfg) if accelerator.is_main_process else None

    # World Model
    world_model = hydra.utils.instantiate(cfg.model)

    # Keep original SIGReg on CPU/single-device runs; use distributed SIGReg
    # when more than one CUDA device is available.
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        base_sigreg = world_model.sigreg
        world_model.sigreg = DistributedSIGReg(
            knots=int(base_sigreg.t.numel()),
            num_proj=int(base_sigreg.num_proj),
        )

    # Replay Writer (single-process write to avoid HDF5 corruption)
    replay_writer = (
        hydra.utils.instantiate(cfg.replay) if accelerator.is_main_process else None
    )
    accelerator.wait_for_everyone()

    # Dataset
    dataset = hydra.utils.instantiate(cfg.dataset)

    # Imagination Env
    imagination_env = ImaginationEnv(
        world_model=world_model,
        dataset=dataset,
        **cfg.imagination,
    )

    # Agent
    agent = hydra.utils.instantiate(cfg.agent)

    # Optimizers
    wm_optimizer = build_optimizer(
        world_model.parameters(),
        cfg.trainer.optimizer,
    )
    agent_optimizer = build_optimizer(
        agent.parameters(),
        cfg.agent_trainer.optimizer,
    )

    # Prepare distributed objects
    world_model, agent, wm_optimizer, agent_optimizer = accelerator.prepare(
        world_model,
        agent,
        wm_optimizer,
        agent_optimizer,
    )

    if hasattr(agent, "module"):
        agent.module.device = device
    else:
        agent.device = device

    #########################
    ##      Training       ##
    #########################

    obs = None
    memory = None
    if accelerator.is_main_process:
        obs, _ = atari_env.reset(seed=cfg.seed)
        agent_for_env = accelerator.unwrap_model(agent)
        memory = agent_for_env.reset(1)

    # Tracking values
    total_collected_interactions = 0
    collection_per_epoch = cfg.collection_schedule.collection_per_epoch

    # Creating directories for checkpointing
    wm_ckp_dir = Path(cfg.checkpointing.wm_path)
    wm_ckp_dir.mkdir(parents=True, exist_ok=True)

    agent_ckp_dir = Path(cfg.checkpointing.agent_path)
    agent_ckp_dir.mkdir(parents=True, exist_ok=True)

    # Training Loop
    for epoch_idx in range(cfg.trainer.total_epochs):
        epoch_num = epoch_idx + 1
        if accelerator.is_main_process:
            print("running epoch: ", epoch_num)

        # Collection
        is_random = epoch_num < cfg.collection_schedule.random_collection_epochs

        if total_collected_interactions < cfg.collection_schedule.collection_limit:
            total_collected_interactions += collection_per_epoch

            if accelerator.is_main_process:
                assert replay_writer is not None
                agent_for_env = accelerator.unwrap_model(agent)
                wm_for_env = accelerator.unwrap_model(world_model)
                obs, memory = collect_real_interactions(
                    num_interactions=collection_per_epoch,
                    is_random=is_random,
                    obs=obs,
                    agent=agent_for_env,
                    memory=memory,
                    world_model=wm_for_env,
                    env=atari_env,
                    writer=replay_writer,
                    device=device,
                )

                log_wandb(
                    wandb_run,
                    {"replay/collection_size": replay_writer.size},
                    epoch_idx,
                )

                print("collection complete. size: ", replay_writer.size)

            accelerator.wait_for_everyone()

        # Training World Model
        wm_train_epochs = None
        if cfg.wm_schedule.start_epoch <= epoch_num < cfg.wm_schedule.periodic_start_epoch:
            wm_train_epochs = cfg.wm_schedule.regular_epochs
        elif (
            cfg.wm_schedule.periodic_start_epoch <= epoch_num <= cfg.wm_schedule.stop_epoch
            and epoch_num % cfg.wm_schedule.period == 0
        ):
            wm_train_epochs = cfg.wm_schedule.periodic_epochs

        if wm_train_epochs is not None:
            wm_losses = {}
            for _ in range(wm_train_epochs):
                wm_losses = train_world_model(
                    world_model=world_model,
                    dataset=dataset,
                    loader_cfg=cfg.loader,
                    trainer_cfg=cfg.trainer,
                    loss_weights=cfg.loss_weights,
                    history_size=cfg.history_size,
                    optimizer=wm_optimizer,
                    accelerator=accelerator,
                )
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                log_wandb(
                    wandb_run,
                    {f"world_model/{key}": value for key, value in wm_losses.items()},
                    epoch_idx,
                )

                print("world model training complete. losses:")
                print(wm_losses)

        # Checkpointing World Model
        if (
            epoch_num % cfg.checkpointing.wm_per_epoch == 0
        ) and accelerator.is_main_process:
            file_name = f"epoch_{epoch_num}.pt"
            wm_ckp_path = wm_ckp_dir / file_name
            accelerator.save(
                accelerator.unwrap_model(world_model).state_dict(), wm_ckp_path
            )

        # Training Agent
        if epoch_num >= cfg.agent_trainer.agent_start_epoch:
            agent_losses = train_agent(
                agent=agent,
                optimizer=agent_optimizer,
                trainer_cfg=cfg.agent_trainer,
                imagination_env=imagination_env,
                accelerator=accelerator,
            )
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                log_wandb(
                    wandb_run,
                    {f"agent/{key}": value for key, value in agent_losses.items()},
                    epoch_idx,
                )

                print("agent training done. losses: ")
                print(agent_losses)

        # Checkpointing Agent
        if (
            epoch_num % cfg.checkpointing.agent_per_epoch == 0
        ) and accelerator.is_main_process:
            file_name = f"epoch_{epoch_num}.pt"
            agent_ckp_path = agent_ckp_dir / file_name
            accelerator.save(
                accelerator.unwrap_model(agent).state_dict(), agent_ckp_path
            )

        # Sanity Eval Checks
        if (
            epoch_num % cfg.sanity_eval.every_x_epoch == 0
        ) and accelerator.is_main_process:
            agent_for_eval = accelerator.unwrap_model(agent)
            wm_for_eval = accelerator.unwrap_model(world_model)
            sanity_eval = eval_agent(
                episodes=cfg.sanity_eval.episodes,
                per_episode_limit=cfg.sanity_eval.per_episode_limit,
                agent=agent_for_eval,
                world_model=wm_for_eval,
                env_cfg=cfg.env,
                device=device,
                seed=cfg.seed,
                at_end=False,
            )
            log_wandb(
                wandb_run,
                {
                    "sanity_eval/mean_rew": sanity_eval["mean_reward"],
                    "sanity_eval/mean_len": sanity_eval["mean_length"],
                },
                epoch_idx,
            )
            print("sanity eval complete. results:")
            print(sanity_eval)

    # Checkpointing Final World Model
    if accelerator.is_main_process:
        file_name = "final.pt"
        wm_ckp_path = wm_ckp_dir / file_name
        accelerator.save(
            accelerator.unwrap_model(world_model).state_dict(), wm_ckp_path
        )

    # Checkpointing Final Agent
    if accelerator.is_main_process:
        file_name = "final.pt"
        agent_ckp_path = agent_ckp_dir / file_name
        accelerator.save(accelerator.unwrap_model(agent).state_dict(), agent_ckp_path)

    #########################
    ##     Evaluation      ##
    #########################

    if accelerator.is_main_process:
        eval_agent(
            episodes=cfg.eval.episodes,
            per_episode_limit=cfg.eval.per_episode_limit,
            agent=accelerator.unwrap_model(agent),
            world_model=accelerator.unwrap_model(world_model),
            env_cfg=cfg.env,
            device=device,
            seed=cfg.seed,
            at_end=True,
            eval_path=cfg.eval.output_path,
        )

    atari_env.close()
    if replay_writer is not None:
        replay_writer.close()
    dataset.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    run()
