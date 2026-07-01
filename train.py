import json
from pathlib import Path

import random
import numpy as np
import torch

import hydra
from omegaconf import OmegaConf

from lewm.imagination import ImaginationEnv
from atari.env import AtariEnv
from utils import build_optimizer, try_wandb_init, log_wandb

def collect_real_interactions(
        num_interactions,
        is_random,
        obs,
        agent,
        world_model,
        env,
        writer,
        device
):
    """
    collect given number of real interactions
    """
    agent.eval()
    world_model.eval()
    
    needs_reset = False

    for _ in range(num_interactions):
        if needs_reset:
            obs, _ = env.reset()
        
        obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = world_model.encode(obs)
            if not is_random:
                action = agent.predict(
                    emb,
                    deterministic=False,
                )
            else:
                num_actions = env.num_actions
                action = random.randint(0, num_actions-1)

        next_obs, reward, terminated, truncated, _ = env.step(int(action))

        done = int(terminated or truncated)      
        writer.append(
            obs.cpu().numpy(),
            action,
            reward,
            done,
        )

        obs = next_obs
        needs_reset = terminated or truncated

    writer.flush()
    return obs

def train_world_model(
        world_model,
        dataset,
        loader_cfg,
        trainer_cfg,
        loss_weights,
        history_size,
        optimizer,
        device,
):
    """
    """
    dataset.refresh()

    world_model.to(device)
    world_model.train()

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        drop_last=True,
        **loader_cfg
    )

    global_loss_tracker = None
    num_total_batches = 0

    for batch in dataloader:
        observations = batch['obs'].to(device, non_blocking=True).float()
        actions = batch['action'].to(device, non_blocking=True).long()
        rewards = batch['reward'].to(device, non_blocking=True).float()
        dones = batch['done'].to(device, non_blocking=True).float()

        optimizer.zero_grad()

        with torch.autocast(
            device_type=device,
            dtype=torch.bfloat16,
            enabled=(trainer_cfg.precision == 'bf16-mixed')
        ):
            losses = world_model.loss(
                observations,
                actions,
                rewards,
                dones,
                loss_weights,
                history_size,
            )
            total_loss = losses['total_loss']
        
        total_loss.backward()

        clip_val = trainer_cfg.gradient_clip_val
        if clip_val is not None and clip_val > 0:
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), clip_val)

        optimizer.step()

        detached_losses = {
            name: float(value.detach().clone().cpu())
            for name, value in losses.items()
        }

        if global_loss_tracker is None:
            global_loss_tracker = {name: 0.0 for name in detached_losses}
        
        for name, value in detached_losses.items():
            global_loss_tracker[name] += value
        
        num_total_batches += 1
    
    return {
        name: value / num_total_batches
        for name, value in global_loss_tracker.items()
    }

def eval_agent(
        episodes,
        per_episode_limit,
        agent,
        world_model,
        env_cfg,
        device,
        at_end=False,
        eval_path=None,
):
    """
    """
    world_model.eval()

    reward_history = []
    length_history = []

    eval_env = hydra.utils.instantiate(env_cfg)

    for _ in range(episodes):
        obs, _ = eval_env.reset()

        done = False
        total_return  = 0.0
        length = 0

        while not done:
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = world_model.encode(obs)
                action = agent.predict(
                    emb,
                    deterministic=True,    # Exploitation
                )
            
            obs, reward, terminated, truncated, _ = eval_env.step(int(action))
            
            done = terminated or truncated

            total_return += reward
            length += 1

            if length >= per_episode_limit:
                done = True
        
        reward_history.append(total_return)
        length_history.append(length)

    eval_env.close()
    result = {
        "episodes": episodes,
        "ep_rewards": reward_history,
        "ep_lengths": length_history,
        "mean_reward": np.mean(reward_history),
        "std_reward": np.std(reward_history),
        "mean_length": np.mean(length_history)
    }

    if at_end:
        output_path = Path(eval_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    
    return result

@hydra.main(version_base=None, config_path='./config', config_name='dummy')
def run(cfg):
    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Atari Env
    atari_env = hydra.utils.instantiate(cfg.env)
    num_actions = atari_env.num_actions

    OmegaConf.update(
        cfg,
        'model.action_encoder.num_actions',
        int(num_actions),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        'agent.num_actions',
        int(num_actions),
        merge=False,
    )

    # W&B
    wandb_run = try_wandb_init(cfg)

    # World Model
    world_model = hydra.utils.instantiate(cfg.model)
    world_model.to(cfg.device)

    # Replay Writer
    replay_writer = hydra.utils.instantiate(cfg.replay)

    # Dataset
    dataset = hydra.utils.instantiate(cfg.dataset)

    # Imagination Env
    imagination_env = ImaginationEnv(
        world_model=world_model,
        dataset=dataset,
        **cfg.imagination
    )

    # Agent
    agent = hydra.utils.instantiate(cfg.agent)

    #########################
    ##      Training       ##
    #########################

    # Initial observation
    obs, _ = atari_env.reset(seed=cfg.seed)

    # World Model optimizer
    wm_optimizer = build_optimizer(
        world_model.parameters(),
        cfg.trainer.optimizer,
    )
    agent_optimizer = build_optimizer(
        agent.parameters(),
        cfg.agent_trainer.optimizer,
    )

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
        
        epoch_num = epoch_idx+1
        print("running epoch: ", epoch_num)

        # Collection
        if (epoch_num < cfg.collection_schedule.random_collection_epochs):
            is_random = True
        else:
            is_random = False

        if total_collected_interactions < cfg.collection_schedule.collection_limit:
            total_collected_interactions += collection_per_epoch  
            obs = collect_real_interactions(
                num_interactions=collection_per_epoch,
                is_random=is_random,
                obs=obs,
                agent=agent,
                world_model=world_model,
                env=atari_env,
                writer=replay_writer,
                device=cfg.device,
            )

            log_wandb(
                wandb_run,
                {'replay/collection_size': replay_writer.size},
                epoch_idx
            )

            print("collection complete. size: ", replay_writer.size)
        
        # Training World Model
        wm_train_epochs = None
        if cfg.wm_schedule.start_epoch <= (epoch_num) < cfg.wm_schedule.periodic_start_epoch:
            wm_train_epochs = cfg.wm_schedule.regular_epochs
        elif cfg.wm_schedule.periodic_start_epoch <= (epoch_num) <= cfg.wm_schedule.stop_epoch and (epoch_num) % cfg.wm_schedule.period == 0:
            wm_train_epochs = cfg.wm_schedule.periodic_epochs
        
        if wm_train_epochs is not None:
            for _ in range(wm_train_epochs):
                wm_losses = train_world_model(
                    world_model=world_model,
                    dataset=dataset,
                    loader_cfg=cfg.loader,
                    trainer_cfg=cfg.trainer,
                    loss_weights=cfg.loss_weights,
                    history_size=cfg.history_size,
                    optimizer=wm_optimizer,
                    device=cfg.device,
                )

            log_wandb(
                wandb_run,
                {
                    f"world_model/{key}": value
                    for key, value in wm_losses.items()
                },
                epoch_idx
            )
        
            print("world model training complete. losses:")
            print(wm_losses)

        # Checkpointing World Model
        if((epoch_num) % cfg.checkpointing.wm_per_epoch == 0):
            file_name = f"epoch_{epoch_num}.pt"
            wm_ckp_path = wm_ckp_dir / file_name
            torch.save(world_model.state_dict(), wm_ckp_path)
        
        # Training Agent
        if(epoch_num >= cfg.agent_trainer.agent_start_epoch):
            agent_losses = agent.learn(
                imagination_env,
                cfg.agent_trainer,
                agent_optimizer
            )

            log_wandb(
                wandb_run,
                {
                    f"agent/{key}": value
                    for key, value in agent_losses.items()
                },
                epoch_idx
            )

            print("agent training done. losses: ")
            print(agent_losses)

        # Checkpointing Agent
        if((epoch_num) % cfg.checkpointing.agent_per_epoch == 0):
            file_name = f"epoch_{epoch_num}.pt"
            agent_ckp_path = agent_ckp_dir / file_name
            torch.save(agent.state_dict(), agent_ckp_path)

        # Sanity Eval Checks
        if((epoch_idx + 1) % cfg.sanity_eval.every_x_epoch == 0):
            sanity_eval = eval_agent(
                episodes=cfg.sanity_eval.episodes,
                per_episode_limit=cfg.sanity_eval.per_episode_limit,
                agent=agent,
                world_model=world_model,
                env_cfg=cfg.env,
                device=cfg.device,
                at_end=False,
            )
            log_wandb(
                wandb_run,
                {
                    'sanity_eval/mean_rew': sanity_eval["mean_reward"],
                    'sanity_eval/mean_len': sanity_eval['mean_length'],
                },
                epoch_idx
            )
            print("sanity eval complete. results:")
            print(sanity_eval)

    # Checkpointing Final World Model
    file_name = f"final.pt"
    wm_ckp_path = wm_ckp_dir / file_name
    torch.save(world_model.state_dict(), wm_ckp_path)

    # Checkpointing Final Agent
    file_name = f"final.pt"
    agent_ckp_path = agent_ckp_dir / file_name
    torch.save(agent.state_dict(), agent_ckp_path)

    #########################
    ##     Evaluation      ##
    #########################

    eval_agent(
        episodes=cfg.eval.episodes,
        per_episode_limit=cfg.eval.per_episode_limit,
        agent=agent,
        world_model=world_model,
        env_cfg=cfg.env,
        device=cfg.device,
        at_end=True,
        eval_path=cfg.eval.output_path,
    )

    atari_env.close()
    replay_writer.close()
    dataset.close()
    if wandb_run is not None:
        wandb_run.finish()
    
if __name__ == "__main__":
    run()
