import json
from pathlib import Path

import random
import numpy as np
import torch

import hydra
from omegaconf import OmegaConf

from lewm.imagination import ImaginationEnv
from utils import get_agent, _build_optimizer


def collect_real_interactions(
        num_interactions,
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
    world_model.eval()
    
    # Handles env.reset() externally as its necessary for data collection
    needs_reset = False

    for _ in range(num_interactions):
        if needs_reset:
            obs, _ = env.reset()
        
        obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = world_model.encode(obs).squeeze(0).squeeze(0)
            action, _ = agent.predict(
                emb.cpu().numpy(),
                deterministic=False,    # Exploration
            )

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
    print("how much data is in the file: ", writer.size)

    return obs

def train_world_model(
        num_epochs,
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

    for epoch_idx in range(num_epochs):
        loss_tracker = None
        num_batches = 0

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

            ##### Tracking #####
            detached_losses = {
                name: float(value.detach().clone().cpu())
                for name, value in losses.items()
            }

            if loss_tracker is None:
                loss_tracker = {name: 0.0 for name in detached_losses}
            
            for name, value in detached_losses.items():
                loss_tracker[name] += value
            
            num_batches += 1

            history = {
                'epoch': epoch_idx,
                **{name: value / num_batches for name, value in loss_tracker.items()}
            }
            print("training loss", history)

def eval_agent(
        episodes,
        per_episode_limit,
        agent,
        real_env,
        world_model, 
        device,
        at_end=False,
        seed=None,
        eval_path=None,
):
    """
    """
    world_model.eval()

    reward_history = []
    length_history = []

    for ep_idx in range(episodes):
        if not at_end:
            obs, _ = real_env.reset()
        else:
            obs, _ = real_env.reset(seed + ep_idx)

        done = False
        total_return  = 0.0
        length = 0

        while not done:
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = world_model.encode(obs).squeeze(0).squeeze(0)
                action, _ = agent.predict(
                    emb.cpu().numpy(),
                    deterministic=True,    # Exploitation
                )
            
            obs, reward, terminated, truncated, info = real_env.step(int(action))
            
            done = terminated or truncated

            total_return += reward
            length += 1

            if length >= per_episode_limit:
                done = True
        
        reward_history.append(total_return)
        length_history.append(length)
    
    print("eval mean return is ", np.mean(reward_history))
    print("eval std is ", np.std(reward_history))
    print("eval mean length is ", np.mean(length_history))

    if at_end:
        result = {
            "episodes": episodes,
            "ep_rewards": reward_history,
            "ep_lengths": length_history,
            "mean_reward": np.mean(reward_history),
            "std_reward": np.std(reward_history),
            "mean_length": np.mean(length_history)
        }
        output_path = Path(eval_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')


@hydra.main(version_base=None, config_path='./config', config_name='dummy')
def run(cfg):
    #########################
    ##         Seed        ##
    #########################
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #########################
    ##      Atari Env      ##
    #########################
    atari_env = hydra.utils.instantiate(cfg.env)
    num_actions = atari_env.num_actions

    OmegaConf.update(
        cfg,
        'model.action_encoder.num_actions',
        int(num_actions),
        merge=False,
    )

    #########################
    ##     World Model     ##
    #########################
    world_model = hydra.utils.instantiate(cfg.model)
    world_model.to(cfg.device)

    #########################
    ##        Replay       ##
    #########################
    replay_writer = hydra.utils.instantiate(cfg.replay)

    #########################
    ##       Dataset       ##
    #########################
    dataset = hydra.utils.instantiate(cfg.dataset)

    #########################
    ##     Imagination     ##
    #########################
    imagination_env = ImaginationEnv(
        num_actions=num_actions,
        world_model=world_model,
        dataset=dataset,
        **cfg.imagination
    )

    #########################
    ##        Agent        ##
    #########################
    agent = get_agent(
        cfg.agent,
        env=imagination_env,
        device=cfg.device
    )

    #########################
    ##      Training       ##
    #########################

    obs, _ = atari_env.reset(seed=cfg.seed)

    wm_optimizer = _build_optimizer(
        world_model.parameters(),
        cfg.trainer.optimizer,
    )

    total_collected_interactions = 0
    collection_size = cfg.collection_trainer.collection_per_epoch
    num_imagine_interactions = int(
        cfg.agent_trainer.total_steps / cfg.agent_trainer.per_rollout_steps
    )

    for epoch_idx in range(cfg.trainer.total_epochs):
        
        # Collection
        if total_collected_interactions < cfg.collection_trainer.collection_limit:
            total_collected_interactions += collection_size  
            obs = collect_real_interactions(
                num_interactions=collection_size,
                obs=obs,
                agent=agent,
                world_model=world_model,
                env=atari_env,
                writer=replay_writer,
                device=cfg.device,
            )

        # Training World Model
        if(
            (epoch_idx+1 >= cfg.world_model_trainer.world_model_start_epoch) and (epoch_idx+1 <= cfg.world_model_trainer.world_model_stop_epoch)
        ):
            train_world_model(
                num_epochs=cfg.world_model_trainer.world_model_epochs,
                world_model=world_model,
                dataset=dataset,
                loader_cfg=cfg.loader,
                trainer_cfg=cfg.trainer,
                loss_weights=cfg.loss_weights,
                history_size=cfg.history_size,
                optimizer=wm_optimizer,
                device=cfg.device,
            )

        # Training Agent
        print("training agent going awol")
        if(epoch_idx+1 >= cfg.agent_trainer.agent_start_epoch):
            for _ in range(num_imagine_interactions):
                agent.learn(cfg.agent_trainer.per_rollout_steps)
                print("agent has finished learning for certain steps")
        
        # Sanity Eval Checks
        if((epoch_idx + 1) % cfg.trainer.sanity_eval.every_x_epoch == 0):
            eval_agent(
                episodes=cfg.trainer.sanity_eval.episodes,
                per_episode_limit=cfg.trainer.sanity_eval.per_episode_limit,
                agent=agent,
                real_env=atari_env,
                world_model=world_model,
                device=cfg.device,
                at_end=False,
            )

    #########################
    ##     Evaluation      ##
    #########################

    eval_agent(
        episodes=cfg.eval.episodes,
        per_episode_limit=cfg.eval.per_episode_limit,
        agent=agent,
        real_env=atari_env,
        world_model=world_model,
        device=cfg.device,
        at_end=True,
        seed=cfg.seed,
        eval_path=cfg.eval.output_path,
    )

    atari_env.close()
    replay_writer.close()
    dataset.close()
    
if __name__ == "__main__":
    run()
