import json
from pathlib import Path
import hydra
from omegaconf import OmegaConf

import numpy as np
import torch

import stable_pretraining

@hydra.main(version_base=None, config_path='./config', config_name='config')
def run(cfg):
    # Env
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

    # Instantiate models
    world_model = hydra.utils.instantiate(cfg.model)
    agent = hydra.utils.instantiate(cfg.agent)
    
    # Prepare paths
    wm_ckp_dir = Path(cfg.checkpointing.wm_path)
    agent_ckp_dir = Path(cfg.checkpointing.agent_path)

    file_name = f"final.pt"
    wm_ckp_path = wm_ckp_dir / file_name
    agent_ckp_path = agent_ckp_dir / file_name

    # Load saved models
    wm_state_dict = torch.load(wm_ckp_path)
    agent_state_dict = torch.load(agent_ckp_path)
    world_model.load_state_dict(wm_state_dict)
    agent.load_state_dict(agent_state_dict)
    world_model.to(cfg.device)
    agent.to(cfg.device)

    world_model.eval()
    agent.eval()

    # Tracking
    reward_history = []
    length_history = []

    for ep_num in range(cfg.eval.episodes):
        obs, _ = atari_env.reset(seed=cfg.seed + ep_num)
        _ = agent.reset(1)

        done = False
        total_return = 0.0
        length = 0

        while not done:
            obs = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(cfg.device)

            with torch.no_grad():
                emb = world_model.encode(obs)
                emb = emb.squeeze(1)
                action, _ = agent.act(
                    emb,
                    deterministic=cfg.eval.deterministic,
                )
            
            obs, reward, terminated, truncated, _ = atari_env.step(int(action))
            
            done = terminated or truncated

            total_return += reward
            length += 1

            if length >= cfg.eval.per_episode_limit:
                done = True
        
        reward_history.append(total_return)
        length_history.append(length)

    atari_env.close()
    result = {
        "episodes": cfg.eval.episodes,
        "ep_rewards": reward_history,
        "ep_lengths": length_history,
        "mean_reward": np.mean(reward_history),
        "std_reward": np.std(reward_history),
        "mean_length": np.mean(length_history)
    }
    
    output_path = Path(cfg.eval.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')

if __name__ == "__main__":
    run()
