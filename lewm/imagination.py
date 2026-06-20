import torch
import numpy as np

import gymnasium as gym
from gymnasium import spaces

class ImaginationEnv(gym.Env):
    """
    """
    def __init__(
        self,
        max_horizon,
        env_id,
        num_actions,
        world_model,
        history_size,
        embed_dim,
        dataset,                # Sequence Dataset
        done_threshold=0.5,
    ):
        super().__init__()

        self.world_model = world_model
        self.history_size = history_size
        self.dataset = dataset

        self.env_id = env_id
        self.num_actions = num_actions

        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(embed_dim,),
            dtype=np.float32,
        )

        self.device = next(self.world_model.parameters()).device

        self.emb_history = None
        self.emb_act_history = None
        
        self.elapsed = 0
        self.max_horizon = max_horizon
        self.done_threshold = done_threshold
    
    def _init_sample(self):
        """
        make sure to set drop_last = true in dataloader
        have T > history_size (this is true generally for the project too)
        """
        # Fetch a random datapoint of size (T, C, H, W)
        idx = np.random.randint(len(self.dataset))
        obs = self.dataset[idx]['obs']                                          # (T, C, H, W)
        actions = self.dataset[idx]['action']

        T = obs.shape[0]

        if not (T >= self.history_size):
            raise RuntimeError("T < history_size")
        
        # Pick a random starting point
        start = np.random.randint(0, T - self.history_size + 1)

        # Cut based on start and history size
        obs = obs[start : start + self.history_size]                                # (H, C, H_img, W_img)
        actions = actions[start : start + self.history_size]
        
        obs = obs.to(self.device).unsqueeze(0)                                      # (1, H, C, H_img, W_img)
        actions = actions.to(self.device).unsqueeze(0)

        with torch.no_grad():
            emb = self.world_model.encode(obs)                                      # (1, H, Z)
            emb_act = self.world_model.encode_action(actions)

        # Leave last action as that will be passed by step() function
        self.emb_history = emb                                                      # (1, H, Z)
        self.emb_act_history = emb_act[:, :-1]                                      # (1, H-1, Z)

        return emb[:, -1].squeeze(0).detach().cpu().numpy().astype(np.float32)      # (Z)


    def reset(self, seed = None):
        super().reset(seed=seed)        
        self.elapsed = 0

        return self._init_sample(), {}
    
    def step(self, action):
        action_tensor = torch.tensor([int(action)], device = self.device).unsqueeze(0)
        
        # Encoding the action and then predicting next values
        with torch.no_grad():
            emb_act = self.world_model.encode_action(action_tensor)
            self.emb_act_history = torch.cat([self.emb_act_history, emb_act], dim = 1)

            nxt_emb, nxt_rew, nxt_don = self.world_model.transition(
                self.emb_history, self.emb_act_history
            )
        
        # Handling the recieved rewards
        reward = float(nxt_rew.squeeze().detach().cpu()) if nxt_rew is not None else 0.0

        # Computing terminated flag
        done_probability = 0.0
        terminated = False
        if nxt_don is not None:
            done_probability = torch.sigmoid(nxt_don.squeeze()).item()
            terminated = (done_probability >= self.done_threshold)
        
        # Computing truncated flag
        self.elapsed += 1
        truncated = (self.elapsed >= self.max_horizon)

        info = {
            "done_probability": done_probability,
            "elapsed_steps": self.elapsed,
        }

        # Resetting the env if the previous rollout is done
        if terminated or truncated:
            nxt_emb, _ = self.reset()
            nxt_emb = torch.from_numpy(nxt_emb).to(self.device)
            nxt_emb = nxt_emb.unsqueeze(0).unsqueeze(0)

        # Adding the recieved embedding to history
        self.emb_history = torch.cat([self.emb_history, nxt_emb], dim=1)

        # Updating history for efficiency
        self.emb_history = self.emb_history[:, -self.history_size:]
            # Take only last two actions, the other will be added at step()
        self.emb_act_history = self.emb_act_history[:, -self.history_size + 1:]

        # Preparing the agent observation
        agent_obs = nxt_emb.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

        return agent_obs, reward, terminated, truncated, info
