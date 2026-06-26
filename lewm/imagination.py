import torch

class ImaginationEnv:
    """
    Batched Env
    """
    def __init__(
        self,
        max_horizon,
        world_model,
        history_size,
        dataset,
        done_threshold=0.5,
    ):
        self.world_model = world_model
        self.dataset = dataset
        self.history_size = history_size

        self.max_horizon = max_horizon
        self.done_threshold = done_threshold

        self.device = next(world_model.parameters()).device

        self.elapsed = None
        self.emb_history = None
        self.emb_act_history = None

    @torch.no_grad()
    def _sample_context(self, batch_size):
        obs, actions = [], []
        
        for idx in torch.randint(len(self.dataset), (batch_size,)):
            sample = self.dataset[int(idx)]
            T = sample['obs'].shape[0]
            start = int(torch.randint(T - self.history_size + 1, (1,)))
            
            obs.append(sample['obs'][start:start + self.history_size])
            actions.append(sample['action'][start:start + self.history_size])
        
        obs = torch.stack(obs).to(self.device).float()
        actions = torch.stack(actions).to(self.device).long()
        
        emb = self.world_model.encode(obs)
        act_emb = self.world_model.encode_action(actions)
        return emb, act_emb[:, :-1]

    @torch.no_grad()
    def reset(self, batch_size=None, mask=None):
        if mask is None:
            self.elapsed = torch.zeros(batch_size, device=self.device)
            self.emb_history, self.emb_act_history = self._sample_context(batch_size)
            return self.emb_history[:, -1]
        
        emb, act_emb = self._sample_context(int(mask.sum()))
        
        self.elapsed[mask] = 0
        self.emb_history[mask] = emb
        self.emb_act_history[mask] = act_emb
        
        return self.emb_history[:, -1]

    @torch.no_grad()
    def step(self, action):
        act_emb = self.world_model.encode_action(action).unsqueeze(1)
        self.emb_act_history = torch.cat([self.emb_act_history, act_emb], dim=1)

        next_emb, reward, done_logit = self.world_model.transition(self.emb_history, self.emb_act_history)
        next_emb = next_emb.squeeze(1)
        reward = reward.squeeze(-1)

        terminated = torch.sigmoid(done_logit.squeeze(-1)) >= self.done_threshold
        
        self.elapsed += 1
        truncated = self.elapsed >= self.max_horizon
        
        done = terminated | truncated
        
        self.emb_history = torch.cat([self.emb_history, next_emb.unsqueeze(1)], dim=1)[:, -self.history_size:]
        self.emb_act_history = self.emb_act_history[:, -self.history_size + 1:]
        
        if done.any():
            self.reset(mask=done)
            next_emb = torch.where(done[:, None], self.emb_history[:, -1], next_emb)
        
        return next_emb, reward, done, {'terminated': terminated, 'truncated': truncated}
