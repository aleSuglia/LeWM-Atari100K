from pathlib import Path
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torch.distributions import Categorical

def layer_init(layer, std=np.sqrt(2), bias=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, embed_dim, num_actions, hidden_dim):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(embed_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, num_actions), 0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(embed_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), 1.0),
        )

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)

class PPOAgent(nn.Module):
    def __init__(
            self, 
            embed_dim,
            hidden_dim,
            num_envs,
            rollout_steps,
            num_actions, 
            device
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps

        self.device = torch.device(device)
        self.net = ActorCritic(self.embed_dim, num_actions, self.hidden_dim).to(self.device)

    def step(self, obs, deterministic=False):
        logits, value = self.net(obs)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        logprob = dist.log_prob(action)

        return action, logprob, value

    @torch.no_grad()
    def predict(self, obs, deterministic=False):
        # always recieves single obs
        action, _, _ = self.step(obs, deterministic)
        return int(action.item())

    @torch.no_grad()
    def rollout(self, imagination_env, trainer_cfg):
        obs = imagination_env.reset(self.num_envs)
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        
        for _ in range(self.rollout_steps):
            action, logprob, value = self.step(obs)
            next_obs, reward, done, _ = imagination_env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(logprob)
            rew_buf.append(reward)
            done_buf.append(done.float())
            val_buf.append(value)
            obs = next_obs
        
        next_value = self.net(obs)[1]
        rewards = torch.stack(rew_buf)
        dones = torch.stack(done_buf)
        values = torch.stack(val_buf)
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(self.num_envs, device=self.device)
        
        # GAE Computation
        for t in reversed(range(self.rollout_steps)):
            nextnonterminal = 1.0 - dones[t]
            nextvalues = next_value if t == self.rollout_steps - 1 else values[t + 1]
            delta = rewards[t] + trainer_cfg.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + trainer_cfg.gamma * trainer_cfg.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
        
        return {
            'obs': torch.stack(obs_buf).reshape(-1, obs.shape[-1]),
            'actions': torch.stack(act_buf).reshape(-1),
            'logprobs': torch.stack(logp_buf).reshape(-1),
            'advantages': advantages.reshape(-1),
            'returns': returns.reshape(-1),
        }

    def evaluate_actions(self, obs, actions):
        logits, value = self.net(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value

    def loss(self, batch, trainer_cfg):
        newlogprob, entropy, newvalue = self.evaluate_actions(batch['obs'], batch['actions'])
        
        logratio = newlogprob - batch['logprobs']
        ratio = logratio.exp()
        
        adv = batch['advantages']
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        pg_loss = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1 - trainer_cfg.clip_coef, 1 + trainer_cfg.clip_coef)).mean()
        v_loss = F.mse_loss(newvalue, batch['returns'])
        entropy_loss = entropy.mean()

        loss = pg_loss + trainer_cfg.vf_coef * v_loss - trainer_cfg.ent_coef * entropy_loss
        return loss, {'policy_loss': pg_loss, 'value_loss': v_loss, 'entropy': entropy_loss}

    def update(self, rollout, trainer_cfg, optimizer):
        batch_size = rollout['obs'].size(0)
        metrics = {}
        
        for _ in range(trainer_cfg.update_epochs):
            idx = torch.randperm(batch_size, device=self.device)

            batch = {k: v[idx] for k, v in rollout.items()}
            loss, metrics = self.loss(batch, trainer_cfg)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), trainer_cfg.max_grad_norm)
            optimizer.step()
        
        return {k: float(v.detach().cpu()) for k, v in metrics.items()}

    def learn(self, imagination_env, trainer_cfg, optimizer):
        self.train()
        imagination_env.world_model.eval()
        
        metrics = {}
        steps_done = 0
        
        while steps_done < trainer_cfg.total_steps:
            rollout = self.rollout(imagination_env, trainer_cfg)
            steps_done += self.num_envs * self.rollout_steps
            metrics = self.update(rollout, trainer_cfg, optimizer)
        
        return metrics
