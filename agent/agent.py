from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

@dataclass
class ActorCriticOutput:
    logits_actions: torch.FloatTensor
    logits_values: torch.FloatTensor

class ActorCritic(nn.Module):
    def __init__(self, embed_dim, num_actions, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hx, self.cx = None, None
        self.lstm = nn.LSTMCell(embed_dim, hidden_dim)
        self.actor_linear = nn.Linear(hidden_dim, num_actions)
        self.critic_linear = nn.Linear(hidden_dim, 1)
        self.critic_linear.weight.data.zero_()
        self.critic_linear.bias.data.zero_()

    @property
    def device(self):
        return self.lstm.weight_hh.device

    def reset(self, n):
        self.hx = torch.zeros(n, self.hidden_dim, device=self.device)
        self.cx = torch.zeros(n, self.hidden_dim, device=self.device)

    def clear(self):
        self.hx, self.cx = None, None

    @torch.no_grad()
    def burn_in(self, obs):
        if obs.size(1) > 0:
            _ = self(obs)

    def forward(self, obs, **kwargs):
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)

        assert obs.ndim == 3

        if self.hx is None or self.hx.size(0) != obs.size(0):
            self.reset(obs.size(0))

        all_logits_actions = []
        all_logits_values = []

        for i in range(obs.size(1)):
            self.hx, self.cx = self.lstm(obs[:, i], (self.hx, self.cx))
            all_logits_actions.append(self.actor_linear(self.hx).unsqueeze(1))
            all_logits_values.append(self.critic_linear(self.hx).unsqueeze(1))

        return ActorCriticOutput(
            torch.cat(all_logits_actions, dim=1),
            torch.cat(all_logits_values, dim=1),
        )

def compute_lambda_returns(rewards, values, ends, value_bootstrap, gamma, lambda_):
    lambda_returns = rewards + ends.logical_not() * gamma * (1 - lambda_) * torch.cat(
        (values[:, 1:], value_bootstrap.unsqueeze(1)), dim=1
    )
    last = value_bootstrap

    for t in list(range(rewards.size(1)))[::-1]:
        lambda_returns[:, t] += ends[:, t].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, t]

    return lambda_returns

def compute_mask_after_first_done(ends):
    first_one_index = torch.argmax(ends.long(), dim=1)
    mask = torch.arange(ends.size(1), device=ends.device).unsqueeze(0) <= first_one_index.unsqueeze(1)
    mask = torch.logical_or(mask, ends.sum(dim=1, keepdim=True) == 0)
    return mask

class Agent(nn.Module):
    def __init__(
            self,
            embed_dim,
            hidden_dim,
            num_envs,
            num_actions,
            device,
            rollout_steps=None,
            history_size=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.history_size = history_size

        self.device = torch.device(device)
        self.model = ActorCritic(self.embed_dim, num_actions, self.hidden_dim).to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)
        self.target_model.requires_grad_(False)

    @property
    def net(self):
        return self.model

    def reset(self, n=1):
        self.model.reset(n)
        self.target_model.reset(n)

    def clear(self):
        self.model.clear()
        self.target_model.clear()

    def update_target(self):
        source_state_dict = self.model.state_dict()
        target_state_dict = self.target_model.state_dict()
        TAU = 0.995
        for key in source_state_dict:
            target_state_dict[key] = source_state_dict[key] * (1 - TAU) + target_state_dict[key] * TAU
        self.target_model.load_state_dict(target_state_dict)

    def step(self, obs, deterministic=False):
        obs = obs.to(self.device).float()
        outputs = self.model(obs)
        logits = outputs.logits_actions[:, -1]
        value = outputs.logits_values[:, -1, 0]
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        logprob = dist.log_prob(action)

        return action, logprob, value

    @torch.no_grad()
    def predict(self, obs, deterministic=False):
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        action, _, _ = self.step(obs, deterministic)
        return int(action.item())

    def imagine(self, imagination_env):
        context = imagination_env.reset(self.num_envs).to(self.device).float()
        obs = context[:, -1]

        self.reset(n=self.num_envs)
        self.model.burn_in(context[:, :-1])
        self.target_model.burn_in(context[:, :-1])

        all_actions = []
        all_logits_actions = []
        all_logits_values = []
        all_target_values = []
        all_rewards = []
        all_ends = []

        for _ in range(self.rollout_steps):
            outputs = self.model(obs.unsqueeze(1))
            logits_actions = outputs.logits_actions[:, -1]
            action = Categorical(logits=logits_actions).sample()

            with torch.no_grad():
                target_values = self.target_model(obs.unsqueeze(1)).logits_values[:, -1, 0]

            obs, reward, done, _ = imagination_env.step(action)

            all_actions.append(action.unsqueeze(1))
            all_logits_actions.append(outputs.logits_actions)
            all_logits_values.append(outputs.logits_values[:, :, 0])
            all_target_values.append(target_values.unsqueeze(1))
            all_rewards.append(reward.reshape(-1, 1))
            all_ends.append(done.reshape(-1, 1))

        with torch.no_grad():
            value_bootstrap = self.target_model(obs.unsqueeze(1)).logits_values[:, -1, 0]

        return {
            'actions': torch.cat(all_actions, dim=1),
            'logits_actions': torch.cat(all_logits_actions, dim=1),
            'logits_values': torch.cat(all_logits_values, dim=1),
            'rewards': torch.cat(all_rewards, dim=1).to(self.device),
            'ends': torch.cat(all_ends, dim=1).bool().to(self.device),
            'target_values': torch.cat(all_target_values, dim=1),
            'value_bootstrap': value_bootstrap,
        }

    def loss(self, rollout, trainer_cfg):
        lambda_ = trainer_cfg.lambda_ if hasattr(trainer_cfg, 'lambda_') else trainer_cfg.gae_lambda
        entropy_weight = trainer_cfg.entropy_weight if hasattr(trainer_cfg, 'entropy_weight') else trainer_cfg.ent_coef

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=rollout['rewards'],
                values=rollout['target_values'],
                ends=rollout['ends'],
                value_bootstrap=rollout['value_bootstrap'],
                gamma=trainer_cfg.gamma,
                lambda_=lambda_,
            )

        dist = Categorical(logits=rollout['logits_actions'])
        log_probs = dist.log_prob(rollout['actions'])
        mask = compute_mask_after_first_done(rollout['ends'])

        loss_actions = torch.mean(-log_probs[mask] * (lambda_returns[mask] - rollout['target_values'][mask]))
        loss_values = F.mse_loss(rollout['logits_values'][mask], lambda_returns[mask])
        entropy = torch.mean(dist.entropy()[mask])
        loss_entropy = -entropy_weight * entropy
        loss = loss_actions + loss_values + loss_entropy

        self.update_target()

        metrics = {
            'loss': loss,
            'policy_loss': loss_actions,
            'value_loss': loss_values,
            'entropy': entropy,
        }
        return loss, metrics

    def learn(self, imagination_env, trainer_cfg, optimizer):
        self.train()
        imagination_env.world_model.eval()

        metrics = {}
        steps_done = 0

        while steps_done < trainer_cfg.total_steps:
            rollout = self.imagine(imagination_env)
            loss, metrics = self.loss(rollout, trainer_cfg)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), trainer_cfg.max_grad_norm)
            optimizer.step()

            self.clear()
            steps_done += self.num_envs * self.rollout_steps

        return {k: float(v.detach().cpu()) for k, v in metrics.items()}
