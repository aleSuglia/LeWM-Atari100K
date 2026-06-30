from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from torch.distributions import Categorical

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

    def forward(self, obs):
        if obs.ndim == 2:
            obs = obs.unsqueeze(1)

        assert obs.ndim == 3

        if self.hx is None or self.hx.size(0) != obs.size(0):
            self.reset(obs.size(0))

        logits_actions = []
        logits_values = []

        for i in range(obs.size(1)):
            self.hx, self.cx = self.lstm(obs[:, i], (self.hx, self.cx))
            logits_actions.append(self.actor_linear(self.hx))
            logits_values.append(self.critic_linear(self.hx))

        return torch.stack(logits_actions, dim=1), torch.stack(logits_values, dim=1)

    @torch.no_grad()
    def burn_in(self, obs):
        _ = self(obs)

def compute_mask_after_first_done(dones):
    assert dones.ndim == 2
    first_done_idx = torch.argmax(dones.long(), dim=1)
    mask = torch.arange(dones.size(1), device=dones.device).unsqueeze(0) <= first_done_idx.unsqueeze(1)
    return torch.logical_or(mask, dones.sum(dim=1, keepdim=True) == 0)

@torch.no_grad()
def compute_lambda_returns(rewards, values, dones, value_bootstrap, gamma, lambda_):
    assert rewards.ndim == 2
    assert rewards.size() == values.size() == dones.size()
    assert value_bootstrap.ndim == 1 and value_bootstrap.size(0) == rewards.size(0)

    lambda_returns = rewards + dones.logical_not() * gamma * (1 - lambda_) * torch.cat(
        (values[:, 1:], value_bootstrap.unsqueeze(1)),
        dim=1,
    )
    last = value_bootstrap

    for t in list(range(rewards.size(1)))[::-1]:
        lambda_returns[:, t] += dones[:, t].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, t]

    return lambda_returns

class Agent(nn.Module):
    def __init__(
            self,
            embed_dim,
            hidden_dim,
            num_envs,
            burn_in_length,
            imagination_horizon,
            num_actions,
            device,
            target_tau=0.995,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_envs = num_envs
        self.burn_in_length = burn_in_length
        self.imagination_horizon = imagination_horizon
        self.num_actions = num_actions
        self.target_tau = target_tau

        self.device = torch.device(device)
        self.net = ActorCritic(self.embed_dim, self.num_actions, self.hidden_dim).to(self.device)
        self.target_net = deepcopy(self.net).to(self.device)
        self.target_net.requires_grad_(False)

    def reset(self, n):
        self.net.reset(n)
        self.target_net.reset(n)

    def clear(self):
        self.net.clear()
        self.target_net.clear()

    def update_target(self):
        source_state_dict = self.net.state_dict()
        target_state_dict = self.target_net.state_dict()

        for key in source_state_dict:
            target_state_dict[key] = source_state_dict[key] * (1 - self.target_tau) + target_state_dict[key] * self.target_tau

        self.target_net.load_state_dict(target_state_dict)

    def _to_device(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs)
        return obs.to(self.device).float()

    def _prepare_reset_obs(self, obs):
        obs = self._to_device(obs)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        self.reset(obs.size(0))

        if obs.ndim == 3 and obs.size(1) > 1:
            burn_in = obs[:, :-1]
            self.net.burn_in(burn_in)
            self.target_net.burn_in(burn_in)
            obs = obs[:, -1]

        return obs

    def step(self, obs, deterministic=False):
        obs = self._to_device(obs)
        logits_actions, logits_values = self.net(obs)
        logits_actions = logits_actions[:, -1]
        values = logits_values[:, -1, 0]

        dist = Categorical(logits=logits_actions)
        action = logits_actions.argmax(dim=-1) if deterministic else dist.sample()
        logprob = dist.log_prob(action)

        return action, logprob, values

    @torch.no_grad()
    def predict(self, obs, deterministic=False):
        # always recieves single obs
        action, _, _ = self.step(obs, deterministic)
        return int(action.reshape(-1)[0].item())

    def imagine(self, imagination_env):
        obs = self._prepare_reset_obs(imagination_env.reset(self.num_envs))

        all_actions = []
        all_logits_actions = []
        all_logits_values = []
        all_target_values = []
        all_rewards = []
        all_dones = []

        for _ in range(self.imagination_horizon):
            logits_actions, logits_values = self.net(obs)
            logits_actions = logits_actions[:, -1]
            logits_values = logits_values[:, -1, 0]

            dist = Categorical(logits=logits_actions)
            action = dist.sample()

            with torch.no_grad():
                target_logits_values = self.target_net(obs)[1][:, -1, 0]
                next_obs, reward, done, _ = imagination_env.step(action)

            all_actions.append(action)
            all_logits_actions.append(logits_actions)
            all_logits_values.append(logits_values)
            all_target_values.append(target_logits_values)
            all_rewards.append(self._to_device(reward).reshape(-1))
            all_dones.append(self._to_device(done).reshape(-1).bool())

            obs = self._to_device(next_obs)

        with torch.no_grad():
            value_bootstrap = self.target_net(obs)[1][:, -1, 0]

        self.clear()

        return {
            'actions': torch.stack(all_actions, dim=1),
            'logits_actions': torch.stack(all_logits_actions, dim=1),
            'logits_values': torch.stack(all_logits_values, dim=1),
            'rewards': torch.stack(all_rewards, dim=1),
            'dones': torch.stack(all_dones, dim=1),
            'target_values': torch.stack(all_target_values, dim=1),
            'value_bootstrap': value_bootstrap,
        }

    def loss(self, imagination_env, trainer_cfg):
        outputs = self.imagine(imagination_env)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs['rewards'],
                values=outputs['target_values'],
                dones=outputs['dones'],
                value_bootstrap=outputs['value_bootstrap'],
                gamma=trainer_cfg.gamma,
                lambda_=trainer_cfg.lambda_,
            )

        dist = Categorical(logits=outputs['logits_actions'])
        logprobs = dist.log_prob(outputs['actions'])
        entropy = dist.entropy()
        mask = compute_mask_after_first_done(outputs['dones'])

        advantages = lambda_returns[mask] - outputs['target_values'][mask]
        loss_actions = torch.mean(-logprobs[mask] * advantages)
        loss_values = F.mse_loss(outputs['logits_values'][mask], lambda_returns[mask])
        policy_entropy = torch.mean(entropy[mask])
        loss_entropy = -trainer_cfg.entropy_weight * policy_entropy
        loss = loss_actions + loss_values + loss_entropy

        self.update_target()

        return loss, {
            'loss_actions': loss_actions,
            'loss_values': loss_values,
            'loss_entropy': loss_entropy,
            'policy_entropy': policy_entropy,
        }

    def learn(self, imagination_env, trainer_cfg, optimizer):
        self.train()
        imagination_env.world_model.eval()

        metrics = {}
        steps_done = 0

        while steps_done < trainer_cfg.total_steps:
            loss, metrics = self.loss(imagination_env, trainer_cfg)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), trainer_cfg.max_grad_norm)
            optimizer.step()
            steps_done += self.num_envs * self.imagination_horizon

        return {key: float(value.detach().cpu()) for key, value in metrics.items()}
