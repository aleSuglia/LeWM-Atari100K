import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


def init_lstm(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(p.data)
        elif "bias_ih" in name:
            p.data.fill_(0)
            # Set forget-gate bias to 1
            n = p.size(0)
            p.data[(n // 4) : (n // 2)].fill_(1)
        elif "bias_hh" in name:
            p.data.fill_(0)


@torch.no_grad()
def compute_lambda_returns(
    rew,
    end,
    trunc,
    val_bootstrap,
    gamma,
    lambda_,
):
    assert rew.ndim == 2 and rew.size() == end.size() == trunc.size() == val_bootstrap.size()

    rew = rew.sign()  # clip reward

    end = end.float()
    trunc = trunc.float()

    end_or_trunc = (end + trunc).clip(max=1)
    not_end = 1 - end
    not_trunc = 1 - trunc

    lambda_returns = rew + not_end * gamma * (not_trunc * (1 - lambda_) + trunc) * val_bootstrap

    if lambda_ == 0:
        return lambda_returns

    last = val_bootstrap[:, -1]
    for t in reversed(range(rew.size(1))):
        lambda_returns[:, t] += end_or_trunc[:, t].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, t]

    return lambda_returns


class ActorCritic(nn.Module):
    """
    Recurrent actor-critic that receives latent states directly.

    Input:
        obs: (b, z)

    Output:
        act_logit: (b, num_actions)
        val:       (b,)
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_actions,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.critic_linear = nn.Linear(hidden_dim, 1)
        self.actor_linear = nn.Linear(hidden_dim, num_actions)

        self.actor_linear.weight.data.fill_(0)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)
        init_lstm(self.lstm)

        self.hx = None
        self.cx = None

    @property
    def device(self) -> torch.device:
        return self.lstm.weight_hh.device

    @torch.no_grad()
    def burn_in(self, context):                 # (b, t, z)
        assert self.hx is not None
        assert self.cx is not None
        assert context.ndim == 3

        context = context.to(self.device)
        for t in range(context.size(1)):
            self.hx, self.cx = self.lstm(context[:, t], (self.hx, self.cx))

    def predict_act_value(self, obs):            # (b, z)
        assert self.hx is not None
        assert self.cx is not None
        assert obs.ndim == 1 or obs.ndim == 2

        obs = obs.to(self.device)

        self.hx, self.cx = self.lstm(obs, (self.hx, self.cx))
        act_logit = self.actor_linear(self.hx)
        val = self.critic_linear(self.hx).squeeze(dim=1)

        return act_logit, val, (self.hx, self.cx)

class Agent(nn.Module):
    """
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_envs,
        num_actions,
        rollout_steps,
        burn_in_len,
        device,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.burn_in_len = burn_in_len

        self.device = torch.device(device)
        self.model = ActorCritic(
            self.input_dim,
            self.hidden_dim,
            num_actions,
        ).to(self.device)

    def set_memory(self, hx_cx):
        """
        """
        self.model.hx = hx_cx[0]
        self.model.cx = hx_cx[1]

    def reset(self, n):
        """
        """
        self.model.hx = torch.zeros(n, self.hidden_dim, device=self.device)
        self.model.cx = torch.zeros(n, self.hidden_dim, device=self.device)
        return (self.model.hx, self.model.cx)

    def clear(self):
        self.model.hx = None
        self.model.cx = None

    @torch.no_grad()
    def burn_in(self, context):
        """
        """
        self.model.burn_in(context)
        return (self.model.hx, self.model.cx)
    
    @torch.no_grad()
    def act(self, obs, deterministic):
        """
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        
        act_logit, _, hx_cx = self.model.predict_act_value(obs)
        if deterministic:
            action = torch.argmax(act_logit, dim=-1)
        else:
            action = Categorical(logits=act_logit).sample()
        
        return int(action.item()), hx_cx

    def imagine(self, imagination_env):
        """
        Collects one imagined rollout.

        The imagination environment does not reset internally after
        termination/truncation. Therefore, a mask is created so that
        the loss ignores all steps after the first dead step.
        """
        with torch.no_grad():
            context = imagination_env.reset(self.num_envs)              # (b, t, z)

        context = context.to(self.device)
        obs = context[:, -1]                                            # (b, z)

        self.reset(self.num_envs)

        burn_in_context = context[:, -self.burn_in_len - 1 : -1]            # (b, t, z)
        _ = self.burn_in(burn_in_context)

        all_actions = []
        all_rewards = []
        all_truncated = []
        all_terminated = []
        all_act_logits = []
        all_vals = []
        all_val_bootstraps = []
        all_masks = []

        alive = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        for step in range(self.rollout_steps):
            act_logit, val, _ = self.model.predict_act_value(obs)

            if step > 0:
                all_val_bootstraps[-1] = val.detach().clone()

            action = Categorical(logits=act_logit).sample()

            with torch.no_grad():
                obs, rew, terminated, truncated, info = imagination_env.step(action)

            obs = obs.to(self.device)
            rew = rew.to(self.device)
            terminated = terminated.to(self.device).bool()
            truncated = truncated.to(self.device).bool()

            mask = alive.float()
            dead = torch.logical_or(terminated, truncated)
            alive = torch.logical_and(alive, torch.logical_not(dead))

            all_actions.append(action)
            all_rewards.append(rew)
            all_truncated.append(truncated)
            all_terminated.append(terminated)
            all_act_logits.append(act_logit)
            all_vals.append(val)
            all_val_bootstraps.append(None)
            all_masks.append(mask)

        with torch.no_grad():
            hx, cx = self.model.hx, self.model.cx
            _, val_bootstrap, _ = self.model.predict_act_value(obs)
            self.model.hx, self.model.cx = hx, cx

        all_val_bootstraps[-1] = val_bootstrap.detach().clone()

        return {
            "actions": torch.stack(all_actions, dim=1),
            "rewards": torch.stack(all_rewards, dim=1),
            "truncated": torch.stack(all_truncated, dim=1),
            "terminated": torch.stack(all_terminated, dim=1),
            "act_logits": torch.stack(all_act_logits, dim=1),
            "vals": torch.stack(all_vals, dim=1),
            "val_bootstraps": torch.stack(all_val_bootstraps, dim=1),
            "mask": torch.stack(all_masks, dim=1),
        }

    def loss(
        self,
        rollout,
        cfg,
    ):
        gamma = cfg.gamma
        lambda_ = cfg.lambda_

        actions = rollout["actions"]
        rewards = rollout["rewards"]
        truncated = rollout["truncated"]
        terminated = rollout["terminated"]
        act_logits = rollout["act_logits"]
        vals = rollout["vals"]
        val_bootstraps = rollout["val_bootstraps"]
        mask = rollout["mask"].float()

        dist = Categorical(logits=act_logits)

        lambda_returns = compute_lambda_returns(
            rewards,
            terminated,
            truncated,
            val_bootstraps,
            gamma,
            lambda_,
        )

        advantage = lambda_returns - vals

        valid_count = mask.sum().clamp_min(1.0)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        loss_actions = (-(log_prob * advantage.detach()) * mask).sum() / valid_count

        loss_values = ((vals - lambda_returns.detach()).pow(2) * mask).sum() / valid_count
        loss_values = cfg.weight_value_loss * loss_values

        entropy = (entropy * mask).sum() / valid_count
        loss_entropy = -cfg.weight_entropy_loss * entropy

        loss_total = loss_actions + loss_values + loss_entropy

        metrics = {
            "loss_total": loss_total.detach(),
            "loss_actions": loss_actions.detach(),
            "loss_values": loss_values.detach(),
            "loss_entropy": loss_entropy.detach(),
            "policy_entropy": entropy.detach(),
            "valid_steps": valid_count.detach(),
        }

        return loss_total, metrics