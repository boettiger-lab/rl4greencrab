"""
Recurrent (GRU) PPO for TwoActGPU, entirely on the GPU.

All envs start together and every episode lasts exactly Tmax + 1 steps, so a
rollout of n_steps = Tmax + 1 is one batch of complete, aligned episodes. The
GRU hidden state therefore starts at zero at the beginning of every rollout,
and the policy is trained by backpropagating through whole episodes;
minibatches are groups of envs (complete sequences), not individual steps.

Actor and critic have separate GRUs (as in sb3-contrib's RecurrentPPO default).
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from rl4greencrab.agents.gpu_ppo import DEFAULTS as PPO_DEFAULTS

DEFAULTS = {**PPO_DEFAULTS, "n_seq_minibatches": 8}


class GRUNet(nn.Module):
    """obs -> Linear+tanh -> GRU -> MLP head."""

    def __init__(self, obs_dim, out_dim, hidden=128, embed=64, head=(64,), out_gain=1.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(obs_dim, embed), nn.Tanh())
        self.gru = nn.GRU(embed, hidden)
        layers, d = [], hidden
        for h in head:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.head = nn.Sequential(*layers)
        for m in list(self.embed) + list(self.head):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head[-1].weight, out_gain)

    def forward(self, x, h):
        """x: (T, B, obs_dim), h: (1, B, hidden) -> (T, B, out_dim), h_T"""
        y, h = self.gru(self.embed(x), h)
        return self.head(y), h


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=2, hidden=128, embed=64, head=(64,), log_std_init=0.0):
        super().__init__()
        self.hidden = hidden
        self.actor = GRUNet(obs_dim, act_dim, hidden, embed, head, out_gain=0.01)
        self.critic = GRUNet(obs_dim, 1, hidden, embed, head, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def init_state(self, B, device):
        z = torch.zeros(1, B, self.hidden, device=device)
        return z, z.clone()

    def dist(self, mean):
        return torch.distributions.Normal(mean, self.log_std.exp())


class GPURecurrentPPO:
    def __init__(self, env, seed=None, tensorboard_log=None, policy_kwargs=None, **hyperparams):
        self.env, self.device = env, env.device
        self.hp = {**DEFAULTS, **hyperparams}
        self.hp["n_steps"] = env.Tmax + 1  # whole, aligned episodes per rollout
        if seed is not None:
            torch.manual_seed(seed)
        self.policy_kwargs = dict(policy_kwargs or {})
        self.policy = RecurrentActorCritic(env.flat_obs_dim, **self.policy_kwargs).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.hp["learning_rate"], eps=1e-5)
        self.tensorboard_log = tensorboard_log
        self.num_timesteps = 0

    def learn(self, total_timesteps, log_interval=1, verbose=1, callback=None):
        env, hp, pol = self.env, self.hp, self.policy
        T, B, D = hp["n_steps"], env.num_envs, env.flat_obs_dim
        obs_buf = torch.zeros(T, B, D, device=self.device)
        act_buf = torch.zeros(T, B, 2, device=self.device)
        logp_buf, rew_buf, val_buf, done_buf = (torch.zeros(T, B, device=self.device) for _ in range(4))
        n_updates = max(1, total_timesteps // (T * B))
        start = time.time()

        for update in range(n_updates):
            if hp["anneal_lr"]:
                for g in self.opt.param_groups:
                    g["lr"] = hp["learning_rate"] * (1 - update / n_updates)

            # ---- rollout: one full episode per env ----
            obs, _ = env.reset()
            x = env.flatten_obs(obs)
            ha, hc = pol.init_state(B, self.device)
            with torch.no_grad():
                for t in range(T):
                    mean, ha = pol.actor(x.unsqueeze(0), ha)
                    v, hc = pol.critic(x.unsqueeze(0), hc)
                    d = pol.dist(mean[0])
                    a = d.sample()
                    obs_buf[t], act_buf[t], val_buf[t] = x, a, v[0, :, 0]
                    logp_buf[t] = d.log_prob(a).sum(-1)
                    obs, r, done, _, _ = env.step(a)
                    x = env.flatten_obs(obs)
                    rew_buf[t], done_buf[t] = r, done.float()
                assert bool(done_buf[-1].all()), "episodes must be aligned with rollouts"

                adv = torch.zeros_like(rew_buf)
                last = torch.zeros(B, device=self.device)
                for t in reversed(range(T)):
                    nonterminal = 1.0 - done_buf[t]
                    nv = val_buf[t + 1] if t < T - 1 else torch.zeros(B, device=self.device)
                    delta = rew_buf[t] + hp["gamma"] * nv * nonterminal - val_buf[t]
                    last = delta + hp["gamma"] * hp["gae_lambda"] * nonterminal * last
                    adv[t] = last
                ret = adv + val_buf
            ep_mean = rew_buf.sum(0).mean().item()

            # ---- update: minibatches of whole sequences ----
            nmb = min(hp["n_seq_minibatches"], B)
            for epoch in range(hp["n_epochs"]):
                perm = torch.randperm(B, device=self.device)
                for j in perm.chunk(nmb):
                    h0a, h0c = pol.init_state(len(j), self.device)
                    mean, _ = pol.actor(obs_buf[:, j], h0a)
                    v, _ = pol.critic(obs_buf[:, j], h0c)
                    d = pol.dist(mean)
                    logp = d.log_prob(act_buf[:, j]).sum(-1)
                    ratio = (logp - logp_buf[:, j]).exp()
                    A = adv[:, j]
                    A = (A - A.mean()) / (A.std() + 1e-8)
                    pg_loss = -torch.min(ratio * A, ratio.clamp(1 - hp["clip_range"], 1 + hp["clip_range"]) * A).mean()
                    v_loss = 0.5 * (v[..., 0] - ret[:, j]).pow(2).mean()
                    loss = pg_loss + hp["vf_coef"] * v_loss - hp["ent_coef"] * d.entropy().sum(-1).mean()
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(pol.parameters(), hp["max_grad_norm"])
                    self.opt.step()

            self.num_timesteps += T * B
            if (update + 1) % log_interval == 0 or update == n_updates - 1:
                if verbose:
                    print(f"[{self.num_timesteps:>11,d}] ep_rew_mean {ep_mean:8.3f}  std {pol.log_std.exp().mean().item():.3f}  "
                          f"{self.num_timesteps / (time.time() - start):,.0f} steps/s", flush=True)
                if callback is not None:
                    callback(self)
        return self

    def save(self, path):
        path = path if path.endswith(".pt") else path + ".pt"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"recurrent": True, "policy": self.policy.state_dict(), "obs_dim": self.env.flat_obs_dim,
                    "policy_kwargs": self.policy_kwargs, "hyperparams": self.hp}, path)
        return path

    @staticmethod
    def load_policy(path, device="cuda"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        pol = RecurrentActorCritic(ckpt["obs_dim"], **ckpt["policy_kwargs"]).to(device)
        pol.load_state_dict(ckpt["policy"])
        return pol


class RecurrentActor:
    """Stateful batched policy for evaluation: call reset(B) at episode start, then call on flat obs each step."""

    def __init__(self, policy, flatten=None, deterministic=True):
        self.policy, self.flatten, self.deterministic = policy, flatten, deterministic
        self.h = None

    def reset(self, B):
        self.h = self.policy.init_state(B, next(self.policy.parameters()).device)[0]

    @torch.no_grad()
    def __call__(self, obs):
        x = obs if torch.is_tensor(obs) else self.flatten(obs)
        mean, self.h = self.policy.actor(x.unsqueeze(0), self.h)
        a = mean[0] if self.deterministic else self.policy.dist(mean[0]).sample()
        return a.clamp(-1, 1)
