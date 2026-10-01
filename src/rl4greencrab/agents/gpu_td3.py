"""
TD3 for massively parallel GPU simulators (in the spirit of FastTD3, Seo et al. 2025).

Everything, including the replay buffer, lives on the GPU. Many envs step in
parallel; transitions are stored as n-step returns; each env step is followed
by `updates_per_step` large-batch critic updates (twin critics, clipped double
Q, target policy smoothing) and delayed actor updates. Exploration noise std is
drawn per env from [noise_min, noise_max].
"""

import copy
import time

import torch
import torch.nn as nn

from rl4greencrab.utils.precision import configure_tf32

DEFAULTS = dict(
    learning_rate=3e-4,
    buffer_size=2_000_000,
    batch_size=8192,
    gamma=0.99,
    tau=0.1,
    n_step=3,
    updates_per_step=2,
    policy_delay=2,
    policy_noise=0.1,
    noise_clip=0.3,
    noise_min=0.01,
    noise_max=0.4,
    learning_starts=10,     # env steps (per env) before updating
    net_arch=(256, 256),
    tf32="auto",
)


def mlp(i, hidden, o, out_act=None):
    layers = []
    for h in hidden:
        layers += [nn.Linear(i, h), nn.LayerNorm(h), nn.ReLU()]
        i = h
    layers.append(nn.Linear(i, o))
    if out_act:
        layers.append(out_act)
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, hidden, 1)
        self.q2 = mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, x, a):
        xa = torch.cat([x, a], -1)
        return self.q1(xa).squeeze(-1), self.q2(xa).squeeze(-1)


class GPUTD3:
    def __init__(self, env, seed=None, **hyperparams):
        self.env, self.device = env, env.device
        self.tf32_enabled = configure_tf32(self.hp["tf32"], env.device)  # TF32 for the large-batch updates if supported
        self.hp = hp = {**DEFAULTS, **hyperparams}
        if seed is not None:
            torch.manual_seed(seed)
        d, h = env.flat_obs_dim, list(hp["net_arch"])
        self.actor = mlp(d, h, 2, nn.Tanh()).to(self.device)
        self.critic = Critic(d, 2, h).to(self.device)
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=hp["learning_rate"], weight_decay=0.0)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=hp["learning_rate"], weight_decay=0.0)
        cap = hp["buffer_size"]
        z = lambda *s: torch.zeros(cap, *s, device=self.device)
        self.buf = dict(obs=z(d), act=z(2), ret=z(), next_obs=z(d), disc=z())
        self.ptr, self.full = 0, False
        self.num_timesteps, self.n_updates = 0, 0

    # ---- replay ----
    def _add(self, obs, act, ret, next_obs, disc):
        n = obs.shape[0]
        cap = self.hp["buffer_size"]
        i = (torch.arange(n, device=self.device) + self.ptr) % cap
        for k, v in zip(("obs", "act", "ret", "next_obs", "disc"), (obs, act, ret, next_obs, disc)):
            self.buf[k][i] = v
        self.ptr = (self.ptr + n) % cap
        self.full = self.full or self.ptr < n

    def _sample(self):
        size = self.hp["buffer_size"] if self.full else self.ptr
        i = torch.randint(0, size, (self.hp["batch_size"],), device=self.device)
        return {k: v[i] for k, v in self.buf.items()}

    # ---- update ----
    def _update(self):
        hp, b = self.hp, self._sample()
        with torch.no_grad():
            noise = (torch.randn_like(b["act"]) * hp["policy_noise"]).clamp(-hp["noise_clip"], hp["noise_clip"])
            a2 = (self.actor_t(b["next_obs"]) + noise).clamp(-1, 1)
            q1t, q2t = self.critic_t(b["next_obs"], a2)
            target = b["ret"] + b["disc"] * torch.min(q1t, q2t)
        q1, q2 = self.critic(b["obs"], b["act"])
        critic_loss = (q1 - target).pow(2).mean() + (q2 - target).pow(2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()
        self.n_updates += 1
        if self.n_updates % hp["policy_delay"] == 0:
            actor_loss = -self.critic(b["obs"], self.actor(b["obs"]))[0].mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_opt.step()
            with torch.no_grad():
                for net, tgt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.lerp_(p, hp["tau"])
        return critic_loss

    def learn(self, total_timesteps, log_interval=100, callback=None, verbose=1):
        env, hp = self.env, self.hp
        B, n, g = env.num_envs, hp["n_step"], hp["gamma"]
        obs, _ = env.reset()
        x = env.flatten_obs(obs)
        sigma = hp["noise_min"] + torch.rand(B, 1, device=self.device) * (hp["noise_max"] - hp["noise_min"])
        window = []  # last n (x, a, r, done, x_next)
        start, it = time.time(), 0
        while self.num_timesteps < total_timesteps:
            with torch.no_grad():
                a = (self.actor(x) + sigma * torch.randn(B, 2, device=self.device)).clamp(-1, 1)
            obs, r, done, _, info = env.step(a)
            x2 = env.flatten_obs(obs)
            window.append((x, a, r, done.float(), x2))
            if len(window) == n:
                x0, a0 = window[0][0], window[0][1]
                ret = torch.zeros(B, device=self.device)
                alive = torch.ones(B, device=self.device)
                nxt = window[-1][4]
                for k, (_, _, rk, dk, xk) in enumerate(window):
                    ret += alive * (g ** k) * rk
                    alive = alive * (1 - dk)
                disc = alive * g ** n
                self._add(x0, a0, ret, nxt, disc)
                window.pop(0)
            if done.any():
                # (n-step windows that straddle a reset are cut off by `alive` above)
                sigma = torch.where(done.unsqueeze(-1), hp["noise_min"] + torch.rand(B, 1, device=self.device)
                                    * (hp["noise_max"] - hp["noise_min"]), sigma)
            x = x2
            self.num_timesteps += B
            it += 1
            if it > hp["learning_starts"]:
                for _ in range(hp["updates_per_step"]):
                    loss = self._update()
            if it % log_interval == 0:
                if verbose:
                    print(f"[{self.num_timesteps:>11,d}] critic_loss {loss.item():.4f}  "
                          f"{self.num_timesteps / (time.time() - start):,.0f} steps/s", flush=True)
                if callback is not None:
                    callback(self)
        return self

    @torch.no_grad()
    def act(self, x):
        return self.actor(x)

    def save(self, path):
        path = path if path.endswith(".pt") else path + ".pt"
        torch.save({"actor": self.actor.state_dict(), "hp": self.hp, "obs_dim": self.env.flat_obs_dim}, path)
        return path
