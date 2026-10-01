"""
Memory-based PPO for TwoActGPU, entirely on the GPU.

All envs start together and every episode lasts exactly Tmax + 1 steps, so a
rollout of n_steps = Tmax + 1 is one batch of complete, aligned episodes. The
policy's memory therefore starts empty at the beginning of every rollout, and
it is trained on whole episodes; minibatches are groups of envs (complete
sequences), not individual steps.

Options:
  actor_type   "gru" (default) or "transformer" (causal self-attention over the
               episode so far, with GTrXL-style gated residuals)
  critic_type  "recurrent" (default; a separate GRU on observations) or
               "privileged" (an MLP on the observation plus env.privileged():
               true scenario, population and posterior draw -- training only,
               the deployed actor never sees it)
  aux_coef     weight of an auxiliary loss training the actor's memory to
               predict env.privileged() from the observation history (0 = off)
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from rl4greencrab.utils.precision import configure_tf32

from rl4greencrab.agents.gpu_ppo import DEFAULTS as PPO_DEFAULTS

# TF32 "auto": on where supported (measured +17-18% for GRU/transformer training on GB10)
DEFAULTS = {**PPO_DEFAULTS, "n_seq_minibatches": 8, "aux_coef": 0.0, "tf32": "auto"}


def _init_linear(modules, out_gain):
    linears = [m for m in modules if isinstance(m, nn.Linear)]
    for m in linears:
        nn.init.orthogonal_(m.weight, np.sqrt(2))
        nn.init.zeros_(m.bias)
    nn.init.orthogonal_(linears[-1].weight, out_gain)


def _head(d, head, out_dim):
    layers = []
    for h in head:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class GRUNet(nn.Module):
    """obs -> Linear+tanh -> GRU -> MLP head."""

    def __init__(self, obs_dim, out_dim, hidden=128, embed=64, head=(64,), out_gain=1.0):
        super().__init__()
        self.hidden = hidden
        self.feat_dim = hidden
        self.embed = nn.Sequential(nn.Linear(obs_dim, embed), nn.Tanh())
        self.gru = nn.GRU(embed, hidden)
        self.head = _head(hidden, head, out_dim)
        _init_linear(list(self.embed) + list(self.head), out_gain)

    def init_state(self, B, device):
        return torch.zeros(1, B, self.hidden, device=device)

    def seq(self, x, state=None):
        """x: (T, B, D) -> out (T, B, out_dim), features (T, B, H), final state"""
        if state is None:
            state = self.init_state(x.shape[1], x.device)
        y, state = self.gru(self.embed(x), state)
        return self.head(y), y, state

    def step(self, x, state):
        out, _, state = self.seq(x.unsqueeze(0), state)
        return out[0], state

    def forward(self, x, h):  # backward-compatible call signature
        out, _, h = self.seq(x, h)
        return out, h


class GatedBlock(nn.Module):
    """Pre-LN transformer block with GRU-type gating in place of residual adds (GTrXL, Parisotto et al. 2019)."""

    def __init__(self, d, heads, ff, gate_bias=2.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.g1, self.g2 = self._gate(d, gate_bias), self._gate(d, gate_bias)

    @staticmethod
    def _gate(d, bias):
        g = nn.ModuleDict({k: nn.Linear(d, d, bias=(k == "Wz")) for k in ("Wr", "Ur", "Wz", "Uz", "Wg", "Ug")})
        # positive bias on the update gate: start near the identity map (x passes through)
        nn.init.constant_(g["Wz"].bias, -bias)
        return g

    @staticmethod
    def _gated(g, x, y):
        r = torch.sigmoid(g["Wr"](y) + g["Ur"](x))
        z = torch.sigmoid(g["Wz"](y) + g["Uz"](x))
        h = torch.tanh(g["Wg"](y) + g["Ug"](r * x))
        return (1 - z) * x + z * h

    def forward(self, x, mask):
        h = self.ln1(x)
        y, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        return self._post(x, y)

    def step(self, x, K, V, t):
        """
        Incremental causal step for position t: x (B, 1, d). K, V (B, heads, max_len, d/heads) are
        preallocated projected keys/values for this layer; position t is written in place, so memory
        stays constant over the episode. Uses the same weights as the full-sequence path.
        """
        B, _, d = x.shape
        H = self.attn.num_heads
        q, k, v = nn.functional.linear(self.ln1(x), self.attn.in_proj_weight, self.attn.in_proj_bias).chunk(3, -1)
        split = lambda z: z.view(B, 1, H, d // H).transpose(1, 2)  # (B, H, 1, dh)
        K[:, :, t:t + 1], V[:, :, t:t + 1] = split(k), split(v)
        y = nn.functional.scaled_dot_product_attention(split(q), K[:, :, :t + 1], V[:, :, :t + 1])
        y = self.attn.out_proj(y.transpose(1, 2).reshape(B, 1, d))
        return self._post(x, y)

    def _post(self, x, y):
        x = self._gated(self.g1, x, torch.relu(y))
        return self._gated(self.g2, x, torch.relu(self.ff(self.ln2(x))))


class TransformerNet(nn.Module):
    """Causal gated transformer over the episode so far (learned position embedding)."""

    def __init__(self, obs_dim, out_dim, d=64, layers=2, heads=4, ff=128, max_len=128, head=(64,), out_gain=1.0):
        super().__init__()
        self.feat_dim, self.max_len = d, max_len
        self.inp = nn.Linear(obs_dim, d)
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        self.blocks = nn.ModuleList(GatedBlock(d, heads, ff) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = _head(d, head, out_dim)
        _init_linear(list(self.head), out_gain)

    def init_state(self, B, device):
        # (t, per-layer preallocated keys/values); earlier positions never change under a causal mask
        d = self.pos.shape[1]
        kv = lambda blk: torch.zeros(B, blk.attn.num_heads, self.max_len, d // blk.attn.num_heads, device=device)
        return (0, tuple((kv(b), kv(b)) for b in self.blocks))

    def _encode(self, x):
        """x: (B, T, D) -> features (B, T, d)"""
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h = self.inp(x) + self.pos[:T]
        for blk in self.blocks:
            h = blk(h, mask)
        return self.ln(h)

    def seq(self, x, state=None):
        feats = self._encode(x.transpose(0, 1)).transpose(0, 1)  # (T, B, d)
        return self.head(feats), feats, None

    def step(self, x, state):
        t, kvs = state
        h = (self.inp(x) + self.pos[t]).unsqueeze(1)
        for blk, (K, V) in zip(self.blocks, kvs):
            h = blk.step(h, K, V, t)
        return self.head(self.ln(h[:, 0])), (t + 1, kvs)


class PrivilegedCritic(nn.Module):
    """Feed-forward value function on (observation, privileged state)."""

    def __init__(self, obs_dim, priv_dim, hidden=(256, 256)):
        super().__init__()
        layers, d = [], obs_dim + priv_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        _init_linear(list(self.net), 1.0)

    def seq(self, x, priv):
        return self.net(torch.cat([x, priv], -1))


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=2, hidden=128, embed=64, head=(64,), log_std_init=0.0,
                 actor_type="gru", critic_type="recurrent", priv_dim=0, aux=False, transformer=None):
        super().__init__()
        self.actor_type, self.critic_type = actor_type, critic_type
        if actor_type == "gru":
            self.actor = GRUNet(obs_dim, act_dim, hidden, embed, head, out_gain=0.01)
        elif actor_type == "transformer":
            self.actor = TransformerNet(obs_dim, act_dim, head=head, out_gain=0.01, **(transformer or {}))
        else:
            raise ValueError(actor_type)
        if critic_type == "recurrent":
            self.critic = GRUNet(obs_dim, 1, hidden, embed, head, out_gain=1.0)
        elif critic_type == "privileged":
            self.critic = PrivilegedCritic(obs_dim, priv_dim)
        else:
            raise ValueError(critic_type)
        self.aux_head = nn.Linear(self.actor.feat_dim, priv_dim) if aux else None
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def value_seq(self, x, priv):
        """x: (T, B, D), priv: (T, B, P) -> values (T, B)"""
        if self.critic_type == "privileged":
            return self.critic.seq(x, priv)[..., 0]
        return self.critic.seq(x)[0][..., 0]

    def dist(self, mean):
        return torch.distributions.Normal(mean, self.log_std.exp())


class GPURecurrentPPO:
    def __init__(self, env, seed=None, tensorboard_log=None, policy_kwargs=None, **hyperparams):
        self.env, self.device = env, env.device
        self.hp = {**DEFAULTS, **hyperparams}
        # TF32 tensor cores for the networks if requested and supported (falls back to fp32);
        # the simulator's own matmuls always run in full fp32
        self.tf32_enabled = configure_tf32(self.hp.get("tf32", False), env.device)
        self.hp["n_steps"] = env.Tmax + 1  # whole, aligned episodes per rollout
        if seed is not None:
            torch.manual_seed(seed)
        kw = dict(policy_kwargs or {})
        if kw.get("critic_type") == "privileged" or self.hp["aux_coef"] > 0:
            kw["priv_dim"] = env.privileged_dim
        kw["aux"] = self.hp["aux_coef"] > 0
        self.policy_kwargs = kw
        self.policy = RecurrentActorCritic(env.flat_obs_dim, **kw).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.hp["learning_rate"], eps=1e-5)
        self.tensorboard_log = tensorboard_log
        self.num_timesteps = 0

    def learn(self, total_timesteps, log_interval=1, verbose=1, callback=None):
        env, hp, pol = self.env, self.hp, self.policy
        T, B, D = hp["n_steps"], env.num_envs, env.flat_obs_dim
        use_priv = pol.critic_type == "privileged" or pol.aux_head is not None
        P = env.privileged_dim if use_priv else 0
        obs_buf = torch.zeros(T, B, D, device=self.device)
        priv_buf = torch.zeros(T, B, P, device=self.device)
        act_buf = torch.zeros(T, B, 2, device=self.device)
        logp_buf, rew_buf, done_buf = (torch.zeros(T, B, device=self.device) for _ in range(3))
        n_updates = max(1, total_timesteps // (T * B))
        start = time.time()

        for update in range(n_updates):
            if hp["anneal_lr"]:
                for g in self.opt.param_groups:
                    g["lr"] = hp["learning_rate"] * (1 - update / n_updates)

            # ---- rollout: one full episode per env ----
            obs, _ = env.reset()
            x = env.flatten_obs(obs)
            state = pol.actor.init_state(B, self.device)
            with torch.no_grad():
                for t in range(T):
                    if use_priv:
                        priv_buf[t] = env.privileged()
                    mean, state = pol.actor.step(x, state)
                    d = pol.dist(mean)
                    a = d.sample()
                    obs_buf[t], act_buf[t] = x, a
                    logp_buf[t] = d.log_prob(a).sum(-1)
                    obs, r, done, _, _ = env.step(a)
                    x = env.flatten_obs(obs)
                    rew_buf[t], done_buf[t] = r, done.float()
                assert bool(done_buf[-1].all()), "episodes must be aligned with rollouts"
                val_buf = pol.value_seq(obs_buf, priv_buf)

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
                    mean, feats, _ = pol.actor.seq(obs_buf[:, j])
                    v = pol.value_seq(obs_buf[:, j], priv_buf[:, j])
                    d = pol.dist(mean)
                    logp = d.log_prob(act_buf[:, j]).sum(-1)
                    ratio = (logp - logp_buf[:, j]).exp()
                    A = adv[:, j]
                    A = (A - A.mean()) / (A.std() + 1e-8)
                    pg_loss = -torch.min(ratio * A, ratio.clamp(1 - hp["clip_range"], 1 + hp["clip_range"]) * A).mean()
                    v_loss = 0.5 * (v - ret[:, j]).pow(2).mean()
                    loss = pg_loss + hp["vf_coef"] * v_loss - hp["ent_coef"] * d.entropy().sum(-1).mean()
                    if pol.aux_head is not None:
                        loss = loss + hp["aux_coef"] * (pol.aux_head(feats) - priv_buf[:, j]).pow(2).mean()
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
        self.state = None

    def reset(self, B):
        self.state = self.policy.actor.init_state(B, next(self.policy.parameters()).device)

    @torch.no_grad()
    def __call__(self, obs):
        x = obs if torch.is_tensor(obs) else self.flatten(obs)
        mean, self.state = self.policy.actor.step(x, self.state)
        a = mean if self.deterministic else self.policy.dist(mean).sample()
        return a.clamp(-1, 1)
