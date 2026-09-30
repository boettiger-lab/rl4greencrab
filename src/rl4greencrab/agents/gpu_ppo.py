"""
PPO that keeps the whole training loop on the GPU, for use with `TwoActGPU`.

Rollouts, advantage estimation and minibatch updates all operate on device
tensors; nothing is copied to the host except scalar logging values once per
rollout. Defaults mirror stable-baselines3's PPO (separate 64x64 tanh actor
and critic, state-independent log-std, GAE, clipped surrogate, advantage
normalization), except the rollout/minibatch sizes, which are scaled up for
thousands of parallel envs.

Trained agents expose `predict(obs, deterministic)` like an SB3 model, so they
can be evaluated with the existing CPU tools (`simulator`, `evaluate_agent`).
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from rl4greencrab.envs.gpu_env import TwoActGPU, N_MONTHS

DEFAULTS = dict(
    learning_rate=3e-4,
    n_steps=128,          # per env, per rollout
    batch_size=16384,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None,
    anneal_lr=False,      # linearly decay the learning rate to 0 over `learn()`
    squash=False,         # act with tanh(u), u ~ Gaussian, instead of clipping u to [-1, 1]
)

ACTIVATIONS = {"Tanh": nn.Tanh, "ReLU": nn.ReLU, "Sigmoid": nn.Sigmoid}


def _mlp(in_dim, hidden, out_dim, act, out_gain):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, gain=out_gain)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim=2, net_arch=(64, 64), activation="Tanh", log_std_init=0.0):
        super().__init__()
        if isinstance(net_arch, dict):
            pi_arch, vf_arch = net_arch.get("pi", [64, 64]), net_arch.get("vf", [64, 64])
        else:
            pi_arch = vf_arch = list(net_arch)
        act = ACTIVATIONS[activation] if isinstance(activation, str) else activation
        self.actor = _mlp(obs_dim, pi_arch, act_dim, act, out_gain=0.01)
        self.critic = _mlp(obs_dim, vf_arch, 1, act, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def dist(self, x):
        return torch.distributions.Normal(self.actor(x), self.log_std.exp())


class GPUPPO:
    def __init__(self, env: TwoActGPU, seed=None, tensorboard_log=None, policy_kwargs=None, **hyperparams):
        self.env = env
        self.device = env.device
        self.hp = {**DEFAULTS, **hyperparams}
        if seed is not None:
            torch.manual_seed(seed)
        policy_kwargs = dict(policy_kwargs or {})
        self.policy_kwargs = policy_kwargs
        self.policy = ActorCritic(env.flat_obs_dim, **policy_kwargs).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.hp["learning_rate"], eps=1e-5)
        self.tensorboard_log = tensorboard_log
        self.num_timesteps = 0

    def learn(self, total_timesteps, tb_log_name="GPUPPO", log_interval=1, verbose=1, callback=None):
        """`callback(model)` is called every `log_interval` updates (e.g. for held-out evaluation)."""
        env, hp, pol = self.env, self.hp, self.policy
        T, B = hp["n_steps"], env.num_envs
        writer = None
        if self.tensorboard_log:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(self.tensorboard_log, tb_log_name))

        obs_buf = torch.zeros(T, B, env.flat_obs_dim, device=self.device)
        act_buf = torch.zeros(T, B, 2, device=self.device)
        logp_buf = torch.zeros(T, B, device=self.device)
        rew_buf = torch.zeros(T, B, device=self.device)
        done_buf = torch.zeros(T, B, device=self.device)
        val_buf = torch.zeros(T, B, device=self.device)

        obs, _ = env.reset()
        x = env.flatten_obs(obs)
        ep_ret = torch.zeros(B, device=self.device)
        n_updates = max(1, total_timesteps // (T * B))
        start = time.time()

        for update in range(n_updates):
            if hp["anneal_lr"]:
                for group in self.opt.param_groups:
                    group["lr"] = hp["learning_rate"] * (1 - update / n_updates)
            ret_sum = torch.zeros((), device=self.device)
            ret_n = torch.zeros((), device=self.device)

            # ---- rollout ----
            with torch.no_grad():
                for t in range(T):
                    d = pol.dist(x)
                    a = d.sample()
                    obs_buf[t], act_buf[t] = x, a
                    # with squash, the tanh Jacobian is the same under old and new policy,
                    # so it cancels in the PPO ratio and log-probs can stay in u-space
                    logp_buf[t] = d.log_prob(a).sum(-1)
                    val_buf[t] = pol.value(x)
                    obs, r, done, _, _ = env.step(torch.tanh(a) if hp["squash"] else a)  # env clips to [-1, 1], like SB3
                    x = env.flatten_obs(obs)
                    rew_buf[t], done_buf[t] = r, done.float()
                    ep_ret += r
                    ret_sum += (ep_ret * done).sum()
                    ret_n += done.sum()
                    ep_ret = ep_ret * (~done)

                # ---- GAE (episode end is a true termination, as in the CPU env) ----
                adv = torch.zeros_like(rew_buf)
                last = torch.zeros(B, device=self.device)
                next_val = pol.value(x)
                for t in reversed(range(T)):
                    nonterminal = 1.0 - done_buf[t]
                    nv = next_val if t == T - 1 else val_buf[t + 1]
                    delta = rew_buf[t] + hp["gamma"] * nv * nonterminal - val_buf[t]
                    last = delta + hp["gamma"] * hp["gae_lambda"] * nonterminal * last
                    adv[t] = last
                ret = adv + val_buf

            # ---- update ----
            b_obs, b_act = obs_buf.view(T * B, -1), act_buf.view(T * B, 2)
            b_logp, b_adv, b_ret = logp_buf.view(-1), adv.view(-1), ret.view(-1)
            N, mb = T * B, min(hp["batch_size"], T * B)
            for epoch in range(hp["n_epochs"]):
                perm = torch.randperm(N, device=self.device)
                kls = []
                for i in range(0, N - mb + 1, mb):
                    j = perm[i:i + mb]
                    d = pol.dist(b_obs[j])
                    logp = d.log_prob(b_act[j]).sum(-1)
                    ratio = (logp - b_logp[j]).exp()
                    A = b_adv[j]
                    A = (A - A.mean()) / (A.std() + 1e-8)
                    pg_loss = -torch.min(ratio * A, ratio.clamp(1 - hp["clip_range"], 1 + hp["clip_range"]) * A).mean()
                    v_loss = 0.5 * (pol.value(b_obs[j]) - b_ret[j]).pow(2).mean()
                    entropy = d.entropy().sum(-1).mean()
                    loss = pg_loss + hp["vf_coef"] * v_loss - hp["ent_coef"] * entropy
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(pol.parameters(), hp["max_grad_norm"])
                    self.opt.step()
                    with torch.no_grad():
                        kls.append(((ratio - 1) - (logp - b_logp[j])).mean())
                if hp["target_kl"] is not None and torch.stack(kls).mean().item() > 1.5 * hp["target_kl"]:
                    break

            self.num_timesteps += T * B
            if (update + 1) % log_interval == 0 or update == n_updates - 1:
                ep_mean = (ret_sum / ret_n.clamp(min=1)).item() if ret_n.item() > 0 else float("nan")
                sps = self.num_timesteps / (time.time() - start)
                if writer:
                    writer.add_scalar("rollout/ep_rew_mean", ep_mean, self.num_timesteps)
                    writer.add_scalar("train/policy_loss", pg_loss.item(), self.num_timesteps)
                    writer.add_scalar("train/value_loss", v_loss.item(), self.num_timesteps)
                    writer.add_scalar("train/std", pol.log_std.exp().mean().item(), self.num_timesteps)
                    writer.add_scalar("time/fps", sps, self.num_timesteps)
                if verbose:
                    print(f"[{self.num_timesteps:>11,d}] ep_rew_mean {ep_mean:8.3f}  "
                          f"std {pol.log_std.exp().mean().item():.3f}  {sps:,.0f} steps/s", flush=True)
                if callback is not None:
                    callback(self)
        if writer:
            writer.close()
        return self

    # ---- inference ----

    def predict(self, observation, state=None, episode_start=None, deterministic=True, flat=False):
        """SB3-style predict for a single (numpy dict) observation, a batch of GPU obs dicts,
        or (flat=True) a batch of already-flattened observation tensors."""
        if flat:
            with torch.no_grad():
                d = self.policy.dist(observation)
                a = d.mean if deterministic else d.sample()
                return (torch.tanh(a) if self.hp.get("squash") else a.clamp(-1, 1)), None
        single = not torch.is_tensor(observation["crabs"]) and np.ndim(observation["crabs"]) == 1
        obs = {k: torch.as_tensor(np.asarray(v) if not torch.is_tensor(v) else v, device=self.device)
               for k, v in observation.items()}
        if single:
            obs = {k: v.unsqueeze(0) for k, v in obs.items()}
        x = obs["crabs"].float()
        if "months" in obs:
            x = torch.cat([x, nn.functional.one_hot(obs["months"].long().view(-1), N_MONTHS).float()], -1)
        with torch.no_grad():
            d = self.policy.dist(x)
            a = d.mean if deterministic else d.sample()
            a = torch.tanh(a) if self.hp.get("squash") else a.clamp(-1, 1)
        if single:
            return a[0].cpu().numpy(), None
        return a, None

    def save(self, path):
        path = path if path.endswith(".pt") else path + ".pt"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "obs_dim": self.env.flat_obs_dim,
            "policy_kwargs": self.policy_kwargs,
            "hyperparams": self.hp,
            "env_config": {k: v for k, v in self.env.config.items() if k != "param_df"},
        }, path)
        return path

    @classmethod
    def load(cls, path, env=None, device=None):
        """Load a saved agent. Without `env`, the agent can only `predict()`."""
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        agent = cls.__new__(cls)
        agent.env = env
        agent.device = torch.device(device or (env.device if env else "cpu"))
        agent.hp = ckpt["hyperparams"]
        agent.policy_kwargs = ckpt["policy_kwargs"]
        agent.policy = ActorCritic(ckpt["obs_dim"], **agent.policy_kwargs).to(agent.device)
        agent.policy.load_state_dict(ckpt["policy"])
        agent.opt = torch.optim.Adam(agent.policy.parameters(), lr=agent.hp["learning_rate"], eps=1e-5)
        agent.tensorboard_log = None
        agent.num_timesteps = 0
        agent.env_config = ckpt["env_config"]
        return agent


@torch.no_grad()
def gpu_evaluate(policy_fn, env: TwoActGPU, n_episodes=None, deterministic=True):
    """
    Run `env.num_envs` (or `n_episodes` <= num_envs) full episodes in parallel on the GPU.
    `policy_fn(obs_dict) -> (B, 2) action tensor`, e.g. `lambda o: agent.predict(o)[0]`.
    Returns a (n_episodes,) numpy array of episode returns.
    """
    obs, _ = env.reset()
    total = torch.zeros(env.num_envs, device=env.device)
    for _ in range(env.Tmax + 1):
        obs, r, done, _, _ = env.step(policy_fn(obs))
        total += r
    return total[: n_episodes or env.num_envs].cpu().numpy()
