"""Shared helpers for the GPU experiments: env construction, held-out evaluation, baselines."""

import os

import numpy as np
import pandas as pd
import torch

from rl4greencrab.envs.gpu_env import TwoActGPU

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(os.path.dirname(__file__), "results")
PARAMS = pd.read_csv(os.path.join(ROOT, "data", "posterior", "params.csv"))
HF_AGENTS = "/home/jovyan/rl4greencrab_hf/rl4greencrab/saved_agents"
EVAL_SEED = 2026

# Invasion-scenario distributions. NOMINAL is the manuscript model (only the
# initial adult abundance varies, U[0, 2000]); WIDE spans much broader
# uncertainty about the invasion.
NOMINAL = {"init_n_adult": [0, 2000], "K": [25000, 25000], "r": [1, 1], "mig_scale": [1, 1], "p_big": [0.2, 0.2]}
WIDE = {"init_n_adult": [0, 20000], "K": [10000, 60000, "log"], "r": [0.5, 2.0],
        "mig_scale": [0.1, 5.0, "log"], "p_big": [0.0, 0.5]}

# Oversampling high migrant pressure (the regime where wide-trained generalists fail):
WIDE_HIMIG = {**WIDE, "mig_scale": [2.5, 5.0, "log"]}
MIX_HIMIG = [[0.5, WIDE], [0.5, WIDE_HIMIG]]          # half the episodes from the high-migration corner
WIDE_LINMIG = {**WIDE, "mig_scale": [0.1, 5.0]}        # uniform instead of log-uniform migrant pressure


def interp_ranges(a, b, f):
    """Ranges a fraction f of the way from distribution a to b (geometric for log ranges)."""
    out = {}
    for k in b:
        lo0, hi0, lo1, hi1 = a[k][0], a[k][1], b[k][0], b[k][1]
        if len(b[k]) > 2 and b[k][2] == "log":
            out[k] = [lo0 ** (1 - f) * lo1 ** f, hi0 ** (1 - f) * hi1 ** f, "log"]
        else:
            out[k] = [lo0 + f * (lo1 - lo0), hi0 + f * (hi1 - hi0)]
    return out


def test_scenarios(n=24, seed=0):
    """Fixed point scenarios drawn from WIDE (plus the nominal midpoint first)."""
    g = np.random.default_rng(seed)
    pts = [{"init_n_adult": 1000.0, "K": 25000.0, "r": 1.0, "mig_scale": 1.0, "p_big": 0.2}]
    for _ in range(n):
        pt = {}
        for k, rng in WIDE.items():
            u = g.uniform()
            pt[k] = float(np.exp(np.log(rng[0]) + u * np.log(rng[1] / rng[0])) if len(rng) > 2 else rng[0] + u * (rng[1] - rng[0]))
        pts.append(pt)
    return pts


def point(pt):
    return {k: [v, v] for k, v in pt.items()}
os.makedirs(RESULTS, exist_ok=True)


def make_env(obs_type="count-biomass-time", num_envs=4096, seed=0, cuda_graph=False, **overrides):
    cfg = {"random_start": True, "observation_type": obs_type, "param_df": PARAMS, "reset_recruits": True}
    cfg.update(overrides)
    return TwoActGPU(cfg, num_envs=num_envs, seed=seed, cuda_graph=cuda_graph)


@torch.no_grad()
def episode_returns(policy_fn, env, seed=EVAL_SEED):
    """Run one synchronized batch of episodes (one per env) and return per-episode returns (tensor)."""
    obs, _ = env.reset(seed=seed)
    if hasattr(policy_fn, "reset"):
        policy_fn.reset(env.num_envs)
    total = torch.zeros(env.num_envs, device=env.device)
    for _ in range(env.Tmax + 1):
        obs, r, done, _, _ = env.step(policy_fn(obs))
        total += r
    return total


def evaluate(policy_fn, obs_type="count-biomass-time", n=50_000, seed=EVAL_SEED, wrap=None, **overrides):
    """Held-out evaluation: n episodes with a fixed seed (common random numbers across policies)."""
    env = make_env(obs_type, num_envs=n, seed=seed, **overrides)
    if wrap is not None:
        env = wrap(env)
    if not env.reset_recruits:
        # buggy CPU-env behavior only shows from the 2nd episode: warm up recruit carry-over
        episode_returns(policy_fn, env, seed=seed + 1)
    r = episode_returns(policy_fn, env, seed=seed).cpu().numpy()
    return float(r.mean()), float(r.std() / np.sqrt(len(r))), r


def constant_grid(actions, n_eps=4096, seed=EVAL_SEED, chunk=65536, **overrides):
    """Mean return of each constant (normalized) action in `actions` (k, 2)."""
    actions = torch.as_tensor(actions, dtype=torch.float32)
    per = max(1, chunk // n_eps)
    out = []
    for i in range(0, len(actions), per):
        a = actions[i:i + per]
        env = make_env(num_envs=len(a) * n_eps, seed=seed, **overrides)
        A = a.to(env.device).repeat_interleave(n_eps, 0)
        if not env.reset_recruits:
            episode_returns(lambda o: A, env, seed=seed + 1)
        r = episode_returns(lambda o: A, env, seed=seed)
        out.append(r.view(len(a), n_eps).mean(1).cpu())
    return torch.cat(out).numpy()


class SB3Policy:
    """Batched GPU inference for a saved stable-baselines3 agent (deterministic)."""
    def __init__(self, path, device="cuda"):
        from sb3_contrib import TQC
        from stable_baselines3 import PPO, TD3
        algo = {"PPO": PPO, "TD3": TD3, "TQC": TQC}[os.path.basename(path).split("-")[0]]
        self.model = algo.load(path, device=device)

    @torch.no_grad()
    def __call__(self, obs):
        o = {k: v for k, v in obs.items() if k in self.model.observation_space.spaces}
        return self.model.policy._predict(o, deterministic=True).clamp(-1, 1)
