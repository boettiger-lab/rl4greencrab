"""Throughput benchmark: CPU vs GPU simulator, and SB3 PPO vs GPUPPO. Run on an otherwise idle machine."""
import os
import time

import numpy as np
import pandas as pd
import torch

from common import PARAMS, RESULTS, make_env

CFG = {"random_start": True, "observation_type": "count-biomass-time", "param_df": PARAMS}
rows = []


def rec(what, setup, rate, unit="env-steps/s"):
    rows.append(dict(what=what, setup=setup, rate=rate, unit=unit))
    print(f"{what:10s} {setup:45s} {rate:14,.0f} {unit}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "benchmark.csv"), index=False)


def cpu_env_rate(n_steps=30_000):
    from rl4greencrab import TwoActNormalized
    env = TwoActNormalized(CFG)
    env.reset()
    a = np.zeros(2, dtype=np.float32)
    t = time.time()
    for i in range(n_steps):
        _, _, done, _, _ = env.step(a)
        if done:
            env.reset()
    return n_steps / (time.time() - t)


def cpu_vec_rate(n_envs, n_steps=3_000):
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from rl4greencrab import TwoActNormalized
    venv = make_vec_env(TwoActNormalized, n_envs=n_envs, env_kwargs={"config": CFG}, vec_env_cls=SubprocVecEnv)
    venv.reset()
    a = np.zeros((n_envs, 2), dtype=np.float32)
    t = time.time()
    for _ in range(n_steps):
        venv.step(a)
    r = n_envs * n_steps / (time.time() - t)
    venv.close()
    return r


def gpu_env_rate(B, n_steps=505):
    env = make_env(num_envs=B, seed=0)
    env.reset()
    a = torch.zeros(B, 2, device=env.device)
    for _ in range(50):
        env.step(a)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n_steps):
        env.step(a)
    torch.cuda.synchronize()
    return B * n_steps / (time.time() - t)


def sb3_ppo_rate(n_envs, subproc, steps):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from rl4greencrab import TwoActNormalized
    venv = make_vec_env(TwoActNormalized, n_envs=n_envs, env_kwargs={"config": CFG},
                        vec_env_cls=SubprocVecEnv if subproc else DummyVecEnv)
    model = PPO("MultiInputPolicy", venv, device="cpu", verbose=0)
    t = time.time()
    model.learn(steps)
    r = steps / (time.time() - t)
    venv.close()
    return r


def gpu_ppo_rate(steps=20_000_000):
    from rl4greencrab.agents.gpu_ppo import GPUPPO
    env = make_env(num_envs=4096, seed=0)
    model = GPUPPO(env, seed=0)
    model.learn(4096 * 128, verbose=0)  # warm-up
    torch.cuda.synchronize()
    t = time.time()
    model.learn(steps, verbose=0)
    torch.cuda.synchronize()
    return steps / (time.time() - t)


if __name__ == "__main__":
    rec("simulator", "CPU TwoActNormalized, 1 process", cpu_env_rate())
    rec("simulator", "CPU TwoActNormalized, 20 processes (SubprocVecEnv)", cpu_vec_rate(20))
    for B in [1024, 4096, 16384, 65536]:
        rec("simulator", f"GPU TwoActGPU, {B} envs", gpu_env_rate(B))
    rec("ppo", "SB3 PPO, 12 envs, DummyVecEnv (repo setup)", sb3_ppo_rate(12, False, 200_000), "steps/s")
    rec("ppo", "SB3 PPO, 20 envs, SubprocVecEnv", sb3_ppo_rate(20, True, 400_000), "steps/s")
    rec("ppo", "GPUPPO, 4096 envs", gpu_ppo_rate(), "steps/s")
