"""Reference: the original SB3 setup (hyperpars/*/ppo.yaml: PPO, 12 envs, SB3 defaults) on the fixed env, CPU only."""
import sys
import time

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from common import PARAMS, RESULTS, SB3Policy, evaluate
from rl4greencrab.envs.sb3_vec import SB3GPUVecEnv

algo, seed, steps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
cfg = {"random_start": True, "observation_type": "count-biomass-time", "param_df": PARAMS, "reset_recruits": True}
venv = SB3GPUVecEnv(cfg, num_envs=12, device="cpu", seed=seed)
model = PPO("MultiInputPolicy", venv, device="cpu", seed=seed, verbose=0)
curve, t0 = [], time.time()


class Eval(BaseCallback):
    def __init__(self, every):
        super().__init__()
        self.every, self.next = every, 0

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.next += self.every
            self.model.policy.to("cuda")
            pol = lambda o: self.model.policy._predict(o, deterministic=True).clamp(-1, 1)
            m, se, _ = evaluate(pol, n=10_000)
            self.model.policy.to("cpu")
            curve.append(dict(tag=f"sb3-ppo-s{seed}", seed=seed, steps=self.num_timesteps, eval_mean=m, eval_se=se, wall=time.time() - t0))
            print(curve[-1], flush=True)
            pd.DataFrame(curve).to_csv(f"{RESULTS}/curves/sb3-ppo-s{seed}.csv", index=False)
        return True


model.learn(int(steps), callback=Eval(1_000_000))
model.save(f"{RESULTS}/agents/sb3-ppo-s{seed}")
