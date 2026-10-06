"""Behavior of generalist vs oracle vs specialist in high-migration scenarios (results/behavior_hard.csv)."""
import os

import pandas as pd
import torch

from common import RESULTS, make_env, point, test_scenarios
from eval_scenarios import load, policy_for

SCEN = [1, 8, 16, 20, 21]
N = 2048
rows = []
for i in SCEN:
    pt = test_scenarios()[i]
    for name, tag in [("generalist", "gru-priv-s1-best"), ("oracle", "rppo32-oracle-s0-best"), ("specialist", f"rspec-s{i:02d}-best")]:
        args, net = load(tag)
        pol, wrap, extra = policy_for(args, net)
        env = make_env(num_envs=N, seed=2026, scenario_ranges=point(pt), **extra)
        env = wrap(env) if wrap else env
        obs, _ = env.reset(seed=2026)
        pol.reset(N)
        for t in range(101):
            a = pol(obs)
            traps = 3000 * (1 + a.clamp(-1, 1)) / 2
            pop_before = env.pop.sum(-1)
            obs, r, *_ = env.step(a)
            rows.append(dict(scenario=f"s{i:02d}", policy=name, t=t, year=t // 7, month=4 + t % 7,
                             minnow=traps[:, 0].mean().item(), fukui=traps[:, 1].mean().item(),
                             pop=pop_before.mean().item(), cpue=obs["crabs"][:, 0].mean().item(), reward=r.mean().item()))
    print(f"s{i:02d} done", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, "behavior_hard.csv"), index=False)
