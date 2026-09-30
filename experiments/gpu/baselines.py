"""
Baselines on the fixed (recruits reset) environment:
  1. best constant action (grid search, then local refinement)
  2. best open-loop seasonal schedule: one action per month (14 numbers), found by CEM
  3. the 120 SB3 agents from the manuscript (trained on the buggy env), re-evaluated
Each is also evaluated on the buggy env (reset_recruits=False) for comparison with the manuscript.
"""

import glob
import os
import time

import numpy as np
import pandas as pd
import torch

from common import HF_AGENTS, RESULTS, SB3Policy, constant_grid, episode_returns, evaluate, make_env

rows = []


def record(kind, name, obs_type, fn):
    for label, rr in [("fixed", True), ("buggy", False)]:
        m, se, _ = evaluate(fn, obs_type, n=20_000, reset_recruits=rr)
        rows.append(dict(kind=kind, name=name, obs_type=obs_type, env=label, mean=m, se=se))
    print(rows[-2], rows[-1], flush=True)


def best_constant(reset_recruits):
    g = torch.linspace(-1, 1, 41)
    grid = torch.cartesian_prod(g, g)
    vals = constant_grid(grid, n_eps=2048, reset_recruits=reset_recruits)
    best = grid[vals.argmax()]
    fine = torch.cartesian_prod(torch.linspace(-0.05, 0.05, 11), torch.linspace(-0.05, 0.05, 11)) + best
    fine = fine.clamp(-1, 1)
    vals = constant_grid(fine, n_eps=8192, seed=7, reset_recruits=reset_recruits)
    return fine[vals.argmax()]


def seasonal_cem(reset_recruits, gens=40, pop=128, n_eps=512, elite=0.1):
    """Cross-entropy method over a 7-month x 2-trap open-loop schedule in [-1, 1]."""
    mu, sd = torch.zeros(7, 2), torch.full((7, 2), 0.5)
    for g in range(gens):
        cand = (mu + sd * torch.randn(pop, 7, 2)).clamp(-1, 1)
        env = make_env(num_envs=pop * n_eps, seed=100 + g, reset_recruits=reset_recruits)
        C = cand.to(env.device).repeat_interleave(n_eps, 0)
        ar = torch.arange(env.num_envs, device=env.device)
        # obs months = month just trapped; the next action applies to env.curr_month
        pol = lambda o: C[ar, env.curr_month - 4]
        if not reset_recruits:
            episode_returns(pol, env, seed=200 + g)
        r = episode_returns(pol, env, seed=100 + g)
        score = r.view(pop, n_eps).mean(1).cpu()
        top = cand[score.argsort(descending=True)[: int(elite * pop)]]
        mu, sd = top.mean(0), top.std(0) + 0.02 * (1 - g / gens)
    return mu


class Seasonal:
    """Open-loop schedule; normalized-env obs months are the month about to be trapped."""
    def __init__(self, sched):
        self.sched = sched.cuda()

    def __call__(self, obs):
        return self.sched[obs["months"] - 4]


if __name__ == "__main__":
    t0 = time.time()
    for rr in [True, False]:
        tag = "fixed" if rr else "buggy"
        a = best_constant(rr)
        print(f"best constant ({tag}-trained):", a.tolist(), f"{time.time()-t0:.0f}s", flush=True)
        record("constant", f"const-opt-{tag}[{a[0]:.3f},{a[1]:.3f}]", "count-biomass-time",
               lambda o, a=a: a.cuda().expand(len(o["months"]), 2))
        s = seasonal_cem(rr)
        print(f"seasonal ({tag}-trained):", s.tolist(), f"{time.time()-t0:.0f}s", flush=True)
        torch.save(s, os.path.join(RESULTS, f"seasonal_{tag}.pt"))
        record("seasonal", f"seasonal-opt-{tag}", "count-biomass-time", Seasonal(s))
    record("constant", "no-action", "count-biomass-time", lambda o: -torch.ones(len(o["months"]), 2, device="cuda"))

    for path in sorted(glob.glob(os.path.join(HF_AGENTS, "*", "*.zip"))):
        obs_type = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path)[:-4]
        record("sb3", name, obs_type, SB3Policy(path))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "baselines.csv"), index=False)
    print(f"done in {time.time()-t0:.0f}s")
