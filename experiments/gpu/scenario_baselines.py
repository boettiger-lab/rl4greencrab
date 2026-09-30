"""Best constant action and best seasonal (per-month) schedule for each test scenario and for the NOMINAL/WIDE distributions."""
import os

import pandas as pd
import torch

from baselines import Seasonal
from common import NOMINAL, RESULTS, WIDE, episode_returns, evaluate, make_env, point, test_scenarios


def seasonal_cem(ranges, gens=30, pop=128, n_eps=256, elite=0.1, seed=0):
    mu, sd = torch.zeros(7, 2), torch.full((7, 2), 0.5)
    g0 = torch.Generator().manual_seed(seed)
    for g in range(gens):
        cand = (mu + sd * torch.randn(pop, 7, 2, generator=g0)).clamp(-1, 1)
        env = make_env(num_envs=pop * n_eps, seed=100 + g, scenario_ranges=ranges)
        C = cand.to(env.device).repeat_interleave(n_eps, 0)
        ar = torch.arange(env.num_envs, device=env.device)
        r = episode_returns(lambda o: C[ar, o["months"] - 4], env, seed=100 + g)
        score = r.view(pop, n_eps).mean(1).cpu()
        top = cand[score.argsort(descending=True)[: int(elite * pop)]]
        mu, sd = top.mean(0), top.std(0) + 0.02 * (1 - g / gens)
    return mu


def constant_cem(ranges, **kw):
    """A constant action is a seasonal schedule with all months tied: optimize the 2 numbers by CEM too."""
    mu, sd = torch.zeros(2), torch.full((2,), 0.5)
    for g in range(20):
        cand = (mu + sd * torch.randn(64, 2)).clamp(-1, 1)
        env = make_env(num_envs=64 * 512, seed=300 + g, scenario_ranges=ranges)
        C = cand.to(env.device).repeat_interleave(512, 0)
        r = episode_returns(lambda o: C, env, seed=300 + g)
        top = cand[r.view(64, 512).mean(1).cpu().argsort(descending=True)[:8]]
        mu, sd = top.mean(0), top.std(0) + 0.01
    return mu


if __name__ == "__main__":
    targets = [("nominal", NOMINAL), ("wide", WIDE)] + [(f"s{i:02d}", point(pt)) for i, pt in enumerate(test_scenarios())]
    pts = {f"s{i:02d}": pt for i, pt in enumerate(test_scenarios())}
    rows, schedules = [], {}
    for name, ranges in targets:
        c = constant_cem(ranges)
        s = seasonal_cem(ranges)
        schedules[name] = dict(constant=c, seasonal=s)
        n = 20000 if name in ("nominal", "wide") else 8192
        for kind, fn in [("constant", lambda o, c=c: c.cuda().expand(len(o["months"]), 2)), ("seasonal", Seasonal(s))]:
            m, se, _ = evaluate(fn, n=n, scenario_ranges=ranges)
            rows.append(dict(policy=f"{kind}-specialist", scenario=name, mean=m, se=se, **pts.get(name, {})))
        # the wide-optimized open-loop policies, applied to this scenario
        if "wide" in schedules:
            for kind in ("constant", "seasonal"):
                w = schedules["wide"][kind]
                fn = (lambda o, c=w: c.cuda().expand(len(o["months"]), 2)) if kind == "constant" else Seasonal(w)
                m, se, _ = evaluate(fn, n=n, scenario_ranges=ranges)
                rows.append(dict(policy=f"{kind}-wide", scenario=name, mean=m, se=se, **pts.get(name, {})))
        print(name, rows[-4:] if name != "nominal" else rows[-2:], flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "scenario_baselines.csv"), index=False)
    torch.save(schedules, os.path.join(RESULTS, "scenario_schedules.pt"))
