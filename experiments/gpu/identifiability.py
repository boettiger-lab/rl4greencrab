"""
How well can each invasion-scenario parameter be inferred from the catch history?
Simulate episodes on WIDE under the deployable generalist, then train a supervised GRU to
predict each (scaled) scenario parameter from the observations so far; report held-out R^2
after 1, 3, 7 and 14 years. Writes results/identifiability.csv
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import RESULTS, WIDE, make_env
from eval_scenarios import load, policy_for

torch.manual_seed(0)
NAMES = list(WIDE)


@torch.no_grad()
def simulate(n, seed):
    args, net = load("gru-priv-s1-best")
    pol, wrap, extra = policy_for(args, net)
    env = wrap(make_env(num_envs=n, seed=seed, scenario_ranges=WIDE, scenario_obs_ranges=WIDE))
    obs, _ = env.reset(seed=seed)
    pol.reset(n)
    y = env._scenario_obs().clone()  # scenario parameters scaled to [-1, 1] over WIDE (log scale where WIDE is log)
    xs = []
    for _ in range(101):
        x = env.flatten_obs(obs)
        a = pol(obs)
        xs.append(torch.cat([x, a], -1))  # what the manager knows: catches, month, and own past effort
        obs, *_ = env.step(a)
    return torch.stack(xs), y  # (T, n, D), (n, P)


X, Y = simulate(32768, 11)
Xt, Yt = simulate(8192, 12)
gru = nn.GRU(X.shape[-1], 128).cuda()
head = nn.Linear(128, Y.shape[-1]).cuda()
opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=1e-3)
for it in range(3000):
    j = torch.randint(0, X.shape[1], (512,), device="cuda")
    h, _ = gru(X[:, j])
    loss = (head(h) - Y[j].unsqueeze(0)).pow(2).mean()  # predict at every time step
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    pred = head(gru(Xt)[0])  # (T, n, P)
rows = []
for years in (1, 3, 7, 14):
    t = min(7 * years, 101) - 1
    err = ((pred[t] - Yt) ** 2).mean(0)
    r2 = 1 - err / Yt.var(0)
    rows += [dict(years=years, parameter=n, r2=float(r2[k])) for k, n in enumerate(NAMES)]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, "identifiability.csv"), index=False)
print(df.pivot(index="parameter", columns="years", values="r2").round(2).to_string())
