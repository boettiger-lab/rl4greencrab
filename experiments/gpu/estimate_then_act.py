"""
"Estimate, then act": a deployable policy built from (1) a supervised GRU that estimates the
invasion-scenario parameters from the catch history, and (2) the oracle policy (trained knowing
the true scenario), which is fed those estimates instead of the truth.

The estimator is causal: before choosing the action at month t it has seen observations up to t
and its own actions up to t-1. Training data: episodes on WIDE driven by the estimate-then-act
policy itself (DAgger-style: round 0 uses the deployable generalist, later rounds the current
estimate-then-act policy), so the estimator is trained on the states it will actually see.
Writes results/agents/estimator.pt and results/scenario_eval_estimate.csv
"""
import os
import sys

import pandas as pd
import torch
import torch.nn as nn

from common import NOMINAL, RESULTS, WIDE, evaluate, make_env, point, test_scenarios
from eval_scenarios import load, policy_for

torch.manual_seed(0)
ORACLE = "rppo32-oracle-s0-best"
N_SCEN = len(WIDE)


class Estimator(nn.Module):
    def __init__(self, in_dim, out_dim=N_SCEN, hidden=128):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x, h=None):
        y, h = self.gru(x, h)
        return self.head(y).clamp(-1, 1), h


class EstimateThenAct:
    """Deployable policy: estimator GRU -> scenario estimate -> oracle policy (which never sees the truth)."""

    def __init__(self, est, oracle_actor, env_holder):
        self.est, self.oracle, self.holder = est, oracle_actor, env_holder

    def reset(self, B):
        self.oracle.reset(B)
        self.h, self.prev_a = None, torch.zeros(B, 2, device="cuda")

    @torch.no_grad()
    def __call__(self, obs):
        # estimator inputs: catches, month, own previous action (never the true scenario)
        x = torch.cat([obs["crabs"], nn.functional.one_hot(obs["months"], 12).float(), self.prev_a], -1)
        est, self.h = self.est(x.unsqueeze(0), self.h)
        a = self.oracle({**obs, "scenario": est[0]})
        self.prev_a = a
        return a


def make_policy(est):
    args, net = load(ORACLE)
    oracle, wrap, extra = policy_for(args, net)
    holder = {}

    def wrap2(e):
        holder["env"] = wrap(e)
        return holder["env"]
    return EstimateThenAct(est, oracle, holder), wrap2, extra


@torch.no_grad()
def collect(policy, wrap, extra, n, seed, use_truth_actions=None):
    """Roll out `policy` on WIDE; return estimator inputs (T, n, D) and true scaled scenario (n, P)."""
    env = wrap(make_env(num_envs=n, seed=seed, scenario_ranges=WIDE, **extra))
    obs, _ = env.reset(seed=seed)
    policy.reset(n)
    y = obs["scenario"].clone()
    prev = torch.zeros(n, 2, device="cuda")
    xs = []
    for _ in range(101):
        xs.append(torch.cat([obs["crabs"], nn.functional.one_hot(obs["months"], 12).float(), prev], -1))
        a = policy(obs)
        prev = a.clamp(-1, 1)
        obs, *_ = env.step(a)
    return torch.stack(xs), y


def fit(est, X, Y, iters=3000):
    opt = torch.optim.Adam(est.parameters(), lr=1e-3)
    for _ in range(iters):
        j = torch.randint(0, X.shape[1], (512,), device="cuda")
        pred, _ = est(X[:, j])
        loss = (pred - Y[j].unsqueeze(0)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return est


class GeneralistDriver:
    """Round-0 data: actions from the deployable generalist (it ignores the scenario obs)."""

    def __init__(self):
        args, net = load("gru-priv-s1-best")
        self.pol, _, _ = policy_for(args, net)
        self.holder = None

    def reset(self, B):
        self.pol.reset(B)

    def __call__(self, obs):
        o = {k: v for k, v in obs.items() if k != "scenario"}
        return self.pol(torch.cat([o["crabs"], nn.functional.one_hot(o["months"], 12).float()], -1))


if __name__ == "__main__":
    est = Estimator(2 + 12 + 2).cuda()
    policy, wrap, extra = make_policy(est)
    extra = {**extra, "scenario_obs_ranges": WIDE}
    Xs, Ys = [], []
    for rnd in range(3):
        driver = GeneralistDriver() if rnd == 0 else policy
        X, Y = collect(driver, wrap, extra, 16384, 100 + rnd)
        Xs.append(X), Ys.append(Y)
        est = fit(est, torch.cat(Xs, 1), torch.cat(Ys, 0))
        m, se, _ = evaluate(policy, n=8192, wrap=wrap, scenario_ranges=WIDE, **extra)
        print(f"round {rnd}: estimate-then-act on WIDE {m:.3f} ± {se:.3f}", flush=True)
    os.makedirs(os.path.join(RESULTS, "agents"), exist_ok=True)
    torch.save(est.state_dict(), os.path.join(RESULTS, "agents", "estimator.pt"))

    rows = []
    for label, ranges, n in [("nominal", NOMINAL, 20000), ("wide", WIDE, 20000)]:
        m, se, _ = evaluate(policy, n=n, wrap=wrap, scenario_ranges=ranges, **extra)
        rows.append(dict(policy="estimate-then-act", scenario=label, mean=m, se=se))
    for i, pt in enumerate(test_scenarios()):
        m, se, _ = evaluate(policy, n=8192, wrap=wrap, scenario_ranges=point(pt), **extra)
        rows.append(dict(policy="estimate-then-act", scenario=f"s{i:02d}", mean=m, se=se, **pt))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "scenario_eval_estimate.csv"), index=False)
    print(pd.DataFrame(rows)[["scenario", "mean"]].head(3).to_string())
