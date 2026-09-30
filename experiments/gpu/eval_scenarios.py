"""
Evaluate trained agents (results/agents/<tag>.pt) and per-scenario baselines on
the fixed test scenarios, the NOMINAL distribution and the WIDE distribution.
usage: python eval_scenarios.py out.csv tag1 tag2 ...
"""
import json
import os
import sys

import pandas as pd
import torch

from common import NOMINAL, RESULTS, WIDE, evaluate, point, test_scenarios
from rl4greencrab.agents.gpu_ppo import GPUPPO
from rl4greencrab.agents.gpu_td3 import GPUTD3, mlp
from rl4greencrab.envs.gpu_wrappers import HistoryObs

N = 8192


def load(tag):
    args = json.load(open(os.path.join(RESULTS, "agents", tag + ".json")))
    path = os.path.join(RESULTS, "agents", tag + ".pt")
    ckpt = torch.load(path, map_location="cuda", weights_only=False)
    if "actor" in ckpt:  # TD3
        actor = mlp(ckpt["obs_dim"], list(ckpt["hp"]["net_arch"]), 2, torch.nn.Tanh()).cuda()
        actor.load_state_dict(ckpt["actor"])
        net = lambda x: actor(x)
    else:
        agent = GPUPPO.load(path, device="cuda")
        net = lambda x: agent.predict(x, flat=True)[0]
    return args, net


def policy_for(args, net):
    hist = args.get("hist", 0)
    wrap = (lambda e: HistoryObs(e, hist)) if hist else None
    extra = dict(observe_scenario=True, scenario_obs_ranges=WIDE) if args.get("oracle") else {}
    holder = {}

    def make_wrap(e):
        holder["env"] = wrap(e) if wrap else e
        return holder["env"]

    @torch.no_grad()
    def pol(o):
        x = o if torch.is_tensor(o) else holder["env"].flatten_obs(o)
        return net(x)
    return pol, make_wrap, extra


def evaluate_everywhere(name, pol, wrap, extra, rows):
    for label, ranges, n in [("nominal", NOMINAL, 20000), ("wide", WIDE, 20000)]:
        m, se, _ = evaluate(pol, n=n, wrap=wrap, scenario_ranges=ranges, **extra)
        rows.append(dict(policy=name, scenario=label, mean=m, se=se))
    for i, pt in enumerate(test_scenarios()):
        m, se, _ = evaluate(pol, n=N, wrap=wrap, scenario_ranges=point(pt), **extra)
        rows.append(dict(policy=name, scenario=f"s{i:02d}", mean=m, se=se, **pt))
    print(name, [round(r["mean"], 3) for r in rows[-27:]], flush=True)


if __name__ == "__main__":
    out, tags = sys.argv[1], sys.argv[2:]
    rows = []
    for tag in tags:
        args, net = load(tag)
        pol, wrap, extra = policy_for(args, net)
        evaluate_everywhere(tag, pol, wrap, extra, rows)
        pd.DataFrame(rows).to_csv(out, index=False)
