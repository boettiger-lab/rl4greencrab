"""Train GPUTD3 with a held-out learning curve (results/curves/<tag>.csv)."""
import argparse
import json
import os
import time

import pandas as pd
import torch

from common import RESULTS, evaluate, make_env
from rl4greencrab.agents.gpu_td3 import GPUTD3
from rl4greencrab.envs.gpu_wrappers import HistoryObs

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--obs", default="count-biomass-time")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=float, default=100e6)
ap.add_argument("--eval-every", type=float, default=5e6)
ap.add_argument("--eval-n", type=int, default=10_000)
ap.add_argument("--hist", type=int, default=0)
ap.add_argument("--n-envs", type=int, default=1024)
ap.add_argument("--gamma", type=float, default=0.99)
ap.add_argument("--n-step", type=int, default=3)
ap.add_argument("--batch-size", type=int, default=8192)
ap.add_argument("--updates", type=int, default=2)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--env-json", default="{}")
args = ap.parse_args()

overrides = json.loads(args.env_json)
wrap = (lambda e: HistoryObs(e, args.hist)) if args.hist else None
env = make_env(args.obs, num_envs=args.n_envs, seed=args.seed, **overrides)
env = wrap(env) if wrap else env
model = GPUTD3(env, seed=args.seed, gamma=args.gamma, n_step=args.n_step, batch_size=args.batch_size,
               updates_per_step=args.updates, learning_rate=args.lr)
curve, next_eval, t0 = [], [0.0], time.time()
policy = lambda o: model.act(o if torch.is_tensor(o) else model.env.flatten_obs(o))


def cb(m):
    if m.num_timesteps < next_eval[0]:
        return
    next_eval[0] += args.eval_every
    mean, se, _ = evaluate(policy, args.obs, n=args.eval_n, wrap=wrap, **overrides)
    curve.append(dict(tag=args.tag, seed=args.seed, steps=m.num_timesteps, eval_mean=mean, eval_se=se, wall=time.time() - t0))
    print(f"{args.tag} {m.num_timesteps/1e6:7.1f}M eval {mean:.3f} ± {se:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    os.makedirs(os.path.join(RESULTS, "curves"), exist_ok=True)
    pd.DataFrame(curve).to_csv(os.path.join(RESULTS, "curves", f"{args.tag}.csv"), index=False)


model.learn(int(args.steps), log_interval=max(1, int(args.eval_every // args.n_envs)), callback=cb, verbose=0)
next_eval[0] = 0
cb(model)
os.makedirs(os.path.join(RESULTS, "agents"), exist_ok=True)
model.save(os.path.join(RESULTS, "agents", args.tag))
json.dump(vars(args), open(os.path.join(RESULTS, "agents", args.tag + ".json"), "w"))
