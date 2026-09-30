"""Train GPUPPO with a held-out learning curve. Writes results/curves/<tag>.csv and results/agents/<tag>.pt"""
import argparse
import json
import os
import time

import pandas as pd
import torch

from common import NOMINAL, RESULTS, WIDE, evaluate, interp_ranges, make_env, point, test_scenarios
from rl4greencrab.agents.gpu_ppo import GPUPPO
from rl4greencrab.agents.gpu_rppo import GPURecurrentPPO, RecurrentActor
from rl4greencrab.envs.gpu_wrappers import HistoryObs

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--obs", default="count-biomass-time")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=float, default=200e6)
ap.add_argument("--eval-every", type=float, default=5e6)
ap.add_argument("--eval-n", type=int, default=20_000)
ap.add_argument("--hist", type=int, default=0, help="history length (0 = current obs only)")
ap.add_argument("--n-envs", type=int, default=4096)
ap.add_argument("--n-steps", type=int, default=128)
ap.add_argument("--batch-size", type=int, default=16384)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--anneal", action="store_true")
ap.add_argument("--squash", action="store_true", help="tanh-squashed Gaussian actions")
ap.add_argument("--recurrent", action="store_true", help="GRU policy (GPURecurrentPPO); --net sets GRU hidden size")
ap.add_argument("--seq-minibatches", type=int, default=8)
ap.add_argument("--actor", default="gru", choices=["gru", "transformer"])
ap.add_argument("--critic", default="recurrent", choices=["recurrent", "privileged"])
ap.add_argument("--aux-coef", type=float, default=0.0, help="auxiliary loss: predict privileged state from memory")
ap.add_argument("--tf", default="64,2,4", help="transformer d_model,layers,heads")
ap.add_argument("--tf32", action="store_true", help="TF32 tensor cores for the networks")
ap.add_argument("--cuda-graph", action="store_true", help="capture the simulator step in a CUDA graph")
ap.add_argument("--gamma", type=float, default=0.99)
ap.add_argument("--lam", type=float, default=0.95)
ap.add_argument("--ent", type=float, default=0.0)
ap.add_argument("--net", default="64,64")
ap.add_argument("--act", default="Tanh")
ap.add_argument("--env-json", default="{}", help="extra env config overrides (json)")
ap.add_argument("--scenario", default=None, choices=[None, "nominal", "wide"], help="training scenario distribution")
ap.add_argument("--curriculum", type=float, default=0.0,
                help="widen scenarios from NOMINAL to WIDE linearly over this fraction of training")
ap.add_argument("--oracle", action="store_true", help="agent observes the scenario parameters")
ap.add_argument("--test-scenario", type=int, default=None, help="train a specialist on test_scenarios()[i]")
args = ap.parse_args()

overrides = json.loads(args.env_json)
if args.scenario:
    overrides["scenario_ranges"] = WIDE if args.scenario == "wide" else NOMINAL
if args.test_scenario is not None:
    overrides["scenario_ranges"] = point(test_scenarios()[args.test_scenario])
if args.oracle:
    overrides.update(observe_scenario=True, scenario_obs_ranges=WIDE)
wrap = (lambda e: HistoryObs(e, args.hist)) if args.hist else None
env = make_env(args.obs, num_envs=args.n_envs, seed=args.seed, cuda_graph=args.cuda_graph, **overrides)
env = wrap(env) if wrap else env
if args.recurrent:
    model = GPURecurrentPPO(env, seed=args.seed, learning_rate=args.lr, n_epochs=args.epochs, gamma=args.gamma,
                            gae_lambda=args.lam, ent_coef=args.ent, anneal_lr=args.anneal,
                            n_seq_minibatches=args.seq_minibatches, aux_coef=args.aux_coef, tf32=args.tf32,
                            policy_kwargs=dict(hidden=int(args.net.split(",")[0]), actor_type=args.actor, critic_type=args.critic,
                                               transformer=dict(zip(("d", "layers", "heads"), map(int, args.tf.split(","))))))
else:
  model = GPUPPO(env, seed=args.seed, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
               n_epochs=args.epochs, gamma=args.gamma, gae_lambda=args.lam, ent_coef=args.ent, anneal_lr=args.anneal, squash=args.squash,
               tf32=args.tf32,
               policy_kwargs=dict(net_arch=[int(h) for h in args.net.split(",")], activation=args.act))


curve, next_eval, t0 = [], [0.0], time.time()
if args.recurrent:
    policy = RecurrentActor(model.policy, flatten=model.env.flatten_obs)
else:
    policy = lambda o: model.predict(o if torch.is_tensor(o) else model.env.flatten_obs(o), flat=True)[0]
best = [-float("inf")]


def cb(m):
    if args.curriculum > 0:
        f = min(1.0, m.num_timesteps / (args.curriculum * args.steps))
        m.env.set_scenario_ranges(interp_ranges(NOMINAL, WIDE, f))
    if m.num_timesteps < next_eval[0]:
        return
    next_eval[0] += args.eval_every
    mean, se, _ = evaluate(policy, args.obs, n=args.eval_n, wrap=wrap, **overrides)
    curve.append(dict(tag=args.tag, seed=args.seed, steps=m.num_timesteps, eval_mean=mean, eval_se=se,
                      std=m.policy.log_std.exp().mean().item(), wall=time.time() - t0))
    print(f"{args.tag} {m.num_timesteps/1e6:7.1f}M eval {mean:.3f} ± {se:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    if mean > best[0]:  # keep the best held-out checkpoint (early stopping)
        best[0] = mean
        os.makedirs(os.path.join(RESULTS, "agents"), exist_ok=True)
        m.save(os.path.join(RESULTS, "agents", args.tag + "-best"))
    os.makedirs(os.path.join(RESULTS, "curves"), exist_ok=True)
    pd.DataFrame(curve).to_csv(os.path.join(RESULTS, "curves", f"{args.tag}.csv"), index=False)


cb(model)
steps_per_update = (env.Tmax + 1 if args.recurrent else args.n_steps) * args.n_envs
log_interval = 1 if args.curriculum > 0 else max(1, int(args.eval_every // steps_per_update))
model.learn(int(args.steps), log_interval=log_interval, verbose=0, callback=cb)
next_eval[0] = 0
cb(model)
os.makedirs(os.path.join(RESULTS, "agents"), exist_ok=True)
model.save(os.path.join(RESULTS, "agents", args.tag))
json.dump(vars(args), open(os.path.join(RESULTS, "agents", args.tag + ".json"), "w"))
json.dump(vars(args), open(os.path.join(RESULTS, "agents", args.tag + "-best.json"), "w"))
