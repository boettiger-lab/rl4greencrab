"""Speedups from CUDA graphs (simulator step) and TF32 (networks). Run on an otherwise idle GPU."""
import os
import time

import pandas as pd
import torch

from common import RESULTS, WIDE, make_env
from rl4greencrab.agents.gpu_ppo import GPUPPO
from rl4greencrab.agents.gpu_rppo import GPURecurrentPPO

rows = []


def rec(what, setup, rate):
    rows.append(dict(what=what, setup=setup, rate=rate))
    print(f"{what:12s} {setup:40s} {rate:14,.0f} steps/s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "benchmark_speedups.csv"), index=False)


def timed(fn):
    torch.cuda.synchronize()
    t = time.time()
    fn()
    torch.cuda.synchronize()
    return time.time() - t


def env_rate(B, graph, n=505):
    env = make_env(num_envs=B, seed=0, cuda_graph=graph, scenario_ranges=WIDE)
    env.reset()
    a = torch.zeros(B, 2, device=env.device)
    for _ in range(110):  # warm-up incl. graph capture and one auto-reset
        env.step(a)
    return B * n / timed(lambda: [env.step(a) for _ in range(n)])


def train_rate(kind, graph, tf32, updates):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    env = make_env(num_envs=4096, seed=0, cuda_graph=graph, scenario_ranges=WIDE)
    if kind == "ff":
        m = GPUPPO(env, seed=0, anneal_lr=True, tf32=tf32)
        per = 128 * 4096
    else:
        kw = dict(actor_type="transformer", critic_type="privileged") if kind == "transformer" else {}
        m = GPURecurrentPPO(env, seed=0, anneal_lr=True, tf32=tf32, n_seq_minibatches=32,
                            aux_coef=0.5 if kind == "transformer" else 0.0, policy_kwargs=kw)
        per = 101 * 4096
    m.learn(per, verbose=0)  # warm-up
    return per * updates / timed(lambda: m.learn(per * updates, verbose=0))


if __name__ == "__main__":
    for B in [1024, 4096, 16384]:
        for graph in (False, True):
            rec("simulator", f"{B} envs, cuda_graph={graph}", env_rate(B, graph))
    for kind, updates in [("ff", 20), ("gru", 20), ("transformer", 5)]:
        for graph, tf32 in [(False, False), (True, False), (False, True), (True, True)]:
            rec(f"ppo-{kind}", f"cuda_graph={graph}, tf32={tf32}", train_rate(kind, graph, tf32, updates))
