import os

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import gamma

from rl4greencrab import GPUPPO, TwoActGPU, TwoActNormalized, gpu_evaluate, twoActEnv
from rl4greencrab.envs.sb3_vec import SB3GPUVecEnv
from rl4greencrab.utils.simulate import simulator

param_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "posterior", "params.csv"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FixedIndex:
    """Stand-in RNG so the CPU env's reset() picks posterior row `i`."""
    def __init__(self, i):
        self.i = i

    def integers(self, lo, hi):
        return self.i


@pytest.mark.parametrize("row", [0, 17, 2500, 4999])
def test_kernels_match_cpu(row):
    cfg = {"observation_type": "size-time", "param_df": param_df, "init_n_adult": 1000}
    cpu = twoActEnv(cfg)
    cpu.np_random = FixedIndex(row)
    cpu.reset()
    g = TwoActGPU(cfg, num_envs=1, device=DEVICE, seed=0, normalized=False)
    g._reset_envs(torch.ones(1, dtype=torch.bool, device=g.device), draw=torch.tensor([row], device=g.device))
    close = lambda a, b: np.testing.assert_allclose(b[0].double().cpu().numpy() if b.dim() > 1 else b.double().cpu().numpy(),
                                                    np.asarray(a, float), rtol=1e-5, atol=1e-6)
    for m in range(7):
        close(cpu.proj_matrices[m], g.proj[:, m])
    close(cpu.overwinter_kernel, g.overwinter)
    close(cpu.size_sel_norm(), g.sel_norm)
    close(cpu.size_sel_log(), g.sel_log)
    var = cpu.init_sd_recruit ** 2
    shape, rate = cpu.init_mean_recruit ** 2 / var, cpu.init_mean_recruit / var
    close(gamma.cdf(cpu.bndry[1:], a=shape, scale=1 / rate) - gamma.cdf(cpu.bndry[:-1], a=shape, scale=1 / rate), g.recruit_dist)
    np.testing.assert_allclose(g.pop[0].double().cpu().numpy(), cpu.state, rtol=1e-5, atol=1e-3)
    action = np.array([700.0, 1200.0])
    cpu.state = g.pop[0].double().cpu().numpy()
    gpu_r = g._reward(torch.tensor(action[None], dtype=g.dtype, device=g.device)).item()
    assert gpu_r == pytest.approx(cpu.reward_func(action), rel=1e-5)


@pytest.mark.parametrize("obs_type", ["count-biomass-time", "size-time", "count"])
def test_matches_cpu_statistically(obs_type):
    # reset_recruits=False reproduces the CPU env's carry-over of recruits across resets
    cfg = {"random_start": True, "observation_type": obs_type, "param_df": param_df, "reset_recruits": False}
    action = np.array([-0.6, 0.1], dtype=np.float32)
    cpu = TwoActNormalized(cfg)
    cpu_returns = []
    for _ in range(150):
        cpu.reset()
        cpu_returns.append(sum(cpu.step(action)[1] for _ in range(101)))
    g = TwoActGPU(cfg, num_envs=4096, device=DEVICE, seed=0)
    const = lambda o: torch.as_tensor(action, device=g.device).expand(g.num_envs, 2)
    gpu_evaluate(const, g)  # warm-up episode: the CPU env carries recruits across resets
    gpu_returns = gpu_evaluate(const, g)
    a, b = np.array(cpu_returns), gpu_returns
    z = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
    assert abs(z) < 4, (a.mean(), b.mean(), z)


def test_step_api_and_spaces():
    cfg = {"random_start": True, "observation_type": "count-biomass-time", "param_df": param_df}
    g = TwoActGPU(cfg, num_envs=8, device=DEVICE, seed=0)
    obs_space, _ = g.spaces()
    obs, _ = g.reset()
    assert obs["crabs"].shape == (8, 2) and obs["months"].tolist() == [4] * 8
    for t in range(101):
        obs, r, term, trunc, info = g.step(torch.zeros(8, 2, device=g.device))
        o = {k: v[0].cpu().numpy() for k, v in obs.items()}
        o["months"] = int(o["months"])
        assert obs_space.contains(o)
        assert term.any().item() == (t == 100)
    assert "final_obs" in info and g.flatten_obs(obs).shape == (8, 2 + 12)


def test_sb3_vec_env():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecEnv
    cfg = {"random_start": True, "observation_type": "count-time", "param_df": param_df}
    venv = SB3GPUVecEnv(cfg, num_envs=16, device=DEVICE, seed=0)
    assert isinstance(venv, VecEnv)
    PPO("MultiInputPolicy", venv, n_steps=32, batch_size=128, device="cpu").learn(1024)


def test_gpu_ppo_train_save_load(tmp_path):
    cfg = {"random_start": True, "observation_type": "count-biomass-time", "param_df": param_df}
    g = TwoActGPU(cfg, num_envs=64, device=DEVICE, seed=0)
    model = GPUPPO(g, seed=0, n_steps=16, batch_size=256, n_epochs=2).learn(4 * 16 * 64, verbose=0)
    agent = GPUPPO.load(model.save(str(tmp_path / "agent")))
    # the loaded agent drives the CPU gymnasium env through the existing tools
    returns = simulator(TwoActNormalized(cfg), agent).simulate(reps=2)
    assert len(returns) == 2 and np.all(np.isfinite(returns))


@pytest.mark.parametrize("normalized", [True, False])
def test_month_sequence_matches_cpu(normalized):
    cfg = {"random_start": True, "observation_type": "count-biomass-time", "param_df": param_df}
    cpu = TwoActNormalized(cfg) if normalized else twoActEnv(cfg)
    g = TwoActGPU(cfg, num_envs=2, device=DEVICE, seed=0, normalized=normalized)
    o_cpu, _ = cpu.reset()
    o_gpu, _ = g.reset()
    cpu_months, gpu_months = [o_cpu["months"]], [o_gpu["months"][0].item()]
    action = np.zeros(2, dtype=np.float32)
    for _ in range(30):
        cpu_months.append(cpu.step(action)[0]["months"])
        gpu_months.append(g.step(torch.zeros(2, 2, device=g.device))[0]["months"][0].item())
    assert cpu_months == gpu_months


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_cuda_graph_matches_eager():
    cfg = {"random_start": True, "observation_type": "size-time", "param_df": param_df}
    out = []
    for graph in (False, True):
        g = TwoActGPU(cfg, num_envs=256, device="cuda", seed=3, cuda_graph=graph)
        obs, _ = g.reset()
        total = torch.zeros(256, device="cuda")
        for _ in range(2 * 101):  # two episodes: covers capture and a post-capture auto-reset
            a = torch.stack([obs["crabs"].mean(-1), 0.1 * (obs["months"] - 7)], -1)
            obs, r, done, _, _ = g.step(a)
            total += r
        out.append(total.cpu())
    assert torch.equal(out[0], out[1])


def test_precision_fallbacks_on_cpu():
    from rl4greencrab.utils.precision import configure_tf32, device_capabilities
    assert device_capabilities("cpu")["tf32"] is False
    assert configure_tf32("auto", "cpu") is False
    with pytest.warns(UserWarning, match="TF32 requested"):
        assert configure_tf32(True, "cpu") is False
    cfg = {"random_start": True, "observation_type": "count", "param_df": param_df}
    with pytest.warns(UserWarning, match="CUDA graphs requested"):
        g = TwoActGPU(cfg, num_envs=4, device="cpu", seed=0, cuda_graph=True)
    assert g.cuda_graph is False
    g.reset()
    g.step(torch.zeros(4, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cuda_graph_capture_failure_falls_back(monkeypatch):
    cfg = {"random_start": True, "observation_type": "count-time", "param_df": param_df}

    def broken_graph(*args, **kwargs):
        raise RuntimeError("simulated capture failure")

    out = []
    for graph in (False, True):
        g = TwoActGPU(cfg, num_envs=64, device="cuda", seed=5, cuda_graph=graph)
        if graph:
            monkeypatch.setattr(torch.cuda, "graph", broken_graph)
        obs, _ = g.reset()
        total = torch.zeros(64, device="cuda")
        with pytest.warns(UserWarning, match="capture failed") if graph else _nullcontext():
            for _ in range(101):
                obs, r, *_ = g.step(torch.zeros(64, 2, device="cuda"))
                total += r
        if graph:
            assert g.cuda_graph is False
        out.append(total.cpu())
    assert torch.equal(out[0], out[1])


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
