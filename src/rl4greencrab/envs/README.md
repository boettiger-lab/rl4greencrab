# twoActEnv (Gymnasium) — Green Crab IPM Environment

This dir contains a custom **Gymnasium** environment (`twoActEnv`) for simulating and controlling a green crab population with **two trapping actions**. For RL training, you can use normalized environment `TwoActNormalized`

`twoActEnv` env_id = "twoactenv"

`TwoActNormalized` env_id = "twoactenvnorm"

---

## Features

- **Gymnasium-compatible** `Env` with `reset()` and `step()`
- **Continuous 2D action space**: traps/effort per month for two actions
- **Size-structured population dynamics** (21 size bins by default)
- **Seasonal loop**: months advance from April (`curr_month=4`) through October, then recruitment + overwinter mortality
- **Multiple observation modes** via `observation_type`:
  - `count-biomass-time`: number of crabs caught per trap (CPUE, continuous), mean biomass of the crabs caught (continuous), current month (discrete)
  - `count-time`: number of crabs caught per trap (CPUE, continuous), current month (discrete)
  - `count-biomass`: number of crabs caught per trap (CPUE, continuous), mean biomass of the crabs caught (continuous)
  - `biomass-time`: mean biomass of the crabs caught (continuous), current month (discrete)
  - `size-time`: number of crabs caught in size class $x$ per trap (size-structured CPUE, continuous), current month (discrete)
- Optional **reproducibility controls** with separate RNG streams:
  - main environment RNG
  - migration-only RNG (`seed_migration`)
- Optional **curriculum learning** behavior that changes initial adult population range over training progress
- Action smoothness penalty: discourages large within-year variance in actions (applied at month 11)

---

## Use Case: Training an RL Policy for Green Crab Control

This environment can be used to train a reinforcement learning agent that learns
monthly trapping effort for controlling invasive green crab populations.

Below is an example using **PPO** from Stable-Baselines3 with vectorized
environments and a normalized wrapper.

### Example: PPO Training with `TwoActNormalized`

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from rl4greencrab import TwoActNormalized

# Environment configuration
config = {
    'random_start':True,
    'param_csv': '/home/jovyan/rl4greencrab/data/posterior/params.csv'
    'observation_type': 'size-time'
}

# Optional: single environment (useful for debugging)
env = TwoActNormalized(config)

# Vectorized environments for efficient PPO training
vec_env = make_vec_env(TwoActNormalized, n_envs=12, env_kwargs={"config": config},)

# PPO with MultiInputPolicy for Dict observations
model = PPO(
    "MultiInputPolicy",
    vec_env,
    verbose=0,
    tensorboard_log="/home/jovyan/logs",
)

# Train the agent
model.learn(
    total_timesteps=1_000,
    progress_bar=True,
)
```
---

## GPU version: `TwoActGPU`

`gpu_env.py` contains `TwoActGPU`, a batched PyTorch port of `twoActEnv` / `TwoActNormalized` that runs thousands of envs at once on the GPU. It takes the same `config` dict. Kernels, selectivity curves, initial state and reward match the CPU env to float32 precision, and episode returns agree statistically (`tests/test_gpu_env.py`).

```python
from rl4greencrab import TwoActGPU, GPUPPO, gpu_evaluate

config = {'random_start': True, 'observation_type': 'count-time',
          'param_csv': 'data/posterior/params.csv'}
env = TwoActGPU(config, num_envs=4096, seed=0)        # normalized=False for natural units
obs, _ = env.reset()                                   # {"crabs": (B, k), "months": (B,)}
obs, reward, terminated, truncated, info = env.step(actions)   # actions: (B, 2) tensor, auto-reset

model = GPUPPO(env).learn(10_000_000)                  # PPO, entirely on the GPU
returns = gpu_evaluate(lambda o: model.predict(o)[0], env)
model.save('agent')                                    # GPUPPO.load('agent.pt').predict(obs) works with the CPU env too
```

From the command line, with the existing hyperparameter files: `python scripts/train_gpu.py -f hyperpars/count-time/ppo.yaml [--n-envs 4096]`.

To use SB3 algorithms (TD3, TQC, RecurrentPPO) with the GPU simulator, add `gpu_env: True` (and optionally `gpu_n_envs`) to a yaml file for `scripts/train.py`, or use `rl4greencrab.envs.sb3_vec.SB3GPUVecEnv` directly. Training is then limited by SB3's numpy-based loop rather than by the simulator.

Differences from the CPU env:
- One `torch.Generator` (`seed=`) drives all randomness, so trajectories are not draw-for-draw identical to the numpy env, and there is no separate migration RNG.
- `reset_recruits` (default `True`): every episode starts with zero recruits. The CPU env never clears its recruit vector on `reset()`, so there an episode's first-year recruits leak in from the previous episode's final winter (a bug). `False` reproduces the CPU behavior.
- Normalized actions are clipped to [-1, 1] inside `step()`.

### Performance modes and hardware fallbacks

- `TwoActGPU(..., cuda_graph=True | False | "auto")` captures the per-step dynamics in a CUDA graph (results are bit-identical to eager stepping). On CPU, or if capture fails, it warns and falls back to eager steps.
- The PPO/TD3 trainers take `tf32=True | False | "auto"` for the neural networks. TF32 needs compute capability >= 8.0 (Ampere or newer); on older GPUs (e.g. Quadro RTX 8000, sm_75) or CPU a `True` request warns and uses full fp32, and `"auto"` silently does the same. The simulator's own matrix products always run in full fp32.
- `rl4greencrab.utils.precision.device_capabilities()` reports what the current device supports.
