# twoActEnv (Gymnasium) — Green Crab IPM Environment

This dir contains a custom **Gymnasium** environment (`twoActEnv`) for simulating and controlling a green crab population with **two trapping actions**. For RL training, you can use normalized environment `TwoActNormalized`

`twoActEnv` env_id = "twoactenv"
`TwoActNormalized` env_id = "twoactenvnorm"

---

## Features

- **Gymnasium-compatible** `Env` with `reset()` and `step()`
- **Continuous 2D action space**: traps/effort per month for two actions
- **Size-structured population dynamics** (21 size bins by default)
- **Seasonal loop**: months advance from March (`curr_month=3`) through November, then recruitment + overwinter effects
- **Multiple observation modes** via `observation_type`:
  - `count`: number of crab caught, continous
  - `biomass`: total biomass the crab caught, continous
  - `size`: size distribution of the crab caught, continous
  - `time`: current month, discrete
  - `count-biomass-time`
  - `count-time`
  - `count-biomass`
  - `biomass-time`
  - `size-time`
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
    "w_mort_scale" : 600,
    "growth_k": 0.70,
    'random_start':True,
    'var_penalty_const': 0,
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