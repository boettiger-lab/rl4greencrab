"""
Stable-baselines3 VecEnv backed by the batched GPU simulator.

This lets the existing SB3 algorithms (PPO, TD3, TQC, RecurrentPPO) train on
`TwoActGPU` with thousands of envs. SB3 itself works on numpy, so observations
and actions cross the host/device boundary once per step (one batched copy,
not one per env). For the fastest training use `GPUPPO`, which never leaves
the GPU.
"""

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from rl4greencrab.envs.gpu_env import TwoActGPU


class SB3GPUVecEnv(VecEnv):
    def __init__(self, config=None, num_envs=1024, device=None, seed=None, normalized=True):
        self.gpu = TwoActGPU(config, num_envs=num_envs, device=device, seed=seed, normalized=normalized)
        obs_space, act_space = self.gpu.spaces()
        super().__init__(num_envs, obs_space, act_space)
        self._actions = None

    def reset(self):
        obs, _ = self.gpu.reset(seed=self._seeds[0] if getattr(self, "_seeds", None) else None)
        self._reset_seeds()
        return self._to_numpy(obs)

    def step_async(self, actions):
        self._actions = torch.as_tensor(actions, dtype=self.gpu.dtype, device=self.gpu.device)

    def step_wait(self):
        obs, reward, done, _, info = self.gpu.step(self._actions)
        obs_np = self._to_numpy(obs)
        done_np = done.cpu().numpy()
        infos = [{} for _ in range(self.num_envs)]
        if "final_obs" in info:
            final = self._to_numpy(info["final_obs"])
            for i in np.flatnonzero(done_np):
                # the CPU env sets terminated=True at the horizon, so SB3 does not bootstrap
                infos[i]["terminal_observation"] = {k: v[i] for k, v in final.items()}
                infos[i]["TimeLimit.truncated"] = False
        return obs_np, reward.cpu().numpy(), done_np, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.gpu, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.gpu, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("SB3GPUVecEnv runs all envs as one batched simulator")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    @staticmethod
    def _to_numpy(obs):
        return {k: v.cpu().numpy() for k, v in obs.items()}
