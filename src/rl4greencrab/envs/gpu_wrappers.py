"""Wrappers for TwoActGPU that change what the agent observes."""

import torch


class HistoryObs:
    """
    Observation = the last `k` flattened observations and the last `k` actions
    (zeros before the episode start), newest first. Gives a memoryless policy
    access to recent catch history, since a single month's CPUE says little about
    the hidden population state.

    Exposes the TwoActGPU interface used by GPUPPO (reset/step/flatten_obs/
    flat_obs_dim/num_envs/device/config/Tmax); observations are flat tensors.
    """

    def __init__(self, env, k=4):
        self.env, self.k = env, k
        self.num_envs, self.device, self.config, self.Tmax = env.num_envs, env.device, env.config, env.Tmax
        self.step_dim = env.flat_obs_dim + 2
        self.flat_obs_dim = k * self.step_dim
        self.hist = torch.zeros(env.num_envs, k, self.step_dim, device=env.device)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _push(self, obs, action):
        x = torch.cat([self.env.flatten_obs(obs), action], -1)
        self.hist = torch.cat([x.unsqueeze(1), self.hist[:, :-1]], 1)

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        self.hist.zero_()
        self._push(obs, torch.zeros(self.num_envs, 2, device=self.device))
        return self.hist.flatten(1), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        a = torch.as_tensor(action, dtype=self.hist.dtype, device=self.device).clamp(-1, 1)
        self._push(obs, a)
        done = term | trunc
        if done.any():
            # finished envs were auto-reset: restart their history from the fresh obs
            fresh = torch.zeros_like(self.hist)
            fresh[:, 0, :-2] = self.env.flatten_obs(obs)
            self.hist = torch.where(done.view(-1, 1, 1), fresh, self.hist)
        return self.hist.flatten(1), r, term, trunc, info

    def flatten_obs(self, obs):
        return obs
