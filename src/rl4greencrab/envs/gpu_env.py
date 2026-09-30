"""
Batched, GPU-resident version of `twoActEnv` / `TwoActNormalized`.

`TwoActGPU` simulates `num_envs` independent green crab IPM environments at
once, entirely as torch tensors on one device. There are no per-env python
loops, no numpy/scipy calls and no host<->device copies inside `step()`, so it
can be driven by an RL algorithm that also lives on the GPU (see
`rl4greencrab.agents.gpu_ppo`) or wrapped as an SB3 VecEnv (see
`rl4greencrab.envs.sb3_vec`).

The dynamics follow `twoActEnv.step()` line by line:
  harvest (binomial, size-selective) -> monthly growth/survival projection ->
  recruits added in May -> reward -> month += 1 -> overwinter (binomial
  survival + density-dependent local recruits + non-local migrants) after
  October.

Differences from the gymnasium envs:
  * vectorized: `reset()` / `step()` take and return batched tensors
  * envs auto-reset when their episode ends; the pre-reset observation is
    returned in `info["final_obs"]`
  * observations are `{"crabs": (B, k) float, "months": (B,) long}` dicts, or
    a flat (B, k + 12) tensor from `flatten_obs()` that matches what SB3's
    MultiInputPolicy feeds its network (crabs ++ one_hot(months, 12))
  * one torch.Generator drives all randomness (`seed` in the constructor);
    the separate migration RNG of the CPU env is not reproduced
"""

import math

import numpy as np
import pandas as pd
import torch

PARAM_COLS = [
    "growth_k", "growth_xinf", "growth_sd", "growth_A", "growth_ds",
    "mort_alpha", "mort_beta",
    "trapm_pmax", "trapm_sigma", "trapm_xmax",
    "trapf_pmax", "trapf_k", "trapf_midpoint",
    "init_mean_recruit", "init_sd_recruit", "init_mean_adult", "init_sd_adult",
]

OBS_TYPES = [
    "count-biomass-time", "count-time", "size-time", "biomass-time",
    "size", "count", "count-biomass",
]

N_MONTHS = 12  # size of the "months" Discrete space (one-hot width)


class TwoActGPU:
    def __init__(
        self,
        config=None,
        num_envs=1024,
        device=None,
        dtype=torch.float32,
        seed=None,
        normalized=True,
        cuda_graph=False,
    ):
        config = config or {}
        self.config = config
        self.num_envs = num_envs
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.normalized = normalized
        # capture the per-step dynamics in a CUDA graph (removes kernel-launch overhead)
        self.cuda_graph = bool(cuda_graph) and self.device.type == "cuda"
        self._graph, self._graph_calls = None, 0

        self.gen = torch.Generator(device=self.device)
        if seed is None:
            self.gen.seed()
        else:
            self.gen.manual_seed(seed)

        # same config keys and defaults as twoActEnv / TwoActNormalized
        self.random_start = config.get("random_start", False)
        self.init_n_adult = config.get("init_n_adult", 0)
        self.w_mort_scale = config.get("w_mort_scale", 600)
        self.K = config.get("K", 25000)
        self.r = config.get("r", 1)
        self.env_stoch = config.get("env_stoch", 0.1)
        self.max_action = config.get("max_action", 3000)
        self.max_obs = config.get("max_obs", 2000)
        self.minsize = config.get("minsize", 0)
        self.maxsize = config.get("maxsize", 110)
        self.nsize = config.get("nsize", 22)
        self.Tmax = config.get("Tmax", 100)
        self.action_reward_exponent = config.get("action_reward_exponent", 1)
        self.area = config.get("area", 30000)
        self.loss_a = config.get("loss_a", 0.265)
        self.loss_b = config.get("loss_b", 2.80)
        self.loss_c = config.get("loss_c", 2.99)
        self.cpue_normalization = config.get("cpue_normalization", 100)
        self.observation_type = config.get("observation_type", "count-biomass-time")
        if self.observation_type not in OBS_TYPES:
            raise ValueError(f"unknown observation_type {self.observation_type!r}")
        # Start every episode with zero recruits. The CPU env never clears its
        # recruit vector in reset() (a bug: first-year recruits leak in from the
        # previous episode); set False to reproduce that behavior.
        self.reset_recruits = config.get("reset_recruits", True)

        # Per-episode invasion-scenario parameters. Each is fixed at its config
        # value unless config["scenario_ranges"][name] gives a range to draw it
        # from at every reset: [lo, hi] (uniform) or [lo, hi, "log"] (log-uniform).
        #   init_n_adult  initial adult abundance (a range overrides random_start)
        #   K             carrying capacity
        #   r             local recruitment rate
        #   mig_scale     multiplier on the mean number of non-local migrants
        #   p_big         probability of a large migration pulse in a year
        self.scenario_defaults = {"K": self.K, "r": self.r, "mig_scale": config.get("mig_scale", 1.0),
                                  "p_big": config.get("p_big", 0.2)}
        self.scenario_ranges = dict(config.get("scenario_ranges", {}))
        # observe_scenario: append the drawn scenario parameters (scaled to [-1, 1]
        # over their ranges) to the observation, as an "oracle" information bound
        self.observe_scenario = config.get("observe_scenario", False)
        self.scenario_obs_ranges = dict(config.get("scenario_obs_ranges", self.scenario_ranges))

        param_df = config.get("param_df")
        if param_df is None:
            if "param_csv" not in config:
                raise ValueError("config needs 'param_df' or 'param_csv' (posterior draws)")
            param_df = pd.read_csv(config["param_csv"])
        # kernels are built in float64 at reset time, then cast to self.dtype
        self.posterior = torch.tensor(
            param_df[PARAM_COLS].to_numpy(dtype=np.float32), dtype=torch.float64, device=self.device
        )
        self.post_mean, self.post_std = self.posterior.mean(0), self.posterior.std(0).clamp(min=1e-12)

        # IPM mesh and fixed size-dependent quantities
        f64 = dict(dtype=torch.float64, device=self.device)
        self.bndry64 = self.minsize + torch.arange(self.nsize + 1, **f64) * (self.maxsize - self.minsize) / self.nsize
        self.midpts64 = 0.5 * (self.bndry64[:-1] + self.bndry64[1:])
        y = self.midpts64
        self.biomass_size = torch.clamp(-0.071 * y + 0.003 * y**2 + 0.00002 * y**3, min=0).to(dtype)
        self.w_mort_exp = torch.exp(-self.w_mort_scale / self.midpts64**2).to(dtype)
        self.D = (torch.tensor([91, 121, 152, 182, 213, 244, 274, 305], **f64) - 91) / 365
        self.action_reward_scale = torch.tensor(
            config.get("action_reward_scale", [0.08, 0.08]), dtype=dtype, device=self.device
        )

        self.obs_dim = self.nsize if "size" in self.observation_type else (2 if self.observation_type.startswith("count-biomass") else 1)
        self.has_time = self.observation_type.endswith("time")
        self.n_scenario_obs = len(self.scenario_obs_ranges) if self.observe_scenario else 0
        self.flat_obs_dim = self.obs_dim + (N_MONTHS if self.has_time else 0) + self.n_scenario_obs

        # per-env state
        B, n = num_envs, self.nsize
        z = dict(dtype=dtype, device=self.device)
        self.pop = torch.zeros(B, n, **z)
        self.recruit_sizes = torch.zeros(B, n, **z)
        self.curr_month = torch.full((B,), 4, dtype=torch.long, device=self.device)
        self.month_passed = torch.zeros(B, dtype=torch.long, device=self.device)
        self.proj = torch.zeros(B, 7, n, n, **z)       # monthly growth * survival, months 4..10
        self.overwinter = torch.zeros(B, n, n, **z)    # growth Oct -> Apr
        self.sel_norm = torch.zeros(B, n, **z)         # minnow trap hazard per trap
        self.sel_log = torch.zeros(B, n, **z)          # fukui trap hazard per trap
        self.recruit_dist = torch.zeros(B, n, **z)     # size distribution of recruits
        self.scenario = {k: torch.full((B,), float(v), **z) for k, v in self.scenario_defaults.items()}
        self.scenario["init_n_adult"] = torch.zeros(B, **z)
        self.theta = torch.zeros(B, len(PARAM_COLS), **z)  # standardized posterior draw of each env
        self.curr_month_prev = self.curr_month.clone()
        self.obs = self._initial_obs()
        self._arange = torch.arange(B, device=self.device)
        # all envs reset together and share the horizon, so episode ends are known on the
        # host (no device sync per step)
        self._t = 0
        self._done_true = torch.ones(B, dtype=torch.bool, device=self.device)
        self._done_false = torch.zeros(B, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def reset(self, seed=None):
        """Reset all envs. Returns (obs, info)."""
        if seed is not None:
            self.gen.manual_seed(seed)
        self._reset_envs(self._done_true)
        self._t = 0
        return self._clone_obs(self.obs), {}

    def step(self, action):
        """
        action: (B, 2) tensor. In [-1, 1] if normalized, else number of traps
        (minnow, fukui). Returns (obs, reward, terminated, truncated, info);
        finished envs are auto-reset and their last obs is in info["final_obs"].
        As in the CPU env, terminated == truncated at the end of an episode.
        """
        action = torch.as_tensor(action, dtype=self.dtype, device=self.device)
        out = self._graph_step(action) if self.cuda_graph else self._dynamics(action)
        obs, reward = self._clone_obs(out["obs"]), out["reward"].clone()
        self._t += 1
        info = {}
        if self._t > self.Tmax:
            done = self._done_true
            info["final_obs"] = self._clone_obs(obs)
            self._reset_envs(done)
            self._t = 0
            obs = self._clone_obs(self.obs)
        else:
            done = self._done_false
            self.obs = obs
        return obs, reward, done.clone(), done.clone(), info

    def _dynamics(self, action):
        """One month for all envs. Updates state tensors in place (required for CUDA graph replay)."""
        if self.normalized:
            a = self.max_action * (1 + action.clamp(-1, 1)) / 2
        else:
            a = action.clamp(min=0)

        # size-selective harvest
        hazard = self.sel_norm * a[:, :1] + self.sel_log * a[:, 1:]
        harvest_rate = (1 - torch.exp(-hazard)).clamp(0, 1)
        removed = torch.binomial(torch.floor(self.pop), harvest_rate, generator=self.gen)

        # growth + survival for this month, recruits arrive in May. The projection is an
        # elementwise multiply-and-sum rather than a matmul so that TF32 matmul settings
        # used for neural networks never reduce the precision of the population dynamics.
        P = self.proj[self._arange, self.curr_month - 4]
        next_pop = (P * (self.pop - removed).unsqueeze(-2)).sum(-1)
        next_pop = next_pop + (self.curr_month == 5).unsqueeze(-1) * self.recruit_sizes
        self.pop.copy_(next_pop.clamp(min=0))

        # observation; month is set below, after the month advances
        crab_counts = removed.sum(-1)
        biomass_caught = (removed * self.biomass_size).sum(-1)
        mean_biomass = torch.where(crab_counts > 0, biomass_caught / crab_counts.clamp(min=1), 0.0)
        obs = self._make_obs(removed, crab_counts, mean_biomass, a)

        reward = self._reward(a)

        self.curr_month_prev.copy_(self.curr_month)
        self.month_passed += 1
        self.curr_month += 1
        self._overwinter(self.curr_month > 10)
        if self.has_time:
            # twoActEnv reports the month just trapped; TwoActNormalized builds its obs
            # after the base step, so it reports the upcoming month (5, ..., 10, 4)
            obs["months"] = (self.curr_month if self.normalized else self.curr_month_prev).clone()
        return {"obs": obs, "reward": reward}

    def _graph_step(self, action, warmup=3):
        """Run _dynamics eagerly for a few warm-up steps (on a side stream), then capture it once and replay."""
        if self._graph is None:
            if self._graph_calls < warmup:
                self._graph_calls += 1
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    out = self._dynamics(action)
                torch.cuda.current_stream().wait_stream(s)
                return out
            self._g_action = action.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._g_out = self._dynamics(self._g_action)
            self._graph = g
        self._g_action.copy_(action)
        self._graph.replay()
        return self._g_out

    def flatten_obs(self, obs):
        """(B, flat_obs_dim) tensor: crabs ++ one_hot(months, 12) [++ scenario], as SB3's CombinedExtractor sees it."""
        parts = [obs["crabs"]]
        if self.has_time:
            parts.append(torch.nn.functional.one_hot(obs["months"], N_MONTHS).to(self.dtype))
        if self.observe_scenario:
            parts.append(obs["scenario"])
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    @property
    def privileged_dim(self):
        return len(self.scenario_obs_ranges) + self.nsize + len(PARAM_COLS)

    def privileged(self):
        """
        Hidden information for training-time use only (e.g. a privileged critic or
        auxiliary prediction targets): scenario parameters scaled to [-1, 1] over
        their ranges, log population by size bin, and the standardized posterior draw.
        """
        parts = [self._scenario_obs()] if self.scenario_obs_ranges else []
        parts += [torch.log1p(self.pop) / 5, self.theta]
        return torch.cat(parts, -1)

    def set_scenario_ranges(self, ranges):
        """Change the scenario sampling ranges used by future resets (e.g. for a curriculum)."""
        self.scenario_ranges = dict(ranges)

    def spaces(self):
        """Single-env gymnasium (observation_space, action_space), identical to the CPU env's."""
        from rl4greencrab.envs.twoAction_env import twoActEnv
        from rl4greencrab.envs.twoAction_norm import TwoActNormalized

        cfg = {k: v for k, v in self.config.items() if k != "param_df"}
        cpu = TwoActNormalized(cfg) if self.normalized else twoActEnv(cfg)
        return cpu.observation_space, cpu.action_space

    # ------------------------------------------------------------------ #
    # dynamics
    # ------------------------------------------------------------------ #

    def _reward(self, a):
        biomass = (self.biomass_size * self.pop).sum(-1)
        eco = -self.loss_a / (1 + torch.exp(-self.loss_b * (biomass / self.area - self.loss_c)))
        cost = (self.action_reward_scale * (a / self.max_action) ** self.action_reward_exponent).sum(-1)
        return eco - cost

    def _overwinter(self, winter):
        """Apply overwinter survival and draw next year's recruits for envs in `winter`."""
        pop = self.pop
        w = winter.unsqueeze(-1)
        grown = (self.overwinter * pop.unsqueeze(-2)).sum(-1)
        # zero trials outside winter so those draws are near-free
        new_adults = torch.binomial(
            torch.where(w, torch.floor(grown), 0.0), self.w_mort_exp.expand_as(grown).contiguous(), generator=self.gen
        )

        total = pop.sum(-1)
        B = self.num_envs
        sc = self.scenario
        local = sc["r"] * total * (1 - total / sc["K"]) + self.env_stoch * self._randn(B)
        big = torch.rand(B, generator=self.gen, device=self.device, dtype=self.dtype) < sc["p_big"]
        migrants = sc["mig_scale"] * torch.where(big, 80000 + 10000 * self._randn(B), 8000 + 1000 * self._randn(B))
        nonlocal_ = (migrants * (1 - total / sc["K"])).clamp(min=0)
        recruits = self.recruit_dist * (local + nonlocal_).unsqueeze(-1)

        self.recruit_sizes.copy_(torch.where(w, recruits, self.recruit_sizes))
        self.pop.copy_(torch.where(w, new_adults.clamp(min=0), pop))
        self.curr_month.copy_(torch.where(winter, 4, self.curr_month))

    def _reset_envs(self, mask, draw=None, chunk=16384):
        """Draw fresh posterior parameters and initial state for envs where mask is True."""
        idx_all = mask.nonzero(as_tuple=True)[0]
        # chunked so the float64 kernel temporaries stay small for very large batches
        for i in range(0, idx_all.numel(), chunk):
            self._reset_idx(idx_all[i:i + chunk], None if draw is None else draw[i:i + chunk])
        init = self._initial_obs()
        for k in self.obs:
            self.obs[k] = torch.where(mask.view(-1, *[1] * (init[k].dim() - 1)), init[k], self.obs[k])

    def _reset_idx(self, idx, draw=None):
        m = idx.numel()
        if draw is None:
            draw = torch.randint(0, self.posterior.shape[0], (m,), generator=self.gen, device=self.device)
        p = {name: self.posterior[draw, i] for i, name in enumerate(PARAM_COLS)}
        self.theta[idx] = ((self.posterior[draw] - self.post_mean) / self.post_std).to(self.dtype)
        col = lambda v: v.unsqueeze(-1)  # (m,) -> (m, 1) for broadcasting over size bins

        x = self.midpts64
        up, lo = self.bndry64[1:], self.bndry64[:-1]

        # monthly projection matrices (growth kernel * survival) and overwinter kernel
        D1 = torch.cat([self.D[:7], self.D[7:8]])
        D2 = torch.cat([self.D[1:8], self.D[0:1] + 1.0])
        kern = self._growth_kernel(p, D1, D2)                                     # (m, 8, n, n)
        surv = torch.exp(-(D2[:7].view(1, 7, 1) - D1[:7].view(1, 7, 1))
                         * (p["mort_beta"].view(m, 1, 1) + p["mort_alpha"].view(m, 1, 1) / x**2))
        self.proj[idx] = (kern[:, :7] * surv.unsqueeze(-2)).to(self.dtype)
        self.overwinter[idx] = kern[:, 7].to(self.dtype)

        # trap selectivity curves
        self.sel_norm[idx] = (col(p["trapm_pmax"]) * torch.exp(
            -(x - col(p["trapm_xmax"])) ** 2 / (2 * col(p["trapm_sigma"]) ** 2))).to(self.dtype)
        self.sel_log[idx] = (col(p["trapf_pmax"]) / (1 + torch.exp(
            -col(p["trapf_k"]) * (x - col(p["trapf_midpoint"]))))).to(self.dtype)

        # recruit size distribution: gamma with given mean / sd
        var = p["init_sd_recruit"] ** 2
        shape, rate = col(p["init_mean_recruit"] ** 2 / var), col(p["init_mean_recruit"] / var)
        self.recruit_dist[idx] = (torch.special.gammainc(shape, up * rate)
                                  - torch.special.gammainc(shape, lo * rate)).to(self.dtype)
        if self.reset_recruits:
            self.recruit_sizes[idx] = 0

        for name in self.scenario:
            if name in self.scenario_ranges:
                self.scenario[name][idx] = self._draw(self.scenario_ranges[name], m)
            elif name != "init_n_adult":
                self.scenario[name][idx] = float(self.scenario_defaults[name])

        # initial adults: lognormal size distribution
        if "init_n_adult" in self.scenario_ranges:
            n_adult = self.scenario["init_n_adult"][idx].double()
        elif self.random_start:
            n_adult = torch.randint(0, self.max_obs + 1, (m,), generator=self.gen, device=self.device).double()
        else:
            n_adult = torch.full((m,), float(self.init_n_adult), dtype=torch.float64, device=self.device)
        self.scenario["init_n_adult"][idx] = n_adult.to(self.dtype)
        mu, s = col(p["init_mean_adult"]), col(p["init_sd_adult"])
        lncdf = lambda b: torch.special.ndtr((torch.log(b) - mu) / s)
        self.pop[idx] = ((lncdf(up) - lncdf(lo)) * col(n_adult)).to(self.dtype)

        self.curr_month[idx] = 4
        self.month_passed[idx] = 0

    def _growth_kernel(self, p, D1, D2):
        """Seasonal von Bertalanffy growth kernels, (m, T, n, n); column j = source size bin."""
        m = p["growth_k"].shape[0]
        k, A, ds = (p[c].view(m, 1) for c in ("growth_k", "growth_A", "growth_ds"))
        S_t = (A * k / (2 * math.pi)) * torch.sin(2 * math.pi * (D2 - ds))
        S_t0 = (A * k / (2 * math.pi)) * torch.sin(2 * math.pi * (D1 - ds))
        growth = 1 - torch.exp(-k * (D2 - D1) - S_t + S_t0)                        # (m, T)
        x = self.midpts64
        means = x + (p["growth_xinf"].view(m, 1, 1) - x) * growth.unsqueeze(-1)   # (m, T, n)
        sd = p["growth_sd"].view(m, 1, 1, 1)
        z = lambda b: (b.view(1, 1, -1, 1) - means.unsqueeze(-2)) / sd             # (m, T, n_to, n_from)
        kern = torch.special.ndtr(z(self.bndry64[1:])) - torch.special.ndtr(z(self.bndry64[:-1]))
        return kern / kern.sum(-2, keepdim=True)

    # ------------------------------------------------------------------ #
    # observations
    # ------------------------------------------------------------------ #

    def _make_obs(self, removed, crab_counts, mean_biomass, a):
        t = self.observation_type
        if self.normalized:
            effort = a.sum(-1, keepdim=True)
            denom = (self.cpue_normalization * effort).clamp(min=1e-30)
            if "size" in t:
                cpue = torch.where(effort > 0, removed / denom, 0.0)
            else:
                cpue = torch.where(effort > 0, crab_counts.unsqueeze(-1) / denom, 0.0)
            count = 2 * cpue - 1
            b0, b1 = self.biomass_size[0], self.biomass_size[-1]
            bio = (-1 + 2 * (mean_biomass - b0) / (b1 - b0)).unsqueeze(-1)
        else:
            count = removed if "size" in t else crab_counts.unsqueeze(-1)
            bio = mean_biomass.unsqueeze(-1)

        if t.startswith("count-biomass"):
            crabs = torch.cat([count, bio], -1)
        elif t.startswith("biomass"):
            crabs = bio
        else:
            crabs = count
        obs = {"crabs": crabs.to(self.dtype)}
        if self.observe_scenario:
            obs["scenario"] = self._scenario_obs()
        return obs

    def _initial_obs(self):
        fill = -1.0 if self.normalized else 0.0
        obs = {"crabs": torch.full((self.num_envs, self.obs_dim), fill, dtype=self.dtype, device=self.device)}
        if self.has_time:
            obs["months"] = torch.full((self.num_envs,), 4, dtype=torch.long, device=self.device)
        if self.observe_scenario:
            obs["scenario"] = self._scenario_obs()
        return obs

    def _scenario_obs(self):
        if not self.scenario_obs_ranges:
            return torch.zeros(self.num_envs, 0, dtype=self.dtype, device=self.device)
        cols = []
        for name, rng in self.scenario_obs_ranges.items():
            lo, hi = float(rng[0]), float(rng[1])
            v = self.scenario[name]
            if len(rng) > 2 and rng[2] == "log":
                v, lo, hi = torch.log(v.clamp(min=1e-12)), np.log(lo), np.log(hi)
            cols.append(2 * (v - lo) / max(hi - lo, 1e-12) - 1)
        return torch.stack(cols, -1).to(self.dtype)

    def _draw(self, rng, m):
        lo, hi = float(rng[0]), float(rng[1])
        u = torch.rand(m, generator=self.gen, device=self.device, dtype=torch.float64)
        if len(rng) > 2 and rng[2] == "log":
            return torch.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))).to(self.dtype)
        return (lo + u * (hi - lo)).to(self.dtype)

    def _randn(self, n):
        return torch.randn(n, generator=self.gen, device=self.device, dtype=self.dtype)

    @staticmethod
    def _clone_obs(obs):
        return {k: v.clone() for k, v in obs.items()}
