# GPU experiments: faster simulation and robust RL policies for green crab management

This folder contains the experiments run with the GPU (graphics card) version of the green crab simulation. The goals were (1) to make simulation and RL training fast enough to ask questions that were impractical before, and (2) to use that speed to test how good, how stable, and how robust the RL trapping policies are.

## Top-level findings

1. **The GPU simulator is ~900x faster, and RL training ~130x faster.** One CPU process simulates ~6,500 crab-months per second; the GPU version simulates ~5.9 million. Training an RL agent for 10 million steps drops from about an hour to under a minute. This made the hundreds of training runs below possible.

2. **A bug in the original simulator was found and fixed.** Recruits from the end of one simulated episode carried over into the first year of the next one ([issue #39](https://github.com/boettiger-lab/rl4greencrab/issues/39)). All GPU results use the corrected model. The manuscript's agents, re-scored on the corrected model, still perform about as reported.

3. **Most of RL's advantage over a constant trapping policy comes from seasonality.** A fixed month-by-month trapping schedule (no observations used) closes most of the gap between the best constant policy and the manuscript's RL agents. Responding to catch data adds a smaller, real improvement on top.

4. **Training length and settings matter more than expected.** With the settings used in the manuscript, PPO peaked within a few million steps and then *degraded*. With a decaying learning rate and keeping the best checkpoint, training is stable and keeps improving to roughly 100 million steps. 10 million steps was not enough with our GPU settings. Trained this way, PPO *without* memory beats the manuscript's agents (same algorithm and observations) by about 1-1.3 reward points per episode on average and 0.4-0.5 points against the best previous replicate, with consistent replicates (section 2).

5. **Training across the range of plausible invasions (our curriculum approach) trades a small cost for robustness.** The policies differ in how much they assume about the invasion:
   - A **specialist** assumes the exact scenario is known. It is the best achievable for that scenario: an upper bound, not something a manager could actually use.
   - An agent trained only on the **nominal** model (as in the manuscript) assumes that scenario. It is excellent when the assumption holds (-5.99 for an agent with memory) but loses 1.6-3.9 reward points per episode on average when the invasion differs (different carrying capacity, migrant pressure, etc.).
   - A **generalist** assumes only that the invasion lies somewhere in a plausible range. As expected from its weaker assumptions, it gives up a little: ~0.55 points per episode relative to the specialists, and ~0.4 on the nominal scenario relative to the nominal-trained agent. In return it is robust everywhere, and far outperforms fixed schedules.

6. **Memory is the most important design choice.** Agents that remember the whole catch history of an episode, rather than reacting only to the latest month's catch, do better everywhere. On the nominal model a memory agent is the best of all (-5.99 vs -6.19 without memory). Across the wide range, memory cuts the average loss relative to scenario specialists from 0.89 to about 0.6. The type of memory mattered little: a transformer (the architecture behind modern language models) did no better than a simpler recurrent network (GRU). Giving the training-only critic access to the true hidden state (a "privileged critic") helped a little more (0.55). Memory also has a downside: trained on the nominal scenario only, memory agents are the most brittle of all (finding 5).

7. **Training techniques for the generalist made little or no difference.** A staged schedule that starts on the nominal scenario and gradually widens to the full range performed the same as or slightly worse than training on the full range from the start (average loss 0.93 vs 0.90 for otherwise identical runs without memory; 1.22 vs 1.12 with larger networks). Over-sampling the hardest (high-migration) scenarios gave no overall gain either. Neither did larger networks, longer training, giving memory-less agents a short window of recent observations, or an extra training task asking the memory to predict the hidden state.

8. **The remaining shortfall is concentrated in high-migration scenarios, and it is a timing problem the agents fail to learn.** Where migrant pressure is very high, the best strategy traps early in the season and nearly stops in September-October (late effort is wasted when a new wave of migrants replaces the removed crabs). The broadly trained agents keep the usual late-season pattern, just with more traps. This is not for lack of information: migrant pressure can be inferred from the catch history within ~3-7 years. Over-sampling these scenarios in training barely helped. A deployable "estimate, then act" design (infer the scenario from catches, then apply a policy trained with known scenarios) improves the worst cases but not the average. Closing this gap remains open.

## Key terms

| Term | Meaning here |
|---|---|
| **Episode** | One simulated management period: 101 monthly decisions (about 14 years of April-October trapping seasons). |
| **Reward / score** | The quantity the agent maximizes, summed over an episode: ecological damage (a function of crab biomass) plus the cost of traps. It is always negative; **closer to zero is better**. Typical values are -6 to -10; doing nothing scores about -23. |
| **Policy / agent** | A rule mapping what the manager observes (catch per trap, mean size of caught crabs, month) to how many minnow and Fukui traps to set. |
| **Constant policy** | Same number of traps every month, regardless of observations. |
| **Seasonal schedule** | A fixed number of traps for each calendar month (same every year), regardless of observations. |
| **Nominal scenario** | The manuscript's model: carrying capacity 25,000, usual migrant pressure, 0-2,000 initial adults. |
| **Wide scenarios** | A broad range of invasion conditions (table below), drawn at random each episode. |
| **Generalist** | An agent trained on the wide range of scenarios, drawn at random each episode. This is our curriculum approach: the agent assumes only that the invasion lies somewhere in that range, rather than assuming specific parameter values. |
| **Specialist** | An agent trained on one specific scenario only, i.e. assuming that scenario is known exactly. Used as a yardstick (an upper bound) for what is achievable in that scenario. |
| **Regret** | How much reward a policy loses in a scenario compared with the best specialist for that scenario. 0 = as good as the specialist; -0.5 = half a reward point worse per episode. |
| **Oracle** | An agent that is *told* the true scenario parameters (carrying capacity, migrant pressure, ...). Not usable in practice; it measures how much is lost by not knowing the scenario. |
| **Privileged critic** | A training aid. RL training uses a second network (the "critic") that estimates future reward to score the agent's choices. A privileged critic is shown the true hidden state of the simulation during training only. The deployed policy never sees it, so the resulting agent is usable in practice. |
| **Memory (recurrent / transformer)** | Agents that remember the whole catch history of the episode instead of only the latest observation. |
| **Seed** | An independent replicate training run. Results are averaged over seeds. |

**The wide scenario range:**

| Parameter | Nominal | Wide range |
|---|---|---|
| Initial adult crabs | 0-2,000 | 0-20,000 |
| Carrying capacity K | 25,000 | 10,000-60,000 |
| Local recruitment rate r | 1 | 0.5-2 |
| Migrant (propagule) pressure | 1x | 0.1x-5x |
| Probability of a large migration pulse in a year | 0.2 | 0-0.5 |

Biological parameters (growth, mortality, trap selectivity) are drawn from the Bayesian posterior in every episode in all experiments, as in the manuscript.

## What was built

- **GPU simulator** (`src/rl4greencrab/envs/gpu_env.py`): the same population model as the original environment, run for thousands of independent episodes at once. Its growth kernels, trap selectivity, initial state and reward match the original to numerical precision, and whole-episode results agree statistically (`tests/test_gpu_env.py`).
- **GPU training algorithms** (`src/rl4greencrab/agents/`): PPO (`gpu_ppo.py`), PPO with memory (GRU or transformer; `gpu_rppo.py`) and TD3 (`gpu_td3.py`), all running entirely on the GPU.
- Tools to evaluate any policy (including the manuscript's saved agents) on thousands of episodes in seconds.

## Experiments

All scores below are average episode rewards on held-out simulations (thousands of episodes, so sampling error is about +/-0.02 unless noted).

### 1. Baselines on the corrected model

| Policy | Score |
|---|---|
| No trapping | -23.16 |
| Best constant policy | -9.67 |
| Best seasonal schedule | -7.71 |
| Manuscript agents (TQC, catch + biomass + month), mean of 10 | -7.36 |
| Best manuscript agent (any algorithm) | -6.71 |

The best seasonal schedule sets almost no traps from April to July and traps heavily in August-October. That simple pattern accounts for most of the improvement over the constant policy (finding 3). Scored on the original (buggy) model, the manuscript agents reproduce the manuscript's Table 1 closely, which confirms the GPU simulator matches the original.

### 2. How long should we train?

| Training setup | Score at 10M steps | Best checkpoint | Final checkpoint | What happened later |
|---|---|---|---|---|
| Original settings (SB3 PPO, 12 envs) | collapsed (-10.7) | -6.74 (at 6M) | -10.7 (10M) | degraded after ~6M steps |
| GPU PPO, constant learning rate | -6.81 | -6.24 | -7.6 (300M; mean of 3) | 2 of 3 replicates degraded badly |
| GPU PPO, decaying learning rate | -6.83 | **-6.21** | **-6.23** (150M; mean of 3) | stable; plateau by ~100M |

![Held-out reward during training on the nominal scenario](results/figures/fig1_training_length.png)

*Figure 1. Held-out reward during training (higher is better). Left: the first 10 million steps. The original settings (blue) learn fast, then collapse. The GPU runs were only evaluated at 0 and ~9.4M steps in this window, so they are shown as points. Right: full runs. With a constant learning rate (orange), two of three replicates degrade after ~150M steps; with a decaying learning rate (green), training is stable. Thin lines are individual replicates, thick lines their mean.*

The original settings learn quickly per step but are unstable; this likely explains part of the large variation between replicate agents in the manuscript. A decaying learning rate plus keeping the best-scoring checkpoint gives reliable training (finding 4).

**Comparison with the manuscript's agents (no memory).** Same algorithm (PPO), same observations (catch per trap, mean biomass of the catch, month), scored on the same held-out simulations:

| | Original model (with the recruit bug) | Corrected model |
|---|---|---|
| Manuscript's reported PPO score (mean of 10 replicates) | -7.66 | — |
| Manuscript's PPO agents, re-scored here (mean of 10) | -7.86 | -7.55 |
| Best manuscript PPO replicate | -7.02 | -6.75 |
| Manuscript's TQC agents (its best algorithm), mean / best | -7.76 / -7.55 | -7.36 / -7.16 |
| Best of all 120 manuscript agents | — | -6.71 |
| **GPU PPO, no memory (3 replicates)** | **-6.55 to -6.60** | **-6.18 to -6.24** |

- The new agents beat the manuscript's by about 1-1.3 points on average and by about 0.4-0.5 points against the best previous replicate, on either version of the model. The new agents were trained on the corrected model, so the original model is unfamiliar to them; they are still clearly better there.
- The new replicates agree closely (spread ~0.06), whereas the manuscript's replicates varied widely.
- **What made the difference is not the GPU itself but what it made affordable:** 150 million training steps instead of 10 million (about 14 hours per run on the original CPU setup, about 5 minutes on the GPU), combined with a decaying learning rate and keeping the best checkpoint. Retraining with the original settings on the corrected model peaked at -6.74 (matching the best previous replicate) and then collapsed, so the original setup tops out around -6.7.
- **Fairness caveats:** (1) The manuscript's agents were trained on the original model and are scored here on the corrected one; this does not disadvantage them (they score *better* on the corrected model, -7.55 vs -7.86). (2) Our agents use the best checkpoint, selected on the same held-out simulations used for scoring, which slightly flatters them; the manuscript's agents are simply their final model at 10M steps. With the decaying learning rate the bias is small: final checkpoints score -6.23 on average versus -6.21 for the best.

### 3. Algorithm and setting variations on the nominal model

- **Discounting** (gamma 0.999 instead of 0.99): slightly worse.
- **Larger networks, observation history, squashed actions**: no improvement.
- **TD3** (an off-policy algorithm): about as good as PPO (-6.26), not better. Because simulation is cheap, PPO's need for many samples is no handicap.
- **Agents with memory (GRU)**: best on the nominal model (**-5.99**).

### 4. Robustness across invasion scenarios

Each policy was scored on 24 fixed test scenarios drawn from the wide range. "Regret" compares each scenario with the best specialist trained for that scenario.

| Policy | Score, nominal | Score, wide | Average regret | Worst-scenario regret |
|---|---|---|---|---|
| Oracle with memory (not deployable) | -6.06 | -6.56 | -0.21 | -1.28 |
| **Generalist with memory + privileged critic** (4 seeds) | -6.39 | **-6.90** | **-0.55** | -2.48 |
| Generalist with memory (transformer) | -6.69 | -6.99 | -0.60 | -2.43 |
| Generalist with memory (GRU) | -6.63 to -6.74 | -7.03 to -7.09 | -0.63 to -0.68 | -2.4 to -2.7 |
| Generalist without memory | -6.70 | -7.26 | -0.89 | -3.11 |
| Generalist without memory, staged widening schedule | -6.53 | -7.29 | -0.93 | -3.15 |
| Seasonal schedule tuned *for each scenario* | -7.77 | -9.33 | -1.51 | -3.46 |
| Agent without memory trained on nominal only | -6.19 | -7.73 | -1.58 | -4.60 |
| **Agent with memory trained on nominal only** | **-5.99** | -9.69 | **-3.85** | -6.41 |
| One seasonal schedule tuned on the wide range | — | -9.33 | -3.24 | -6.28 |
| One constant policy tuned on the wide range | — | -10.57 | -4.53 | -10.35 |

![Generalist learning curves on the wide scenario range](results/figures/fig2_generalists_wide.png)

*Figure 2. Held-out reward on the wide range of invasion scenarios during training. Memory and a privileged critic each add a step up. The oracle, which is told the scenario, shows how much is lost by having to infer it from catches.*

![Regret heatmap across 24 test scenarios](results/figures/fig3_regret_heatmap.png)

*Figure 3. Reward lost in each of the 24 test scenarios (columns, sorted by migrant pressure) relative to an agent trained for that scenario. Darker = larger loss; the right-hand numbers are the average loss. The few cases where a policy slightly beat the specialist are shown as 0. Values are in `results/figures/fig3_regret_table.csv`.*

What this shows:
- **Train on the uncertainty you actually have.** The agent with memory trained only on the nominal model is the best of all on the nominal model, and one of the worst anywhere else (finding 5). The generalists' regret is the price of assuming less, and it is small.
- **Memory helps most.** Remembering the catch history reduces regret from -0.89 to about -0.6.
- **A privileged critic helps a little more** (-0.55). It reaches the same performance as training twice as long.
- **Other training techniques did not help:** a staged schedule gradually widening the scenarios during training, larger networks, longer training, transformers instead of GRUs, and an auxiliary task asking the memory to predict the hidden state.
- **The oracle reaches -0.21**, so most of the remaining gap concerns *using* scenario information that the oracle is handed directly (section 5).

### 5. Why do generalists fall short in high-migration scenarios?

The four hardest test scenarios all have 3-4x the usual migrant pressure. There the generalist loses 1.3-2.5 reward points to the specialist. Comparing behavior (`behavior_hard.py`):

- The generalist does **not** under-trap. It sets about as many traps in total as the specialist.
- It **mis-times** them. Specialists and the oracle trap heavily in April-June and lightly in October. The generalist keeps the nominal pattern (light early, heavy August-October) and just scales it up.
- The result: in the hardest scenario, the generalist leaves 40-46k crabs where the specialist holds 28-33k.
- More precisely, the decisive difference is the late season: specialists and the oracle nearly stop trapping in September-October under high migration, while every generalist keeps trapping heavily (table below, mean traps per month in years 4-14).

| Scenario | Policy | Apr-Jun | Aug | Sep | Oct |
|---|---|---|---|---|---|
| 3.3x migration | Generalist | 812 | 5,451 | 4,292 | 2,541 |
| | Generalist, over-sampled training | 1,357 | 5,953 | 4,911 | 2,318 |
| | Oracle | 1,479 | 5,243 | 2,798 | 491 |
| | Specialist | 1,789 | 4,447 | 1,634 | 315 |

We tested three explanations:

1. **Too few high-migration scenarios in training?** Mostly no. Training with 43% instead of 13% high-migration episodes (`mix-himig`, `linmig` runs) moved effort somewhat earlier in the season but did not cut late-season trapping. Loss in the four high-migration scenarios improved only from 1.73 to ~1.55; overall regret was unchanged (-0.54 to -0.56).
2. **The catch data does not reveal migrant pressure?** No. A separate model trained to infer each scenario parameter from the catch history (`identifiability.py`) recovers migrant pressure well after a few years:

   | Inferred from catches (R²; 1 = perfect) | After 1 yr | 3 yrs | 7 yrs | 14 yrs |
   |---|---|---|---|---|
   | Initial adults | 0.99 | 0.99 | 0.99 | 0.99 |
   | Migrant pressure | 0.00 | 0.71 | 0.95 | 0.97 |
   | Carrying capacity K | 0.00 | 0.03 | 0.33 | 0.44 |
   | Local recruitment r | 0.00 | 0.10 | 0.45 | 0.58 |

   Migrants arrive only after the first winter, so year 1 says nothing about them; by year 3 most of the information is there.
3. **Can we use that information explicitly?** Partly. An "estimate, then act" policy (`estimate_then_act.py`) infers the scenario from catches and feeds the estimate to the oracle policy. It is deployable (it never sees the true scenario). It has the same average regret as the best generalist (-0.56) but a better worst case (-1.79 vs -2.48) and less loss in high-migration scenarios (-1.40 vs -1.73), at a small cost elsewhere (-0.39 vs -0.32). It falls short of the true oracle mainly in the early years, when the estimate is still poor but the policy acts as if it were certain.

**Bottom line:** the information needed to manage high-migration invasions well is in the catch data, but standard RL training does not learn to exploit it, and simple fixes recover only part of the gap. Policies that explicitly account for uncertainty in the scenario estimate are a natural next step.

### 6. Speed

Measured on the NVIDIA GB10 (DGX Spark):

| Task | Original (CPU) | GPU | Speed-up |
|---|---|---|---|
| Simulation, steps per second | 6,500 (1 core); 18,500 (20 cores) | 5.9 million | ~900x (1 core), ~300x (20 cores) |
| PPO training, steps per second | 2,900 (manuscript setup) | 385,000 (447,000 with CUDA graphs) | ~130-150x |
| GRU (memory) agent training | — | 166,000 (196,000 with TF32) | |
| Transformer agent training | — | 66,000 (78,000 with TF32) | |

Two optional speed-ups are switched on automatically where they help and the hardware supports them: **CUDA graphs**, which replay the simulation step without per-operation overhead, and **TF32 tensor cores** for the neural networks only. Neither changes the simulated population dynamics: the population projection always runs in full 32-bit precision. On older GPUs (e.g. the Quadro RTX 8000) or CPU, the code falls back automatically. Simulator throughput with CUDA graphs after the final optimizations: 2.9M (1,024 parallel episodes), 5.4M (4,096) and 5.85M (16,384) steps per second.

## Caveats

- Results are from simulation. The wide scenario ranges are our choice and should be reviewed by the team for ecological plausibility.
- The "best checkpoint" of each training run is selected on the same held-out simulations that are reported in the learning curves and the nominal/wide scores. This slightly flatters those numbers; for the main no-memory PPO runs the effect is about 0.02 (final checkpoints -6.23 vs best -6.21). The 24-scenario comparisons use separate test scenarios and are the fairer comparison.
- Comparisons with the manuscript's agents mix two differences: our agents were trained on the corrected model, and theirs on the original one. Both are scored on both models in section 2.
- Some configurations have 1-2 replicate seeds only (noted in the tables); differences smaller than ~0.05 in average regret are not meaningful.

## Files

| File | Purpose |
|---|---|
| `common.py` | Shared setup: scenario definitions (nominal, wide), evaluation helpers |
| `baselines.py` | Constant and seasonal policies; re-scoring the manuscript's agents |
| `train_ppo.py` | Train one agent (PPO, with or without memory, any scenario setting); writes a learning curve and the best checkpoint |
| `train_td3.py` | Train a TD3 agent |
| `train_sb3_ref.py` | Reference run with the original SB3 settings |
| `scenario_baselines.py` | Best constant and seasonal policies for each test scenario |
| `eval_scenarios.py`, `analyze_scenarios.py` | Score agents on the test scenarios; regret tables |
| `behavior_hard.py` | Behavior comparison in high-migration scenarios |
| `identifiability.py` | How well each scenario parameter can be inferred from catch history |
| `estimate_then_act.py` | Deployable policy: scenario estimator + oracle policy |
| `make_figures.py` | Regenerates the figures in `results/figures/` from the results files |
| `benchmark.py`, `benchmark_speedups.py` | Speed benchmarks |
| `jobs_*.txt`, `run_jobs.sh` | The exact training runs, launched in parallel |
| `results/` | All results: `baselines.csv`, `scenario_summary.csv` (main robustness table), `scenario_eval*.csv`, `behavior_hard.csv`, `benchmark*.csv`, and `curves/` (learning curves for every run) |

To reproduce a run, e.g. the best generalist:

```
python train_ppo.py --tag my-run --recurrent --critic privileged --scenario wide --anneal --seq-minibatches 32 --steps 1.5e8
python eval_scenarios.py results/my_eval.csv my-run-best
```
