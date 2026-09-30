"""Combine scenario evaluations into regret tables (results/scenario_summary.csv)."""
import glob

import pandas as pd

e = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/scenario_eval*.csv"))])
b = pd.read_csv("results/scenario_baselines.csv")
e["group"] = e.policy.str.replace(r"-s\d+-best$", "", regex=True).str.replace(r"-best$", "", regex=True)
spec = e[e.policy.str.startswith("spec-")].copy()
spec["own"] = spec.policy.str.extract(r"spec-(s\d+)")[0]
spec_own = spec[spec.scenario == spec.own].set_index("scenario")["mean"]
gen = e[~e.policy.str.startswith("spec-")]
nseeds = gen.groupby("group").policy.nunique()
G = gen.groupby(["group", "scenario"])["mean"].mean().unstack(0)
T = G.join(b.pivot_table(index="scenario", columns="policy", values="mean"))
T["ppo-specialist"] = spec_own
pts = [f"s{i:02d}" for i in range(1, 25)]
sub = T.loc[pts]
regret = sub.sub(sub["ppo-specialist"], axis=0)
summary = pd.DataFrame({
    "seeds": nseeds,
    "nominal": T.loc["nominal"], "wide": T.loc["wide"],
    "mean_s01_24": sub.mean(), "regret_mean": regret.mean(), "regret_worst": regret.min(),
}).sort_values("regret_mean", ascending=False)
summary.to_csv("results/scenario_summary.csv")
print(summary.round(3).to_string())
print("\nper-scenario regret vs PPO specialist:")
cols = ["rob-oracle", "rob-wide", "big-oracle", "big-wide", "big-wide-h16", "big-curr", "v2-A", "seasonal-specialist"]
print(regret[[c for c in cols if c in regret]].round(2).join(sub["ppo-specialist"].round(2)).to_string())
