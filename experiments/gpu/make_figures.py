"""Figures for the README (results/figures/*.png). Re-run after new results: python make_figures.py"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import RESULTS, test_scenarios

FIG = os.path.join(RESULTS, "figures")
os.makedirs(FIG, exist_ok=True)

# reference palette (validated: categorical slots 1-4 pass CVD/normal-vision checks on this surface;
# slots 3-4 are below 3:1 contrast, so every line is direct-labeled)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def curves(prefix):
    files = sorted(glob.glob(os.path.join(RESULTS, "curves", f"{prefix}-s*.csv")))
    return [pd.read_csv(f) for f in files if pd.read_csv(f).shape[0] > 1]


def spread(ys, min_gap):
    """Nudge label y-positions apart (in data units) so direct labels don't collide."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for k in range(1, len(order)):
        a, b = order[k - 1], order[k]
        out[b] = max(out[b], out[a] + min_gap)
    return out


def line_panel(ax, groups, refs, xmax, title, ref_x, label_gap=0.14, markers=False, connect=True, colors=SERIES,
               xlabel="Training steps (millions)"):
    """groups: list of (label, [curve dfs]); each seed a thin line, the seed mean a 2px line, direct-labeled."""
    ends = []
    for (label, dfs), color in zip(groups, colors):
        if not dfs:
            continue
        for d in dfs if connect else []:
            ax.plot(d.steps / 1e6, d.eval_mean, color=color, lw=0.8, alpha=0.35)
        xs = np.unique(np.concatenate([d.steps.values for d in dfs]))
        xs = xs[xs <= min(d.steps.max() for d in dfs)]
        mean = np.mean([np.interp(xs, d.steps, d.eval_mean) for d in dfs], axis=0)
        ax.plot(xs / 1e6, mean, color=color, lw=2 if connect else 0, label=f"{label} (n={len(dfs)})",
                marker="o" if markers else None, ms=4 if connect else 12 - 4 * len(ends), mec=SURFACE, mew=1)
        ends.append((label, xs[-1] / 1e6, mean[-1]))
    ys = spread([e[2] for e in ends], label_gap)
    for (label, x, y0), y in zip(ends, ys):
        ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    for y, text in refs:
        ax.axhline(y, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.annotate(text, (ref_x, y), xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Episode reward (closer to 0 is better)")
    ax.set_title(title, loc="left")


# ---- Figure 1: training length on the nominal model ----
sb3, gpu_const, gpu_decay = curves("sb3-ppo"), curves("conv-default"), curves("v2-A")
refs = [(-9.67, "best constant policy"), (-7.71, "best seasonal schedule")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 2]})
# left: zoom on the first 10M steps (points = held-out evaluations; GPU runs were evaluated every ~5M steps)
first10 = lambda dfs: [d[d.steps <= 10.5e6] for d in dfs]
line_panel(a1, [("SB3, original settings", first10(sb3))], refs, 10, "First 10 million steps", ref_x=0.2, markers=True)
# GPU runs were only evaluated at 0 and ~9.4M steps in this window: show the evaluations as points, not lines
line_panel(a1, [("GPU PPO, constant lr", first10(gpu_const)), ("GPU PPO, decaying lr", first10(gpu_decay))], [], 10,
           "First 10 million steps", ref_x=0.2, label_gap=0.3, markers=True, connect=False, colors=SERIES[1:])
a1.set_xlim(0, 13.5)
a1.set_xticks([0, 2, 4, 6, 8, 10])
a1.set_ylim(-11, -6)
# right: full runs
line_panel(a2, [("SB3, original settings", sb3), ("GPU PPO, constant lr", gpu_const), ("GPU PPO, decaying lr", gpu_decay)],
           refs, 300, "Full training runs", ref_x=150)
a2.set_xlim(0, 360)
a2.set_xticks([0, 50, 100, 150, 200, 250, 300])
a2.axvline(10, color=INK2, lw=1, ls=":")
a2.annotate("10M", (10, -6.05), xytext=(3, 0), textcoords="offset points", va="top", fontsize=8.5, color=INK2)
a2.set_ylim(-11, -6)
a2.legend(loc="lower right", frameon=False, fontsize=8.5)
fig.suptitle("How long to train: held-out reward during PPO training (nominal scenario)", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_training_length.png"), dpi=150)
plt.close(fig)

# ---- Figure 2: generalists on the wide scenario range ----
fig, ax = plt.subplots(figsize=(10, 5))
groups = [
    ("No memory", curves("rob-wide")),
    ("Memory (GRU)", curves("rppo-wide-mb32") + curves("rppo32-wide")),
    ("Memory + privileged critic", curves("gru-priv")),
    ("Oracle (told the scenario)", curves("rppo32-oracle")),
]
line_panel(ax, groups, [(-10.57, "best constant policy (wide)"), (-9.33, "best seasonal schedule (wide)")], 150,
           "Generalist agents trained on the wide range of invasion scenarios", ref_x=40, label_gap=0.22)
ax.set_ylim(-11.5, -6.3)
ax.set_xlim(0, 200)
ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_generalists_wide.png"), dpi=150)
plt.close(fig)

# ---- Figure 3: regret heatmap ----
exec(open(os.path.join(os.path.dirname(__file__), "analyze_scenarios.py")).read().split("summary =")[0])
rows = [
    ("Oracle, memory (not deployable)", "rppo32-oracle"),
    ("Generalist, memory + privileged critic", "gru-priv"),
    ("Estimate, then act (deployable)", "estimate-then-act"),
    ("Generalist, memory (GRU)", "rppo-wide-mb32"),
    ("Generalist, no memory", "rob-wide"),
    ("Seasonal schedule tuned per scenario", "seasonal-specialist"),
    ("Nominal-trained, no memory", "v2-A"),
    ("Nominal-trained, memory", "rppo32-nominal"),
    ("One seasonal schedule (wide)", "seasonal-wide"),
    ("One constant policy (wide)", "constant-wide"),
]
pts = test_scenarios()
order = sorted(range(1, 25), key=lambda i: pts[i]["mig_scale"])
cols = [f"s{i:02d}" for i in order]
M = np.array([[regret.loc[c, key] for c in cols] for _, key in rows])
loss = np.clip(-M, 0, None)  # loss vs best specialist; the few small gains (>0) shown as 0
vmax = 4.0
cmap = LinearSegmentedColormap.from_list("blues", BLUES)
fig, ax = plt.subplots(figsize=(13, 5.2))
im = ax.imshow(np.minimum(loss, vmax), cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
# 2px-style surface gaps between cells
ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.grid(which="major", visible=False)
ax.tick_params(which="minor", length=0)
ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
ax.set_xticks(range(len(cols)), [f"{pts[i]['mig_scale']:.1f}x" for i in order], fontsize=8.5)
ax.set_xlabel("Test scenario, sorted by migrant pressure (1x = nominal)")
avg = [-regret[key].mean() for _, key in rows]
for r, v in enumerate(avg):
    ax.annotate(f"{v:.2f}", (len(cols) - 0.5, r), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
ax.annotate("avg loss", (len(cols) - 0.5, -0.5), xytext=(8, 6), textcoords="offset points", fontsize=8.5, color=INK2)
hi = [k for k, i in enumerate(order) if pts[i]["mig_scale"] > 3]
ax.plot([hi[0] - 0.4, hi[-1] + 0.4], [-0.9, -0.9], color=INK2, lw=1.5, clip_on=False)
ax.annotate("high migrant pressure", ((hi[0] + hi[-1]) / 2, -0.9), xytext=(0, 4), textcoords="offset points",
            ha="center", fontsize=8.5, color=INK2, annotation_clip=False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.08)
cb.set_label("Reward lost vs. best specialist (per episode)")
cb.outline.set_visible(False)
ax.set_title("Robustness: reward lost in each test scenario relative to an agent trained for that scenario",
             loc="left", pad=34)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_regret_heatmap.png"), dpi=150)
plt.close(fig)
pd.DataFrame(M, index=[r[0] for r in rows], columns=[f"{c} ({pts[int(c[1:])]['mig_scale']:.1f}x)" for c in cols]).round(3).to_csv(
    os.path.join(FIG, "fig3_regret_table.csv"))
print("wrote", sorted(os.listdir(FIG)))
