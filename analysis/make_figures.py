"""The paper's figures 2, 3 and 4, from the run analyses (figure 1 is writeup/make_figure.py --paper).

    python -I analysis/make_figures.py RUN_OUTPUTS [OUT]    (needs matplotlib)

Figure 2, chars-by-level.png: generated characters per level, one thin line per game over the two base runs and
the median in front, log scale; the medians by level (the numbers Table 5 states) go to chars-by-level.json.
Figure 3, two-runs.png: two runs of the identical notebook on the 41 generated games, per game: tokens (left) and
score (right), run a against its repeat a3; Spearman annotated from the data (the paper's 0.79 and 0.56).
Figure 4, goal-pairs.png: the goal pairs' cost per completed level, consequence over signalled, one dot per pair,
with the pooled and by-goal t-intervals and the level groups' bootstrap intervals (Figure 4's caption states them).
Sources, in RUN_OUTPUTS: report-a.json, report-a3.json (tokens, score per game), analysis-a.json, analysis-a3.json
(gen_chars_by_level per game), goal-v2/goal-t.json and goal-v2/goal-level1-exploratory.txt. Output to OUT (default:
this folder). Figures 2 and 3 carry one series a panel, so no legend; figure 4 names its two goal kinds.
"""
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def ranks(xs):
    """Average ranks, ties shared (what Spearman needs)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else HERE
BLUE, LIGHT, INK, MUTED, GRID = "#1c5cab", "#86b6ef", "#222222", "#666666", "#d9d9d9"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    with open(os.path.join(GEN, name), encoding="utf-8") as f:
        return json.load(f)


def two_runs():
    a, b = load("report-a.json")["games"], load("report-a3.json")["games"]
    ids = sorted(set(a) & set(b))
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    for ax, key, label, fmt in ((axes[0], "tokens", "tokens, thousands", lambda v: v / 1000.0),
                                (axes[1], "score", "score", lambda v: v)):
        x = [fmt(a[g][key]) for g in ids]
        y = [fmt(b[g][key]) for g in ids]
        rho = spearman(x, y)
        lo, hi = min(x + y), max(x + y)
        ax.plot([lo, hi], [lo, hi], color=GRID, lw=1, zorder=1)
        ax.scatter(x, y, s=28, color=BLUE, edgecolors="white", linewidths=1, zorder=2)
        ax.set_xlabel("first run, " + label)
        ax.set_ylabel("repeat, " + label)
        ax.text(0.04, 0.95, "Spearman %.2f" % rho, transform=ax.transAxes, va="top", color=INK)
        ax.grid(color=GRID, lw=0.5)
        print("%s rho %.3f over %d games" % (key, rho, len(ids)))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "two-runs.png"), dpi=200)


def chars_by_level():
    runs = [load("analysis-a.json")["per_game"], load("analysis-a3.json")["per_game"]]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    per_level = {k: [] for k in range(1, 7)}
    for per_game in runs:
        for row in per_game:
            ys = []
            for k in range(1, 7):
                v = row["gen_chars_by_level"].get(str(k))
                ys.append(v if v else None)
                if v:
                    per_level[k].append(v)
            ax.plot(range(1, 7), ys, color=LIGHT, lw=0.8, alpha=0.6, zorder=1)
    med = [statistics.median(per_level[k]) for k in range(1, 7)]
    mean = [statistics.mean(per_level[k]) for k in range(1, 7)]
    ax.plot(range(1, 7), med, color=BLUE, lw=2, marker="o", markersize=5, markeredgecolor="white", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("level")
    ax.set_ylabel("generated characters (log)")
    ax.set_xticks(range(1, 7))
    ax.grid(color=GRID, lw=0.5, which="major")
    ax.text(0.98, 0.95, "median in front; %d game-runs behind" % sum(len(p) for p in runs), transform=ax.transAxes,
            ha="right", va="top", color=INK)
    print("median by level", [int(m) for m in med])
    print("mean by level", [int(m) for m in mean])
    # the numbers §4.1 states are the figure's own (2026-11-06: the text had carried another aggregate)
    with open(os.path.join(OUT, "chars-by-level.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"game_runs": sum(len(p) for p in runs), "n_by_level": [len(per_level[k]) for k in range(1, 7)],
                   "median_by_level": [int(m) for m in med], "mean_by_level": [int(m) for m in mean]}, f, indent=1)
        f.write("\n")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chars-by-level.png"), dpi=200)


def goal_pairs():
    """Figure 4, goal-pairs.png: cost per completed level, consequence over signalled, one dot per pair, with the
    pooled ratio and the by-goal ratios as bars carrying their t-intervals (goal-v2/goal-t.json, 2026-11-06)."""
    with open(os.path.join(GEN, "goal-v2", "goal-t.json"), encoding="utf-8") as f:
        d = json.load(f)
    pairs = sorted(d["per_pair"].items(), key=lambda kv: (kv[1]["goal"], kv[0]))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axvline(1.0, color=MUTED, lw=1, zorder=1)
    ax.axvline(1.15, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(1.152, 1.02, "predicted 1.15", transform=ax.get_xaxis_transform(), color=MUTED, fontsize=8, va="bottom")
    colours = {"keys": BLUE, "plates": "#c8551b"}
    markers = {"keys": "o", "plates": "s"}
    for goal in ("keys", "plates"):
        xs = [v["ratio"] for _, v in pairs if v["goal"] == goal]
        ax.scatter(xs, [5.0] * len(xs), s=34, color=colours[goal], marker=markers[goal], edgecolors="white",
                   linewidths=1, zorder=3, label="%s pairs" % goal)
    # the level groups' bootstrap readings, from the exploratory print (goal-level1-exploratory.txt)
    import re
    with open(os.path.join(GEN, "goal-v2", "goal-level1-exploratory.txt"), encoding="utf-8") as f:
        levels = dict(re.findall(r"EXPLORATORY (level 1|levels 2-6)\s+consequence/signalled ([\d.]+ \[[\d.]+, [\d.]+\])", f.read()))
    def parse(s):
        m = re.match(r"([\d.]+) \[([\d.]+), ([\d.]+)\]", s)
        return float(m.group(1)), [float(m.group(2)), float(m.group(3))]
    l1, l26 = parse(levels["level 1"]), parse(levels["levels 2-6"])
    rows = [("all ten pairs, t-interval", d["ratio"], d["t_ci"], INK, 4.0),
            ("keys pairs, t-interval", d["by_goal"]["keys"]["ratio"], d["by_goal"]["keys"]["t_ci"], colours["keys"], 3.0),
            ("plates pairs, t-interval", d["by_goal"]["plates"]["ratio"], d["by_goal"]["plates"]["t_ci"], colours["plates"], 2.0),
            ("level 1, bootstrap", l1[0], l1[1], MUTED, 1.0),
            ("levels 2 to 6, bootstrap", l26[0], l26[1], MUTED, 0.0)]
    for label, r, ci, colour, y in rows:
        ax.plot(ci, [y, y], color=colour, lw=2, zorder=2)
        ax.scatter([r], [y], s=40, color=colour, edgecolors="white", linewidths=1, zorder=3)
    ax.set_yticks([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    ax.set_yticklabels(["each pair"] + [row[0] for row in rows])
    ax.set_xlabel("cost per completed level, consequence over signalled")
    ax.set_ylim(-0.6, 5.6)
    ax.grid(color=GRID, lw=0.5, axis="x")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    for label, r, ci, _, _ in rows:
        print("%s %.3f [%.3f, %.3f]" % (label, r, ci[0], ci[1]))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "goal-pairs.png"), dpi=200)


if __name__ == "__main__":
    two_runs()
    chars_by_level()
    goal_pairs()
    for n in ("two-runs.png", "chars-by-level.png", "goal-pairs.png"):
        print(n, os.path.getsize(os.path.join(OUT, n)) // 1024, "KB")
