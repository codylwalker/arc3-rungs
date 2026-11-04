"""Generated characters per COMPLETED level, paired by (game, level), treatment against base; game-cluster bootstrap.

python -I analysis/paired_level_cost.py --base DIR [DIR ...] --treat DIR [DIR ...] [--label X]

A run-level "chars per completed level" mixes the level difficulty a run happened to reach. Here each cell is
(game, level k): the mean log chars spent on level k by every game-pass that completed k, in each group. The
effect is the mean over games of (the mean over that game's shared cells of the log difference), exp'd: the
cost ratio treat/base on the same levels. 95% CI from 4,000 resamples of games with replacement. Levels: the
mean per game-pass, per game, treat minus base, same bootstrap. Placebo: two identical base runs should give
a ratio near 1.
"""
import argparse
import math
import os
import random
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "entry", "tools"))
import analyze_run  # noqa: E402


def load(dirs):
    cells = {}   # (game, k) -> [log chars]
    levels = {}  # game -> [levels per pass]
    for d in dirs:
        out = analyze_run.summarise(analyze_run.analyse(d))
        for r in out["per_game"]:
            g = r["game"]
            L = int(r["levels"] or 0)
            levels.setdefault(g, []).append(L)
            for k, n in r["gen_chars_by_level"].items():
                try:
                    k = int(k)
                except ValueError:
                    continue
                if k <= L and n > 0:
                    cells.setdefault((g, k), []).append(math.log(n))
    return cells, levels


def per_game_diffs(base, treat):
    by_game = {}
    for (g, k), xs in treat.items():
        ys = base.get((g, k))
        if ys:
            by_game.setdefault(g, []).append(statistics.mean(xs) - statistics.mean(ys))
    return {g: statistics.mean(v) for g, v in by_game.items()}, sum(len(v) for v in by_game.values())


def boot(values, reps=4000, seed=7):
    rnd = random.Random(seed)
    vals = list(values)
    stats = []
    for _ in range(reps):
        sample = [vals[rnd.randrange(len(vals))] for _ in vals]
        stats.append(statistics.mean(sample))
    stats.sort()
    return statistics.mean(vals), stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--treat", nargs="+", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    base_cells, base_levels = load(args.base)
    treat_cells, treat_levels = load(args.treat)
    diffs, n_cells = per_game_diffs(base_cells, treat_cells)
    m, lo, hi = boot(diffs.values())
    print("== %s" % args.label)
    print("cost per completed level, treat/base on shared (game, level) cells: %+.1f%%  95%% CI [%+.1f%%, %+.1f%%]  "
          "(%d games, %d cells; treat cheaper in %d of %d games)" % (
              100 * (math.exp(m) - 1), 100 * (math.exp(lo) - 1), 100 * (math.exp(hi) - 1), len(diffs), n_cells,
              sum(1 for v in diffs.values() if v < 0), len(diffs)))
    games = sorted(set(base_levels) & set(treat_levels))
    lv = [statistics.mean(treat_levels[g]) - statistics.mean(base_levels[g]) for g in games]
    m2, lo2, hi2 = boot(lv)
    print("levels per game-pass, treat - base: %+.2f  95%% CI [%+.2f, %+.2f]  (%d games; treat ahead in %d, behind in %d)" % (
        m2, lo2, hi2, len(games), sum(1 for v in lv if v > 0), sum(1 for v in lv if v < 0)))


if __name__ == "__main__":
    main()
