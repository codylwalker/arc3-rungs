"""The generated games' mean score with every baseline at the certified optimum instead of 1.5 times it, under the
competition toolkit's own scoring (arc_agi 0.9.9, scorecard.py: a completed level scores min(115, (baseline /
actions)^2 * 100), an uncompleted one 0; a game is the level-index-weighted mean, capped at 100 times the weight
share of its completed levels). Added 2026-11-06, so that the number rests on a scorer shown to reproduce the
competition's own: the script first checks its scorer against every game's recorded score at the real baselines
(ceil(1.5 x optimum)), and reports the treatment of levels never reached that reproduces them.
Usage: python -I analysis/score_at_optimum.py report-a.json [report-a3.json ...] [--json score-at-optimum.json]
"""
import json
import math
import sys


def ranks(v):
    """Ranks from 1, ties given the mean of the ranks they span."""
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = math.sqrt(sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry))
    return num / den if den else None


def level_score(baseline, actions):
    return min(115.0, (baseline / actions) ** 2 * 100) if actions > 0 else 0.0


def game_score(r, baseline_of, unreached_count):
    done = int(r["levels"])
    total = int(r["total_levels"])
    scores, weights = [], []
    for i in range(total):
        idx = i + 1
        if i < done:
            scores.append(level_score(baseline_of(r["optimal"][i]), r["actions_per_level"][i]))
            weights.append(idx)
        elif i == done or unreached_count:
            scores.append(0.0)
            weights.append(idx)
    if not weights:
        return 0.0
    s = sum(a * w for a, w in zip(scores, weights)) / sum(weights)
    cap = sum(w for a, w in zip(scores, weights) if a > 0) / sum(weights) * 100
    return min(s, cap)


def main():
    out_path = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    paths = [p for p in sys.argv[1:] if p.endswith(".json") and p != out_path]
    res = {}
    per_game = []
    for p in paths:
        games = json.load(open(p, encoding="utf-8"))["games"]
        real = lambda o: math.ceil(1.5 * o)
        best = None
        for unreached in (True, False):
            diffs = [abs(game_score(r, real, unreached) - r["score"]) for r in games.values()]
            hit = sum(1 for d in diffs if d < 0.01)
            if best is None or hit > best[1]:
                best = (unreached, hit, max(diffs))
        unreached, hit, worst = best
        mean_real = sum(game_score(r, real, unreached) for r in games.values()) / len(games)
        mean_recorded = sum(r["score"] for r in games.values()) / len(games)
        mean_opt = sum(game_score(r, lambda o: o, unreached) for r in games.values()) / len(games)
        name = p.replace("\\", "/").split("/")[-1][:-len(".json")]
        per_game.append((name, {g: (r["score"], game_score(r, lambda o: o, unreached)) for g, r in games.items()}))
        res[name] = {"games": len(games), "reproduced": hit, "worst_difference": worst,
                     "unreached_levels_count": unreached, "mean_recorded": mean_recorded,
                     "mean_rescored_real_baseline": mean_real, "mean_at_optimum": mean_opt}
        print("%s: the scorer reproduces %d of %d recorded game scores (worst difference %.4f; levels never reached "
              "%s); mean recorded %.2f, rescored %.2f; with every baseline at the optimum %.2f"
              % (name, hit, len(games), worst, "count as 0" if unreached else "left out", mean_recorded, mean_real,
                 mean_opt))
    if len(per_game) >= 2:
        # does the score's ceiling explain why two runs agree less on score than on tokens? Rescored with every
        # baseline at the optimum, few games can reach 100; compare the two runs' agreement (added 2026-11-06 ~08:40)
        (n1, s1), (n2, s2) = per_game[:2]
        ids = sorted(set(s1) & set(s2))
        res["agreement"] = {
            "runs": [n1, n2], "games": len(ids),
            "recorded_scores": spearman([s1[g][0] for g in ids], [s2[g][0] for g in ids]),
            "scores_at_optimum": spearman([s1[g][1] for g in ids], [s2[g][1] for g in ids]),
            "games_at_100_recorded": [sum(1 for g in ids if s[g][0] >= 100 - 1e-9) for s in (s1, s2)],
            "games_at_100_at_optimum": [sum(1 for g in ids if s[g][1] >= 100 - 1e-9) for s in (s1, s2)]}
        a = res["agreement"]
        print("agreement %s-%s over %d games: recorded scores %.3f (at 100: %s); with every baseline at the optimum "
              "%.3f (at 100: %s)" % (n1, n2, len(ids), a["recorded_scores"], a["games_at_100_recorded"],
                                     a["scores_at_optimum"], a["games_at_100_at_optimum"]))
    if out_path:
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
        print("wrote " + out_path)


main()
