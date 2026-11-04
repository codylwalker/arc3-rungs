"""Three runs of the D' harness on the 41 generated games (a = v2, a3 = v3 repeat, b = tools on):
which games fail, whether failures repeat, whether one run's tokens predict another run's failures,
and (added 2026-11-05) the ceiling split of the two identical runs: their tokens and scores agree by
rank (Spearman, ties averaged) over all games, over the games both runs scored 100, and over the
games off that ceiling; and the chance that two runs share no unfinished game if failures fell on
games at random. Since 2026-11-06 also: intervals on the games both runs scored under 100, and each identical
run's optimality on the levels it solved (the share at the certified optimum, the median moves over it).
Usage: python -I analysis/retest_stats.py report-a.json report-a3.json report-b.json [--json ceiling-split.json]
"""
import json
import math
import sys
from itertools import combinations


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)["games"]


def fisher_greater(a, b, c, d):
    """One-sided Fisher p for the 2x2 [[a, b], [c, d]]: P(X >= a), rows fixed."""
    n1, n2, k = a + b, c + d, a + c
    tot = math.comb(n1 + n2, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(n1, k) + 1)) / tot


def auc(scores, positives):
    pos = [s for g, s in scores.items() if g in positives]
    neg = [s for g, s in scores.items() if g not in positives]
    if not pos or not neg:
        return None
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


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


def ceiling_split(runs, games, unwon, x="a", y="a3"):
    """Agreement of two identical runs on and off the score's ceiling of 100."""
    top = [g for g in games if runs[x][g]["score"] == 100 and runs[y][g]["score"] == 100]
    off = [g for g in games if g not in top]

    def rho(sub, key):
        return spearman([runs[x][g][key] for g in sub], [runs[y][g][key] for g in sub])

    n, kx, ky = len(games), len(unwon[x]), len(unwon[y])
    under = [g for g in games if runs[x][g]["score"] < 100 and runs[y][g]["score"] < 100]

    # Intervals (added 2026-11-06, after a cold read): a game bootstrap, 4,000 resamples, seed 11. It carries the
    # spread across games only; run-level variance (throughput, scheduling) is the placebo's to show.
    import random
    rng = random.Random(11)

    def boot(sub, f, reps=4000):
        vals = []
        for _ in range(reps):
            s = [rng.choice(sub) for _ in sub]
            v = f(s)
            if v is not None:
                vals.append(v)
        vals.sort()
        return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]

    def diff(s):
        t, c = rho(s, "tokens"), rho(s, "score")
        return None if t is None or c is None else t - c

    def size(g):
        """A game's size: the sum of its levels' certified optima."""
        return sum(runs[x][g]["optimal"])

    def partial_size(sub):
        """The two runs' token Spearman with each game's size partialled out (ranks, first-order partial)."""
        tx, ty = [runs[x][g]["tokens"] for g in sub], [runs[y][g]["tokens"] for g in sub]
        z = [size(g) for g in sub]
        rxy, rxz, ryz = spearman(tx, ty), spearman(tx, z), spearman(ty, z)
        if None in (rxy, rxz, ryz) or abs(rxz) >= 1 or abs(ryz) >= 1:
            return None
        return (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))

    intervals = {
        "all_tokens": boot(games, lambda s: rho(s, "tokens")),
        "all_score": boot(games, lambda s: rho(s, "score")),
        "all_tokens_minus_score": boot(games, diff),
        "both_100_tokens": boot(top, lambda s: rho(s, "tokens")),
        # added 2026-11-06 (a second cold read), drawn after the four above so their draws are unchanged
        "both_under_100_tokens": boot(under, lambda s: rho(s, "tokens")),
        "both_under_100_score": boot(under, lambda s: rho(s, "score")),
        # added 2026-11-06 ~08:40 (a third cold read: is it only game size?), drawn after all the above
        "all_tokens_partial_on_size": boot(games, lambda s: partial_size(s)),
        "both_100_tokens_partial_on_size": boot(top, lambda s: partial_size(s)),
    }

    def rho_size(sub):
        return spearman([runs[x][g]["tokens"] for g in sub], [size(g) for g in sub])

    def optimality(r):
        """Over the levels a run solved: the share played at the certified optimum, and the median moves/optimum."""
        rat = [mv / opt for g in games
               for mv, opt in list(zip(r[g]["actions_per_level"], r[g]["optimal"]))[:int(r[g]["levels"])]]
        rat.sort()
        mid = len(rat) // 2
        med = rat[mid] if len(rat) % 2 else (rat[mid - 1] + rat[mid]) / 2
        return {"levels": len(rat), "at_optimum": sum(1 for v in rat if v == 1) / len(rat), "median_ratio": med,
                "below_optimum": sum(1 for v in rat if v < 1)}
    out = {
        "runs": [x, y],
        "at_100": {x: sum(1 for g in games if runs[x][g]["score"] == 100),
                   y: sum(1 for g in games if runs[y][g]["score"] == 100)},
        "all": {"n": n, "tokens": rho(games, "tokens"), "score": rho(games, "score"),
                "tokens_minus_score": rho(games, "tokens") - rho(games, "score")},
        "both_100": {"n": len(top), "tokens": rho(top, "tokens")},
        "off_ceiling": {"n": len(off), "tokens": rho(off, "tokens"), "score": rho(off, "score"),
                        "note": "selected on the score (a game enters if either run is under 100): biased against "
                                "the score; see both_under_100"},
        "both_under_100": {"n": len(under), "tokens": rho(under, "tokens"), "score": rho(under, "score")},
        "intervals_95_game_bootstrap": intervals,
        "unfinished": {x: kx, y: ky, "shared": len(unwon[x] & unwon[y]),
                       "p_share_none_at_random": math.comb(n - kx, ky) / math.comb(n, ky)},
        "optimality": {x: optimality(runs[x]), y: optimality(runs[y])},
        "size": {"all": {"tokens_vs_size": rho_size(games), "partial_on_size": partial_size(games)},
                 "both_100": {"tokens_vs_size": rho_size(top), "partial_on_size": partial_size(top)}},
    }
    print("ceiling split %s-%s: at 100 %s | all %d games: tokens %.3f, score %.3f | both at 100 (%d): tokens %.3f | "
          "off the ceiling (%d): tokens %.3f, score %.3f" % (
              x, y, out["at_100"], n, out["all"]["tokens"], out["all"]["score"], len(top),
              out["both_100"]["tokens"], len(off), out["off_ceiling"]["tokens"], out["off_ceiling"]["score"]))
    iv = out["intervals_95_game_bootstrap"]
    print("95%% game bootstrap: tokens %.3f [%.2f, %.2f] | score %.3f [%.2f, %.2f] | tokens - score %.3f [%.2f, %.2f] | "
          "both at 100, tokens %.3f [%.2f, %.2f] | both under 100 (%d): tokens %.3f, score %.3f" % (
              out["all"]["tokens"], iv["all_tokens"][0], iv["all_tokens"][1], out["all"]["score"], iv["all_score"][0],
              iv["all_score"][1], out["all"]["tokens"] - out["all"]["score"], iv["all_tokens_minus_score"][0],
              iv["all_tokens_minus_score"][1], out["both_100"]["tokens"], iv["both_100_tokens"][0],
              iv["both_100_tokens"][1], len(under), out["both_under_100"]["tokens"], out["both_under_100"]["score"]))
    print("unfinished %s %d, %s %d, shared %d; chance of sharing none at random %.3f" % (
        x, kx, y, ky, out["unfinished"]["shared"], out["unfinished"]["p_share_none_at_random"]))
    print("both under 100 (%d), 95%% game bootstrap: tokens [%.2f, %.2f], score [%.2f, %.2f]" % (
        len(under), iv["both_under_100_tokens"][0], iv["both_under_100_tokens"][1], iv["both_under_100_score"][0],
        iv["both_under_100_score"][1]))
    print("size (sum of certified optima): tokens vs size %.3f; tokens partialled on size %.3f [%.2f, %.2f] | both at "
          "100: %.3f, partialled %.3f [%.2f, %.2f]" % (
              out["size"]["all"]["tokens_vs_size"], out["size"]["all"]["partial_on_size"],
              iv["all_tokens_partial_on_size"][0], iv["all_tokens_partial_on_size"][1],
              out["size"]["both_100"]["tokens_vs_size"], out["size"]["both_100"]["partial_on_size"],
              iv["both_100_tokens_partial_on_size"][0], iv["both_100_tokens_partial_on_size"][1]))
    for k in (x, y):
        o = out["optimality"][k]
        print("optimality %s: %d solved levels, %.3f at the certified optimum, median %.3f times it, %d below it" % (
            k, o["levels"], o["at_optimum"], o["median_ratio"], o["below_optimum"]))
    return out


def main():
    names = ["a", "a3", "b"]
    runs = dict(zip(names, (load(p) for p in sys.argv[1:4])))
    games = sorted(runs["a"])
    n = len(games)
    unwon = {k: {g for g in games if r[g]["levels"] < r[g]["total_levels"]} for k, r in runs.items()}
    for k in names:
        print("unwon %-2s %d: %s" % (k, len(unwon[k]), sorted(x[:4] for x in unwon[k])))
    allfail = [g for k in names for g in unwon[k]]
    distinct = set(allfail)
    print("failed game-runs %d in %d distinct games; repeated: %s; in all three: %s" % (
        len(allfail), len(distinct), sorted(g[:4] for g in distinct if allfail.count(g) > 1),
        sorted(g[:4] for g in distinct if allfail.count(g) == 3) or "none"))
    exp = 0.0
    for x, y in combinations(names, 2):
        e = len(unwon[x]) * len(unwon[y]) / n
        exp += e
        obs = len(unwon[x] & unwon[y])
        # hypergeometric P(overlap >= obs) with y's set drawn at random
        kx, ky = len(unwon[x]), len(unwon[y])
        p = sum(math.comb(kx, i) * math.comb(n - kx, ky - i) for i in range(obs, min(kx, ky) + 1)) / math.comb(n, ky)
        print("overlap %s-%s observed %d expected at random %.2f  P(>=obs) %.2f" % (x, y, obs, e, p))
    print("pairwise overlaps total: observed %d, expected if failures fell on games at random %.2f" % (
        sum(len(unwon[x] & unwon[y]) for x, y in combinations(names, 2)), exp))

    for mech in ("switch", "click_move", "pushable", "keydoor", "snake", "teleporter", "wrap", "hazard"):
        mg = [g for g in games if mech in runs["a"][g]["mechanics"]]
        og = [g for g in games if g not in mg]
        fm = sum(g in unwon[k] for k in names for g in mg)
        fo = sum(g in unwon[k] for k in names for g in og)
        per = {k: sum(g in unwon[k] for g in mg) for k in names}
        p = fisher_greater(fm, 3 * len(mg) - fm, fo, 3 * len(og) - fo)
        print("%-10s games %2d | failed %d of %d game-runs vs others %d of %d | per run %s | one-sided p %.3f" % (
            mech, len(mg), fm, 3 * len(mg), fo, 3 * len(og), per, p))

    # the confirmatory reads: each pattern on the runs after it was noticed (switch in a, click_move in b)
    for mech, later in (("switch", ("a3", "b")), ("click_move", ("a3",))):
        mg = [g for g in games if mech in runs["a"][g]["mechanics"]]
        og = [g for g in games if g not in mg]
        fm = sum(g in unwon[k] for k in later for g in mg)
        fo = sum(g in unwon[k] for k in later for g in og)
        p = fisher_greater(fm, len(later) * len(mg) - fm, fo, len(later) * len(og) - fo)
        print("%-10s on the runs after it was noticed (%s): %d of %d game-runs against %d of %d | one-sided p %.3f" % (
            mech, "+".join(later), fm, len(later) * len(mg), fo, len(later) * len(og), p))

    print("cross-run: does run Y's token count predict run X's unfinished games? (AUC; 0.5 = no)")
    for x in names:
        for y in names:
            if x == y:
                continue
            t = {g: runs[y][g]["tokens"] for g in games}
            tl = {g: runs[y][g]["tokens_per_level_solved"] for g in games}
            print("  X=%-2s Y=%-2s tokens AUC %.2f | tokens/level AUC %.2f" % (x, y, auc(t, unwon[x]), auc(tl, unwon[x])))
        others = [y for y in names if y != x]
        # predictor restricted to games won in both other runs, so no unwon-game inflation enters it
        both_won = [g for g in games if all(g not in unwon[y] for y in others)]
        t = {g: sum(runs[y][g]["tokens"] for y in others) for g in both_won}
        print("  X=%-2s Y=other two, games won in both (%d): summed tokens AUC %s | X unwon among them %d" % (
            x, len(both_won), "%.2f" % auc(t, unwon[x]) if auc(t, unwon[x]) is not None else "n/a",
            len(unwon[x] & set(both_won))))

    # sign-count artifact: one draw against the mean of two, on a score with a ceiling
    up = sum(1 for g in games if runs["b"][g]["score"] == 100 and (runs["a"][g]["score"] + runs["a3"][g]["score"]) / 2 < 100)
    dn = sum(1 for g in games if runs["b"][g]["score"] < 100 and runs["a"][g]["score"] == 100 and runs["a3"][g]["score"] == 100)
    print("b at 100 while mean(a, a3) < 100: %d games | b < 100 while a and a3 both 100: %d games" % (up, dn))
    at100 = {k: sum(1 for g in games if runs[k][g]["score"] == 100) for k in names}
    both = sum(1 for g in games if runs["a"][g]["score"] == 100 and runs["a3"][g]["score"] == 100)
    print("games at 100: %s | a and a3 both: %d" % (at100, both))

    d = [runs["a3"][g]["score"] - runs["a"][g]["score"] for g in games]
    ad = sorted(abs(x) for x in d)
    print("identical runs a3-a, |score difference| per game: median %.1f, mean %.1f, max %.1f, >10 points in %d games, 0 in %d" % (
        ad[n // 2], sum(ad) / n, ad[-1], sum(1 for x in ad if x > 10), sum(1 for x in ad if x == 0)))

    split = ceiling_split(runs, games, unwon)
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(split, f, indent=1)
            f.write("\n")
        print("wrote " + path)


main()
