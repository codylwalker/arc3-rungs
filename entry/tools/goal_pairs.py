"""Score the goal-rung pairs against their pre-registration (written 2026-10-30 16:45 UTC, before the set was generated; P-G1..P-G5).

Usage: python -I tools/goal_pairs.py <run output dir> --set <goal set dir> [--json out.json] [--reps 4000] [--seed 11]

The set (arc3/gen/out/goal-set-a, from `corpus.py --goal K --pair`) pairs games: each certified
plan is emitted twice, `signalled` (the goal on the HUD) and `consequence` (no goal shown), under two ids sharing a
`pair_seed`. The run plays every game several times (game-passes). The unit is the PAIR; intervals are bootstraps over
pairs. The reading rules are the pre-registration's, verbatim:

  P-G1 levels: mean over pairs of (signalled - consequence) levels per game-pass. HIT if the interval excludes 0 and
       the mean is >= 0.5; HALF if it excludes 0 below 0.5; MISS if it includes 0; REVERSED if it excludes 0 below.
  P-G2 cost: on levels both variants of a pair complete, the geometric mean ratio of generated characters per
       completed level, consequence / signalled. HIT if the interval excludes 1 and the ratio is >= 1.15; HALF if it
       excludes 1 below 1.15; MISS if it includes 1 ("not shown" when the ratio is inside 0.80-1.20); REVERSED if
       the interval lies below 1.
  P-G3 level 1: HIT if the paired completion difference on level 1 exceeds the mean paired difference on levels 2-6.
  P-G4 unfinished: HIT if the consequence variant has more unfinished game-runs and a one-sided sign test over the
       pairs that differ gives p < 0.10; HALF if the direction holds without it; MISS otherwise.
  P-G5 the control: the signalled variant clears >= 85% of its levels. If not, P-G1..P-G4 are UNINFORMATIVE.
"""
import argparse
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_run  # noqa: E402

LEVELS = 6


def load_pairs(set_dir):
    """{id4: (pair_seed, variant, goal)} from the set's manifest (or a dataset's games.json)."""
    for name in ("manifest.json", "games.json"):
        p = os.path.join(set_dir, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                games = json.load(f)["games"]
            return {g["game_id"].split("-")[0]: (g["pair_seed"], g["variant"], g.get("goal"))
                    for g in games if g.get("variant")}
    raise SystemExit("no manifest.json or games.json under %s" % set_dir)


def rows_by_pair(run_dir, pairs):
    """{pair_seed: {"signalled": [row, ...], "consequence": [...], "goal": g}} from analyze_run's per-game rows."""
    out = {}
    for r in analyze_run.summarise(analyze_run.analyse(run_dir))["per_game"]:
        id4 = str(r["game"]).split("-")[0]
        if id4 not in pairs:
            continue
        seed, variant, goal = pairs[id4]
        d = out.setdefault(seed, {"signalled": [], "consequence": [], "goal": goal})
        d[variant].append(r)
    return out


def boot(values, stat=statistics.mean, reps=4000, seed=11):
    vals = list(values)
    if not vals:
        return None, None, None
    rnd = random.Random(seed)
    sims = sorted(stat([vals[rnd.randrange(len(vals))] for _ in vals]) for _ in range(reps))
    return stat(vals), sims[int(0.025 * reps)], sims[int(0.975 * reps) - 1]


def completed_chars(row):
    """{level k: chars} for the levels this game-pass completed (1..levels)."""
    out = {}
    for k, n in row["gen_chars_by_level"].items():
        try:
            k = int(k)
        except ValueError:
            continue
        if k <= int(row["levels"] or 0) and n > 0:
            out[k] = n
    return out


def sign_p(k, n):
    """One-sided P(X >= k) for X ~ Binomial(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else 1.0


def score(by_pair, reps=4000, seed=11):
    pairs = sorted(s for s, d in by_pair.items() if d["signalled"] and d["consequence"])
    lv_diff, cost_lr, d1, dlater, unf = [], [], [], [], []
    s_levels = s_runs = 0
    for s in pairs:
        S, C = by_pair[s]["signalled"], by_pair[s]["consequence"]
        ls, lc = [int(r["levels"] or 0) for r in S], [int(r["levels"] or 0) for r in C]
        lv_diff.append(statistics.mean(ls) - statistics.mean(lc))
        s_levels += sum(ls)
        s_runs += len(ls)
        # cost on levels both variants complete: mean log chars per level per variant, then C - S
        cs = [completed_chars(r) for r in S]
        cc = [completed_chars(r) for r in C]
        shared = sorted(set().union(*cs) & set().union(*cc)) if cs and cc else []
        diffs = []
        for k in shared:
            a = [math.log(x[k]) for x in cc if k in x]
            b = [math.log(x[k]) for x in cs if k in x]
            diffs.append(statistics.mean(a) - statistics.mean(b))
        if diffs:
            cost_lr.append(statistics.mean(diffs))
        rate = lambda rows, k: sum(1 for r in rows if int(r["levels"] or 0) >= k) / len(rows)
        d1.append(rate(S, 1) - rate(C, 1))
        dlater.append(statistics.mean(rate(S, k) - rate(C, k) for k in range(2, LEVELS + 1)))
        unf.append((sum(1 for x in ls if x < LEVELS), sum(1 for x in lc if x < LEVELS)))
    res = {"pairs": len(pairs), "game_passes": {"signalled": s_runs,
                                                "consequence": sum(len(by_pair[s]["consequence"]) for s in pairs)}}
    # P-G5 first: it decides whether the others are read at all
    clear = s_levels / (LEVELS * s_runs) if s_runs else 0.0
    res["P-G5"] = {"signalled_clear_rate": round(clear, 3), "verdict": "OK" if clear >= 0.85 else "FAILED"}
    informative = clear >= 0.85
    m, lo, hi = boot(lv_diff, reps=reps, seed=seed)
    if m is None:
        v = "NO DATA"
    elif lo > 0:
        v = "HIT" if m >= 0.5 else "HALF"
    elif hi < 0:
        v = "REVERSED"
    else:
        v = "MISS"
    res["P-G1"] = {"mean_levels_diff": m, "ci": [lo, hi], "verdict": v}
    m, lo, hi = boot(cost_lr, reps=reps, seed=seed)
    if m is None:
        res["P-G2"] = {"verdict": "NO DATA", "pairs_with_shared_levels": 0}
    else:
        r, rlo, rhi = math.exp(m), math.exp(lo), math.exp(hi)
        if rlo > 1:
            v = "HIT" if r >= 1.15 else "HALF"
        elif rhi < 1:
            v = "REVERSED"
        else:
            v = "MISS (not shown)" if 0.80 <= r <= 1.20 else "MISS"
        res["P-G2"] = {"ratio_consequence_over_signalled": r, "ci": [rlo, rhi], "verdict": v,
                       "pairs_with_shared_levels": len(cost_lr)}
    if d1:
        a, b = statistics.mean(d1), statistics.mean(dlater)
        res["P-G3"] = {"level1_diff": a, "levels2_6_diff": b, "verdict": "HIT" if a > b else "MISS"}
    else:
        res["P-G3"] = {"verdict": "NO DATA"}
    us, uc = sum(x for x, _ in unf), sum(y for _, y in unf)
    differ = [(x, y) for x, y in unf if x != y]
    k = sum(1 for x, y in differ if y > x)
    p = sign_p(k, len(differ))
    if uc > us:
        v = "HIT" if p < 0.10 else "HALF"
    else:
        v = "MISS"
    res["P-G4"] = {"unfinished_signalled": us, "unfinished_consequence": uc, "pairs_differing": len(differ),
                   "consequence_more": k, "sign_p": p, "verdict": v}
    if not informative:
        for key in ("P-G1", "P-G2", "P-G3", "P-G4"):
            res[key]["verdict"] = "UNINFORMATIVE (P-G5 failed; was %s)" % res[key]["verdict"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--set", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--by-goal", action="store_true",
                    help="also print the same readings per goal kind, labelled EXPLORATORY (the pre-registration's "
                         "'keys against plates' is not scored)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    by_pair = rows_by_pair(a.run, load_pairs(a.set))
    res = score(by_pair, a.reps, a.seed)
    fmt = lambda x: round(x, 4) if isinstance(x, float) else x  # noqa: E731
    print("pairs %d; game-passes %s" % (res["pairs"], res["game_passes"]))
    for key in ("P-G5", "P-G1", "P-G2", "P-G3", "P-G4"):
        print("%-5s %s" % (key, json.dumps(res[key], default=fmt)))
    if a.by_goal:
        res["exploratory_by_goal"] = {}
        for goal in sorted({d.get("goal") for d in by_pair.values() if d.get("goal")}):
            sub = score({s: d for s, d in by_pair.items() if d.get("goal") == goal}, a.reps, a.seed)
            res["exploratory_by_goal"][goal] = sub
            print("EXPLORATORY, not scored: goal %s, %d pairs" % (goal, sub["pairs"]))
            for key in ("P-G5", "P-G1", "P-G2", "P-G3", "P-G4"):
                print("    %-5s %s" % (key, json.dumps(sub[key], default=fmt)))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
