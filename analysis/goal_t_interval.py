"""The goal-rung pairs' cost ratio (P-G2) with a t-interval beside its pre-registered bootstrap (added 2026-11-06,
after a cold read: a percentile bootstrap over ten pairs undercovers). The per-pair statistic is goal_pairs.py's own:
on the levels both variants completed, the mean log generated characters per level in each variant, consequence minus
signalled, averaged over those levels. The ratio is exp of the mean over pairs; the t-interval is on n - 1 degrees of
freedom. Before writing, the script checks that its ratio and its bootstrap equal goal_pairs.py's to full precision.
Usage: python -I analysis/goal_t_interval.py <run output dir> --set <goal set dir> [--json goal-t.json]
"""
import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "entry", "tools"))
import goal_pairs as gp  # noqa: E402

# Student t, the 0.975 quantile by degrees of freedom
T975 = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622,
        10: 2.2281, 11: 2.2010, 12: 2.1788, 13: 2.1604, 14: 2.1448, 15: 2.1314, 16: 2.1199, 17: 2.1098, 18: 2.1009,
        19: 2.0930, 20: 2.0860}


def per_pair_log_ratios(by_pair):
    out = {}
    for s in sorted(s for s, d in by_pair.items() if d["signalled"] and d["consequence"]):
        cs = [gp.completed_chars(r) for r in by_pair[s]["signalled"]]
        cc = [gp.completed_chars(r) for r in by_pair[s]["consequence"]]
        shared = sorted(set().union(*cs) & set().union(*cc)) if cs and cc else []
        diffs = [statistics.mean([math.log(x[k]) for x in cc if k in x]) -
                 statistics.mean([math.log(x[k]) for x in cs if k in x]) for k in shared]
        if diffs:
            out[s] = statistics.mean(diffs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--set", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    by_pair = gp.rows_by_pair(a.run, gp.load_pairs(a.set))
    lr = per_pair_log_ratios(by_pair)
    vals = [lr[s] for s in sorted(lr)]
    n = len(vals)
    m, se = statistics.mean(vals), statistics.stdev(vals) / math.sqrt(n)
    t = T975[n - 1]
    ref = gp.score(by_pair)["P-G2"]
    bm, blo, bhi = gp.boot(vals)
    if math.exp(m) != ref["ratio_consequence_over_signalled"] or [math.exp(blo), math.exp(bhi)] != ref["ci"]:
        raise SystemExit("does not reproduce goal_pairs.py's P-G2: %r against %r" % (
            (math.exp(m), math.exp(blo), math.exp(bhi)), ref))
    res = {"pairs": n, "ratio": math.exp(m), "bootstrap_ci": [math.exp(blo), math.exp(bhi)],
           "t_ci": [math.exp(m - t * se), math.exp(m + t * se)], "t_quantile": t,
           "pairs_above_1": sum(1 for v in vals if v > 0),
           "per_pair": {str(s): {"goal": by_pair[s]["goal"], "ratio": math.exp(lr[s])} for s in sorted(lr)}}
    # by goal kind (exploratory, as pre-registered), each on its own t-interval (added 2026-11-06 ~08:40)
    res["by_goal"] = {}
    for kind in sorted({by_pair[s]["goal"] for s in lr}):
        kv = [lr[s] for s in sorted(lr) if by_pair[s]["goal"] == kind]
        km, kse = statistics.mean(kv), statistics.stdev(kv) / math.sqrt(len(kv))
        kt = T975[len(kv) - 1]
        res["by_goal"][kind] = {"pairs": len(kv), "ratio": math.exp(km),
                                "t_ci": [math.exp(km - kt * kse), math.exp(km + kt * kse)], "t_quantile": kt}
        print("  %-6s %d pairs: ratio %.3f, t-interval [%.3f, %.3f]" % (
            kind, len(kv), math.exp(km), res["by_goal"][kind]["t_ci"][0], res["by_goal"][kind]["t_ci"][1]))
    print("P-G2 over %d pairs: ratio %.4f, bootstrap [%.3f, %.3f] (goal_pairs.py's, reproduced), t-interval "
          "[%.3f, %.3f] (t = %.4f on %d df); %d of %d pairs above 1"
          % (n, res["ratio"], res["bootstrap_ci"][0], res["bootstrap_ci"][1], res["t_ci"][0], res["t_ci"][1], t,
             n - 1, res["pairs_above_1"], n))
    for s, d in res["per_pair"].items():
        print("  pair %s %-6s %.3f" % (s, d["goal"], d["ratio"]))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, indent=1)
            f.write("\n")
        print("wrote " + a.json)


main()
