"""Test-retest between two arms' harness reports: per-game agreement of score, levels, act/opt, tokens.

Usage: python -I tools/arm_agreement.py REPORT_A.json REPORT_B.json
The reports are gen/harness_report.py --json outputs. Two arms that differ in little (2026-10-28: the
twin tools on and off, the tools acting in a few games) read as a retest: how much one run says about
one game.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_predictions import spearman  # noqa: E402

FIELDS = ("score", "levels", "actions_over_optimal", "tokens", "tokens_per_level_solved")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)["games"]


def agreement(a, b):
    ids = sorted(set(a) & set(b))
    out = {"games": len(ids), "fields": {}}
    for field in FIELDS:
        pairs = [(a[i].get(field), b[i].get(field)) for i in ids]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        out["fields"][field] = {"rho": spearman([x for x, _ in pairs], [y for _, y in pairs]),
                                "n": len(pairs), "identical": sum(1 for x, y in pairs if x == y)}
    won = [i for i in ids if a[i]["levels"] == a[i]["total_levels"] and b[i]["levels"] == b[i]["total_levels"]]
    ra = [a[i]["actions_over_optimal"] for i in won]
    rb = [b[i]["actions_over_optimal"] for i in won]
    out["won_in_both"] = {
        "games": len(won),
        "score_rho": spearman([a[i]["score"] for i in won], [b[i]["score"] for i in won]),
        "both_100": sum(1 for i in won if a[i]["score"] == 100 and b[i]["score"] == 100),
        "ratio_rho": spearman(ra, rb),
        "ratio_mean_a": sum(ra) / len(ra) if ra else None,
        "ratio_mean_b": sum(rb) / len(rb) if rb else None}
    out["unwon_a"] = sorted(i for i in ids if a[i]["levels"] < a[i]["total_levels"])
    out["unwon_b"] = sorted(i for i in ids if b[i]["levels"] < b[i]["total_levels"])
    out["unwon_both"] = sorted(set(out["unwon_a"]) & set(out["unwon_b"]))
    return out


def fmt(x, nd=2):
    return "n/a" if x is None else "%.*f" % (nd, x)


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    r = agreement(a, b)
    print("games %d" % r["games"])
    for field, v in r["fields"].items():
        print("%-24s Spearman a vs b %s over %d games | identical %d" % (field, fmt(v["rho"]), v["n"], v["identical"]))
    w = r["won_in_both"]
    print("won in both arms: %d games | score Spearman %s | both 100: %d" % (w["games"], fmt(w["score_rho"]), w["both_100"]))
    print("won in both arms: act/opt Spearman %s | mean a %s b %s" % (
        fmt(w["ratio_rho"]), fmt(w["ratio_mean_a"]), fmt(w["ratio_mean_b"])))
    print("unwon a %s | unwon b %s | in both %s" % (r["unwon_a"], r["unwon_b"], r["unwon_both"] or "none"))
    for i in r["unwon_a"] + r["unwon_b"]:
        print("  %s %s | levels a %s b %s | tokens/level a %s b %s" % (
            i, ",".join(a[i]["mechanics"]), a[i]["levels"], b[i]["levels"], a[i]["tokens_per_level_solved"],
            b[i]["tokens_per_level_solved"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
