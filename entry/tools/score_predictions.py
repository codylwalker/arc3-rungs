"""Score the generated-games predictions P1-P6 (pre-registered 2026-10-27, late) on a run's report.

Usage: python -I tools/score_predictions.py --report A.json [--analysis A_an.json]
                                            [--b-report B.json --b-analysis B_an.json] [--json out.json]
--report is gen/harness_report.py's --json (per game: mechanics, levels, actions per level, the
oracle's optimal per level, tokens, score, T1's best level); --analysis is tools/analyze_run.py's
--json (per game and level: the characters the model generated, a token proxy, ~3 chars a token).
The b pair is the twin-tools arm, for P5. The predictions, verbatim as written
before the run:
- P1 solve rate falls with mechanic count: >=80% of the levels of 1-mechanic games, <=40% of
  4-mechanic games' levels.
- P2 by mechanic, hazard / ice / oneway / wrap levels are solved more often than keydoor / pushable /
  switch / teleporter / state_flip (the order of the public hard set).
- P3 actions per solved level sit at 1.5-3x optimal, worse on long levels.
- P4 tokens per solved level scale with optimal length, not with the number of mechanics.
- P5 twin tools help where dynamics are contact-shaped (hazard, keydoor, pushable) and not on snake /
  switch; the paired CI on levels includes zero (one pass cannot separate arms).
- P6 the T1 memoryless clear rate per game predicts the harness's first-level clears.
Each verdict prints the rule it applied. Intervals are 95% bootstrap intervals over games (games are
resampled whole, with their levels), seeded, so a rerun prints the same numbers.
"""
import argparse
import json
import random

GROUP_A = ("hazard", "ice", "oneway", "wrap")
GROUP_B = ("keydoor", "pushable", "switch", "teleporter", "state_flip")
CONTACT = ("hazard", "keydoor", "pushable")
NON_CONTACT = ("snake", "switch")


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return None if sxx == 0 or syy == 0 else sxy / (sxx * syy) ** 0.5


def boot(games, stat, n=2000, seed=20261027):
    """95% interval of stat(sample) over samples of whole games; None where the statistic is undefined."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        v = stat([games[rng.randrange(len(games))] for _ in games])
        if v is not None:
            vals.append(v)
    if len(vals) < n // 2:
        return None
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


def load_games(report_path, analysis_path=None):
    with open(report_path, encoding="utf-8") as f:
        rep = json.load(f)["games"]
    chars = {}
    if analysis_path:
        with open(analysis_path, encoding="utf-8") as f:
            for row in json.load(f)["per_game"]:
                chars[row["game"]] = {int(k): v for k, v in row.get("gen_chars_by_level", {}).items() if k.isdigit()}
    games = []
    for gid, g in sorted(rep.items()):
        solved = int(round(g["levels"]))
        apl, opt = g.get("actions_per_level") or [], g.get("optimal") or []
        lv = [{"index": i + 1, "optimal": opt[i], "actions": apl[i],
               "chars": chars.get(gid, {}).get(i + 1)} for i in range(min(solved, len(apl), len(opt)))]
        games.append({"id": gid, "mechanics": g["mechanics"], "n": g["n_mechanics"], "solved": g["levels"],
                      "total": g["total_levels"], "levels": lv, "tokens": g.get("tokens"), "score": g.get("score"),
                      "tok_per_level": g.get("tokens_per_level_solved"), "t1": g.get("t1_best_level")})
    return games


def rate(games, pred):
    sel = [g for g in games if pred(g)]
    tot = sum(g["total"] for g in sel)
    return (sum(g["solved"] for g in sel) / tot) if tot else None


def mech_rates(games, mechs):
    return {m: rate(games, lambda g, m=m: m in g["mechanics"]) for m in mechs}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, nd=2):
    return "n/a" if x is None else ("%.*f" % (nd, x))


def fci(ci, nd=2):
    return "n/a" if ci is None else "[%s, %s]" % (fmt(ci[0], nd), fmt(ci[1], nd))


def p1(games):
    r1, r4 = rate(games, lambda g: g["n"] == 1), rate(games, lambda g: g["n"] == 4)
    by_n = {n: rate(games, lambda g, n=n: g["n"] == n) for n in sorted({g["n"] for g in games})}
    rho = spearman([g["n"] for g in games], [g["solved"] / g["total"] for g in games])
    hit = r1 is not None and r4 is not None and r1 >= 0.80 and r4 <= 0.40
    return {"rate_by_count": by_n, "rho_count_vs_game_rate": rho, "verdict": "HIT" if hit else "MISS",
            "rule": "HIT iff the 1-mechanic level rate >= 0.80 and the 4-mechanic level rate <= 0.40"}


def p2(games):
    ra, rb = mech_rates(games, GROUP_A), mech_rates(games, GROUP_B)

    def diff(gs):
        a, b = mean(mech_rates(gs, GROUP_A).values()), mean(mech_rates(gs, GROUP_B).values())
        return None if a is None or b is None else a - b

    def pooled(gs, group):
        tot = sum(g["total"] * sum(1 for m in g["mechanics"] if m in group) for g in gs)
        return sum(g["solved"] * sum(1 for m in g["mechanics"] if m in group) for g in gs) / tot if tot else None

    d, ci = diff(games), boot(games, diff)
    pairs = [(a, b) for a in GROUP_A for b in GROUP_B if ra[a] is not None and rb[b] is not None]
    wins = sum(1 for a, b in pairs if ra[a] > rb[b])
    ties = sum(1 for a, b in pairs if ra[a] == rb[b])
    if d is not None and ci is not None and d > 0 and ci[0] > 0:
        verdict = "HIT"
    elif d is not None and d > 0:
        verdict = "NOT SHOWN (direction right, interval spans zero)"
    else:
        verdict = "MISS"
    return {"A": ra, "B": rb, "mean_A_minus_mean_B": d, "ci": ci, "pooled_A": pooled(games, GROUP_A),
            "pooled_B": pooled(games, GROUP_B), "pairs_A_above_B": "%d of %d (%d ties)" % (wins, len(pairs), ties),
            "verdict": verdict,
            "rule": "mean of the A mechanics' level rates minus mean of B's; HIT iff > 0 and its interval excludes 0"}


def solved_levels(gs):
    return [lv for g in gs for lv in g["levels"] if lv["optimal"]]


def pooled_ratio(levels):
    o = sum(lv["optimal"] for lv in levels)
    return sum(lv["actions"] for lv in levels) / o if o else None


def p3(games):
    lv = solved_levels(games)
    if not lv:
        return {"pooled_ratio": None, "median_level_ratio": None, "levels": 0, "rho_optimal_vs_ratio": None,
                "ci": None, "by_length": {}, "verdict_band": "UNTESTABLE (no solved level)",
                "verdict_longer": "UNTESTABLE"}
    ratio = pooled_ratio(lv)
    per = sorted(l["actions"] / l["optimal"] for l in lv)
    rho_f = lambda gs: spearman([l["optimal"] for l in solved_levels(gs)],
                                [l["actions"] / l["optimal"] for l in solved_levels(gs)])
    rho, ci = rho_f(games), boot(games, rho_f)
    cuts = sorted(l["optimal"] for l in lv)
    t1, t2 = cuts[len(cuts) // 3], cuts[2 * len(cuts) // 3]
    terc = {"short (<%d)" % t1: pooled_ratio([l for l in lv if l["optimal"] < t1]),
            "middle": pooled_ratio([l for l in lv if t1 <= l["optimal"] < t2]),
            "long (>=%d)" % t2: pooled_ratio([l for l in lv if l["optimal"] >= t2])}
    band = ratio is not None and 1.5 <= ratio <= 3.0
    longer = rho is not None and ci is not None and rho > 0 and ci[0] > 0
    return {"pooled_ratio": ratio, "median_level_ratio": per[len(per) // 2] if per else None, "levels": len(lv),
            "rho_optimal_vs_ratio": rho, "ci": ci, "by_length": terc,
            "verdict_band": "HIT" if band else "MISS", "verdict_longer": "HIT" if longer else "MISS",
            "rule": "band: pooled actions / pooled optimal over solved levels in [1.5, 3]; longer: Spearman "
                    "(optimal, actions/optimal) over solved levels > 0 with its interval above 0"}


def p4(games):
    def within(gs):
        rs = [spearman([l["optimal"] for l in g["levels"]], [l["chars"] for l in g["levels"]])
              for g in gs if len(g["levels"]) >= 3 and all(l["chars"] is not None for l in g["levels"])]
        return mean(rs)

    def by_mech(gs):
        sel = [g for g in gs if g["tok_per_level"]]
        return spearman([g["n"] for g in sel], [g["tok_per_level"] for g in sel])

    def by_opt(gs):
        sel = [g for g in gs if g["tok_per_level"] and g["levels"]]
        return spearman([mean([l["optimal"] for l in g["levels"]]) for g in sel], [g["tok_per_level"] for g in sel])

    have_chars = any(l["chars"] is not None for g in games for l in g["levels"])
    w, wci = (within(games), boot(games, within)) if have_chars else (None, None)
    m, mci = by_mech(games), boot(games, by_mech)
    o, oci = by_opt(games), boot(games, by_opt)
    idx = {}
    for g in games:
        for l in g["levels"]:
            if l["chars"] is not None:
                idx.setdefault(min(l["index"], 4), []).append(l["chars"])
    med = {("%d" % k if k < 4 else "4+"): sorted(v)[len(v) // 2] for k, v in sorted(idx.items())}
    first = [g["levels"][0]["chars"] / mean([l["chars"] for l in g["levels"][1:]]) for g in games
             if len(g["levels"]) >= 2 and all(l["chars"] for l in g["levels"])]
    length_ok = w is not None and wci is not None and w > 0 and wci[0] > 0
    mech_ok = mci is not None and mci[0] <= 0 <= mci[1]
    return {"within_game_rho_optimal_vs_chars": w, "ci_within": wci,
            "per_game_rho_mechanics_vs_tokens_per_level": m, "ci_mechanics": mci,
            "per_game_rho_mean_optimal_vs_tokens_per_level": o, "ci_optimal": oci,
            "median_chars_by_level_index": med,
            "level1_over_later_levels_median": sorted(first)[len(first) // 2] if first else None,
            "verdict": "HIT" if (length_ok and mech_ok) else "MISS",
            "rule": "HIT iff the mean within-game Spearman (optimal, generated chars) over solved levels > 0 "
                    "with its interval above 0, and the per-game Spearman (mechanic count, tokens per solved "
                    "level) has an interval containing 0"}


def p5(games_a, games_b):
    b = {g["id"]: g for g in games_b}
    pairs = [(g, b[g["id"]]) for g in games_a if g["id"] in b]
    d_all = lambda ps: mean([y["solved"] - x["solved"] for x, y in ps])
    sel = lambda group: [(x, y) for x, y in pairs if any(m in group for m in x["mechanics"])]
    d, ci = d_all(pairs), boot(pairs, d_all)
    dc, dn = d_all(sel(CONTACT)), d_all(sel(NON_CONTACT))
    ds = mean([(y["score"] or 0) - (x["score"] or 0) for x, y in pairs])
    ds_ci = boot(pairs, lambda ps: mean([(y["score"] or 0) - (x["score"] or 0) for x, y in ps]))
    ci_zero = ci is not None and ci[0] <= 0 <= ci[1]
    shaped = dc is not None and dn is not None and dc > 0 and dc > dn
    return {"pairs": len(pairs), "levels_b_minus_a": d, "ci": ci, "contact_games": len(sel(CONTACT)),
            "levels_diff_contact": dc, "non_contact_games": len(sel(NON_CONTACT)), "levels_diff_non_contact": dn,
            "score_b_minus_a": ds, "ci_score": ds_ci,
            "verdict_ci": "HIT" if ci_zero else "MISS", "verdict_shape": "HIT" if shaped else "MISS",
            "rule": "ci: the paired per-game levels difference's interval contains 0; shape: the mean levels "
                    "gain on games with a contact-shaped mechanic > 0 and > the gain on snake / switch games"}


def p6(games):
    first = [1 if g["solved"] >= 1 else 0 for g in games]
    t1 = [g["t1"] or 0 for g in games]
    testable = 0 < sum(first) < len(first)
    return {"first_level_clears": "%d of %d" % (sum(first), len(first)),
            "rho_t1_vs_first_clear": spearman(t1, first) if testable else None,
            "exploratory_rho_t1_vs_levels": spearman(t1, [g["solved"] for g in games]),
            "exploratory_rho_t1_vs_score": spearman(t1, [g["score"] or 0 for g in games]),
            "verdict": ("HIT" if (spearman(t1, first) or 0) > 0 else "MISS") if testable
            else "UNTESTABLE (every game or no game cleared its first level)",
            "rule": "HIT iff Spearman (T1's best level, the harness's first-level clear) > 0; untestable "
                    "without variance in the clears; the two exploratory correlations were not pre-registered"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--analysis", default=None)
    ap.add_argument("--b-report", default=None)
    ap.add_argument("--b-analysis", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    games = load_games(args.report, args.analysis)
    out = {"games": len(games), "P1": p1(games), "P2": p2(games), "P3": p3(games), "P4": p4(games), "P6": p6(games)}
    if args.b_report:
        out["P5"] = p5(games, load_games(args.b_report, args.b_analysis))
    r = out["P1"]
    print("P1 %s | level rate by mechanic count %s | Spearman(count, game rate) %s" % (
        r["verdict"], {k: fmt(v) for k, v in r["rate_by_count"].items()}, fmt(r["rho_count_vs_game_rate"])))
    r = out["P2"]
    print("P2 %s | A %s | B %s | mean A - mean B %s %s | pooled A %s vs B %s | pairs A>B %s" % (
        r["verdict"], {k: fmt(v) for k, v in r["A"].items()}, {k: fmt(v) for k, v in r["B"].items()},
        fmt(r["mean_A_minus_mean_B"], 3), fci(r["ci"], 3), fmt(r["pooled_A"], 3), fmt(r["pooled_B"], 3),
        r["pairs_A_above_B"]))
    r = out["P3"]
    print("P3 band %s, longer %s | pooled %s, median level %s over %d levels | Spearman(optimal, ratio) %s %s | by length %s" % (
        r["verdict_band"], r["verdict_longer"], fmt(r["pooled_ratio"]), fmt(r["median_level_ratio"]), r["levels"],
        fmt(r["rho_optimal_vs_ratio"]), fci(r["ci"]), {k: fmt(v) for k, v in r["by_length"].items()}))
    r = out["P4"]
    print("P4 %s | within-game Spearman(optimal, chars) %s %s | per-game Spearman(mechanics, tokens/level) %s %s"
          " | per-game Spearman(mean optimal, tokens/level) %s %s | median chars by level index %s | level 1 / later %s" % (
              r["verdict"], fmt(r["within_game_rho_optimal_vs_chars"]), fci(r["ci_within"]),
              fmt(r["per_game_rho_mechanics_vs_tokens_per_level"]), fci(r["ci_mechanics"]),
              fmt(r["per_game_rho_mean_optimal_vs_tokens_per_level"]), fci(r["ci_optimal"]),
              r["median_chars_by_level_index"], fmt(r["level1_over_later_levels_median"])))
    if "P5" in out:
        r = out["P5"]
        print("P5 ci %s, shape %s | %d pairs | levels b-a %s %s | contact games (%d) %s, snake/switch games (%d) %s"
              " | score b-a %s %s" % (r["verdict_ci"], r["verdict_shape"], r["pairs"], fmt(r["levels_b_minus_a"], 3),
                                      fci(r["ci"], 3), r["contact_games"], fmt(r["levels_diff_contact"], 3),
                                      r["non_contact_games"], fmt(r["levels_diff_non_contact"], 3),
                                      fmt(r["score_b_minus_a"]), fci(r["ci_score"])))
    else:
        print("P5 PENDING (no --b-report)")
    r = out["P6"]
    print("P6 %s | first-level clears %s | exploratory Spearman(T1, levels) %s, (T1, score) %s" % (
        r["verdict"], r["first_level_clears"], fmt(r["exploratory_rho_t1_vs_levels"]),
        fmt(r["exploratory_rho_t1_vs_score"])))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
