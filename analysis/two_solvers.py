"""Do two unlike solvers fail the same public games? (the paper's section 3.2)

Reads the LLM harness's per-game final lines from its author's run log (the Kaggle notebook's
stdout as JSON records; each game's line reads "score=..., levels=<mean>/<total>, ...") and the
symbolic agent's stage-4 reports (arc3_s4b.json for the movement games, arc3_s4c.json for the
click games, written by s4b_sequence.py and s4c_click.py), and prints the overlap of the two
solvers' failures against chance.

A game stops the symbolic agent when no stage-4 report has it complete a level. A game is hard
for the harness when its score is below --hard (default 50). Printed: the 2x2 table; the overlap
expected under independence with its exact hypergeometric tails; the harness's mean score on the
games the symbolic agent passed against the rest, with an exact permutation P over every subset
of that size; and the harness's mean levels on the games that stop the symbolic agent and are
hard for the harness.

Usage: python -I analysis/two_solvers.py <run.log.json> <arc3_s4b.json> <arc3_s4c.json> [--hard 50] [--json out.json]
"""
import argparse
import itertools
import json
import re
from math import comb

LINE = re.compile(r"^\s+([a-z0-9]{4})-[0-9a-f]{8}: score=([0-9.]+), levels=([0-9.]+)/([0-9]+),")


def harness_games(path):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    rows = {}
    for rec in records:
        if rec.get("stream_name") != "stdout":
            continue
        for line in rec.get("data", "").splitlines():
            m = LINE.match(line)
            if m:
                rows[m.group(1)] = {"score": float(m.group(2)), "levels": float(m.group(3)), "of": int(m.group(4))}
    return rows


def symbolic_passed(paths):
    passed, seen = set(), set()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            report = json.load(f)
        for game, r in report.items():
            seen.add(game)
            if r.get("levels", 0) >= 1:
                passed.add(game)
    return passed, seen


def symbolic_scores(paths):
    """Each game's score on the harness's 0-100 scale, by the competition toolkit's rule (arc_agi 0.9.9,
    scorecard.py: a completed level min(115, (human / actions)^2 * 100), an uncompleted one 0, the level-index-weighted
    mean capped at 100 times the weight share of the completed levels); the best report where a game has two.
    Added 2026-11-06: the stage files' own game_score is a fraction of 1 and uncapped."""
    best = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            report = json.load(f)
        for game, r in report.items():
            n, done = r["of"], int(r.get("levels", 0))
            w = list(range(1, n + 1))
            lv = [min(115.0, (r["human"][i] / r["level_actions"][i]) ** 2 * 100)
                  if i < done and r["level_actions"][i] > 0 else 0.0 for i in range(n)]
            s = min(sum(x * k for x, k in zip(lv, w)) / sum(w), sum(w[:done]) / sum(w) * 100)
            best[game] = max(best.get(game, 0.0), s)
    return best


def hypergeom_tails(n_all, k_hard, n_stop, k_both):
    """P(X >= k_both) and P(X <= k_both) for the overlap X of a k_hard-set and an n_stop-set among n_all."""
    def pmf(x):
        return comb(k_hard, x) * comb(n_all - k_hard, n_stop - x) / comb(n_all, n_stop)
    lo, hi = max(0, n_stop - (n_all - k_hard)), min(k_hard, n_stop)
    return sum(pmf(x) for x in range(k_both, hi + 1)), sum(pmf(x) for x in range(lo, k_both + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_log")
    ap.add_argument("s4b")
    ap.add_argument("s4c")
    ap.add_argument("--hard", type=float, default=50.0)
    ap.add_argument("--json")
    a = ap.parse_args()

    harness = harness_games(a.run_log)
    passed, seen = symbolic_passed([a.s4b, a.s4c])
    games = sorted(harness)
    if set(games) != seen:
        raise SystemExit("the two records cover different games: %s" % sorted(set(games) ^ seen))
    n_all = len(games)
    stop = [g for g in games if g not in passed]
    hard = [g for g in games if harness[g]["score"] < a.hard]
    both = sorted(set(stop) & set(hard))
    expected = len(stop) * len(hard) / n_all
    p_ge, p_le = hypergeom_tails(n_all, len(hard), len(stop), len(both))

    on_passed = [harness[g]["score"] for g in games if g in passed]
    on_rest = [harness[g]["score"] for g in games if g not in passed]
    observed = sum(on_passed) / len(on_passed)
    scores = [harness[g]["score"] for g in games]
    subsets = 0
    at_least = 0
    for combo in itertools.combinations(scores, len(on_passed)):
        subsets += 1
        if sum(combo) / len(combo) >= observed - 1e-9:
            at_least += 1
    past_level1 = [g for g in games if harness[g]["levels"] >= 1.0]
    stopped_level1 = [g for g in games if harness[g]["levels"] < 1.0]
    sym = symbolic_scores([a.s4b, a.s4c])
    sym_mean = sum(sym.get(g, 0.0) for g in games) / n_all
    print("the symbolic agent's mean score over the %d games, the toolkit's rule on the harness's scale: %.3f" % (
        n_all, sym_mean))

    print("games %d | the symbolic agent passed a first level on %d: %s" % (n_all, len(passed), sorted(passed)))
    print("harness score below %.0f on %d games" % (a.hard, len(hard)))
    print("2x2: symbolic passed & harness hard %d | passed & not hard %d | stopped & hard %d | stopped & not hard %d" % (
        len(set(passed) & set(hard)), len(set(passed) - set(hard)), len(both), len(set(stop) - set(hard))))
    print("both (stop the symbolic agent, hard for the harness): %d, expected at random %.2f; P(>=) %.3f, P(<=) %.3f" % (
        len(both), expected, p_ge, p_le))
    print("harness mean score on the symbolic agent's %d: %.2f | on the other %d: %.2f | exact permutation P(mean >= observed) %.4f over %d subsets" % (
        len(on_passed), observed, len(on_rest), sum(on_rest) / len(on_rest), at_least / subsets, subsets))
    print("harness past level 1 (mean levels >= 1) on %d of %d; below: %s" % (len(past_level1), n_all, stopped_level1))
    print("harness mean levels on the %d 'both' games: %s" % (len(both), {g: harness[g]["levels"] for g in both}))

    if a.json:
        out = {
            "games": n_all,
            "symbolic_passed": sorted(passed),
            "harness_hard": hard,
            "hard_threshold": a.hard,
            "both": both,
            "both_count": len(both),
            "both_expected": expected,
            "p_overlap_ge": p_ge,
            "p_overlap_le": p_le,
            "mean_score_on_passed": observed,
            "mean_score_on_rest": sum(on_rest) / len(on_rest),
            "permutation_p_ge": at_least / subsets,
            "permutation_subsets": subsets,
            "harness_past_level1": len(past_level1),
            "harness_stopped_level1": stopped_level1,
            "both_levels": {g: harness[g]["levels"] for g in both},
            "both_past_level1": sum(1 for g in both if harness[g]["levels"] >= 1.0),
            "symbolic_mean_score": sym_mean,
        }
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        print("wrote " + a.json)


if __name__ == "__main__":
    main()
