"""Per-game and per-mechanic report of a harness run over generated games, against the oracle and T1.

Usage (from tools/arc3): python -I gen/harness_report.py RUN_DIR [RUN_DIR ...] [--b RUN_DIR ...]
    --set gen/out/ab-set-a [...] [--t1 gen/out/set-a/smoke_t1.json ...] [--json out.json]
A RUN_DIR is a harness output dir (summary.txt, artifacts/*_viewer_data.json), as for ab_compare.py.
--set gives each game's mechanics (metadata.json) and optimal per level (cert.json); --t1
gives the memoryless policies' best level per game (smoke_t1.py output, keyed by the 4-char id).

Per game: levels solved / total, the mean of actions ÷ optimal over the solved levels (the efficiency
the scorer squares), tokens per solved level, the local score, mechanics, T1 best level.
Per mechanic (a game counts under each of its mechanics) and per mechanic count: games, levels solved /
total, the solve rate by level index (1, 2, 3+), the actions ÷ optimal ratio, tokens per solved level,
T1 levels. With --b, ab_compare's paired block (levels, score, tokens) and the per-mechanic level
deltas. Standard library only; reads run dirs as data (run with -I on downloaded files).
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ab_compare  # noqa: E402


def set_info(set_dirs):
    """{game_id: {"mechanics": [...], "optimal": [per level], "baseline": [...]}} from a generated set."""
    out = {}
    for s in set_dirs:
        for mp in glob.glob(os.path.join(s, "*", "*", "metadata.json")):
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            rec = {"mechanics": m.get("mechanics", []), "baseline": m.get("baseline_actions", []), "optimal": []}
            cp = os.path.join(os.path.dirname(mp), "cert.json")
            if os.path.isfile(cp):
                with open(cp, encoding="utf-8") as f:
                    rec["optimal"] = [lv["optimal"] for lv in json.load(f)["levels"]]
            out[m["game_id"]] = rec
    return out


def t1_info(paths):
    """{4-char id: best level cleared by any memoryless policy}."""
    out = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        for id4, rec in d.get("per_game", {}).items():
            out[id4] = max(out.get(id4, 0), int(rec.get("best_levels", 0)))
    return out


def per_level_actions(run_dirs, gid):
    """[actions per level] of the best pass (most levels) for a game, from its viewer_data files."""
    best = None
    for d in run_dirs:
        for vd in glob.glob(os.path.join(d, "**", "artifacts", "*_viewer_data.json"), recursive=True):
            with open(vd, encoding="utf-8", errors="replace") as f:
                v = json.load(f)
            if v.get("game_id") != gid:
                continue
            cand = (int(v.get("levels_completed", 0)), list(v.get("actions_per_level", [])))
            if best is None or cand[0] > best[0]:
                best = cand
    return best or (0, [])


def game_rows(arm, run_dirs, info, t1):
    rows = {}
    for gid, rec in sorted(arm.items()):
        meta = info.get(gid, {"mechanics": [], "optimal": [], "baseline": []})
        done, acts = per_level_actions(run_dirs, gid)
        ratios = [a / o for a, o in zip(acts[:done], meta["optimal"]) if o]
        rows[gid] = {
            "mechanics": meta["mechanics"], "n_mechanics": len(meta["mechanics"]),
            "levels": rec["levels"], "total_levels": rec["total_levels"], "passes": rec["passes"],
            "best_pass_levels": done, "actions_per_level": acts[:done],
            "optimal": meta["optimal"], "actions_over_optimal": round(sum(ratios) / len(ratios), 2) if ratios else None,
            "tokens": int(rec["tokens"]), "tokens_per_level_solved": round(rec["tokens"] / rec["levels"]) if rec["levels"] else None,
            "score": round(rec["score"], 2), "t1_best_level": t1.get(gid.split("-")[0]),
        }
    return rows


def group_rows(rows, key):
    """Aggregate game rows by a key function returning a list of group names per game."""
    groups = {}
    for gid, r in rows.items():
        for g in key(r):
            s = groups.setdefault(g, {"games": 0, "levels": 0.0, "total_levels": 0, "tokens": 0, "ratios": [],
                                      "by_index": {}, "t1_levels": 0, "t1_games": 0})
            s["games"] += 1
            s["levels"] += r["levels"]
            s["total_levels"] += r["total_levels"]
            s["tokens"] += r["tokens"]
            if r["actions_over_optimal"] is not None:
                s["ratios"].append(r["actions_over_optimal"])
            for i in range(r["total_levels"]):
                b = "1" if i == 0 else "2" if i == 1 else "3+"
                c = s["by_index"].setdefault(b, [0, 0])
                c[1] += 1
                if i < r["best_pass_levels"]:
                    c[0] += 1
            if r["t1_best_level"] is not None:
                s["t1_games"] += 1
                s["t1_levels"] += r["t1_best_level"]
    out = {}
    for g, s in sorted(groups.items()):
        out[g] = {"games": s["games"], "levels_solved": round(s["levels"], 1), "levels_total": s["total_levels"],
                  "solve_rate": round(s["levels"] / s["total_levels"], 3) if s["total_levels"] else None,
                  "solve_rate_by_index": {b: round(c[0] / c[1], 2) for b, c in sorted(s["by_index"].items())},
                  "actions_over_optimal": round(sum(s["ratios"]) / len(s["ratios"]), 2) if s["ratios"] else None,
                  "tokens_per_level_solved": round(s["tokens"] / s["levels"]) if s["levels"] else None,
                  "t1_levels_per_game": round(s["t1_levels"] / s["t1_games"], 2) if s["t1_games"] else None}
    return out


def print_table(title, groups):
    print("== %s" % title)
    print("  %-22s %5s %9s %6s %-24s %8s %9s %6s" % ("group", "games", "levels", "rate", "by level index", "act/opt", "tok/level", "T1"))
    for g, s in groups.items():
        idx = " ".join("%s:%.2f" % (b, v) for b, v in s["solve_rate_by_index"].items())
        print("  %-22s %5d %4.1f/%-4d %6s %-24s %8s %9s %6s" % (
            g, s["games"], s["levels_solved"], s["levels_total"], "" if s["solve_rate"] is None else "%.2f" % s["solve_rate"],
            idx, "" if s["actions_over_optimal"] is None else s["actions_over_optimal"],
            "" if s["tokens_per_level_solved"] is None else s["tokens_per_level_solved"],
            "" if s["t1_levels_per_game"] is None else s["t1_levels_per_game"]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run output dirs of the (first) arm")
    ap.add_argument("--b", nargs="*", default=[], help="run output dirs of a second arm (paired block)")
    ap.add_argument("--set", nargs="*", default=[], help="generated set dirs (metadata.json, cert.json)")
    ap.add_argument("--t1", nargs="*", default=[], help="smoke_t1.json files")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    info = set_info(args.set)
    t1 = t1_info(args.t1)
    arm = ab_compare.load_arm(args.runs)
    rows = game_rows(arm, args.runs, info, t1)
    out = {"games": rows}
    print("== per game (%d)" % len(rows))
    print("  %-16s %-28s %7s %8s %9s %6s %4s" % ("game", "mechanics", "levels", "act/opt", "tok/level", "score", "T1"))
    for gid, r in rows.items():
        print("  %-16s %-28s %3.1f/%-3d %8s %9s %6.1f %4s" % (
            gid, ",".join(r["mechanics"])[:28], r["levels"], r["total_levels"],
            "" if r["actions_over_optimal"] is None else r["actions_over_optimal"],
            "" if r["tokens_per_level_solved"] is None else r["tokens_per_level_solved"], r["score"],
            "" if r["t1_best_level"] is None else r["t1_best_level"]))
    out["by_mechanic"] = group_rows(rows, lambda r: r["mechanics"] or ["(unknown)"])
    out["by_mechanic_count"] = group_rows(rows, lambda r: ["%d mechanics" % r["n_mechanics"]])
    out["all"] = group_rows(rows, lambda r: ["all"])
    print_table("by mechanic", out["by_mechanic"])
    print_table("by mechanic count", out["by_mechanic_count"])
    print_table("all", out["all"])
    if args.b:
        b = ab_compare.load_arm(args.b)
        pairs = ab_compare.pair(arm, b)
        if pairs:
            out["paired"] = ab_compare.compare(arm, b, pairs)
            ab_compare.report("B vs A", out["paired"])
            mech = {}
            for ga, gb in pairs:
                for m in rows[ga]["mechanics"]:
                    mech.setdefault(m, []).append(b[gb]["levels"] - arm[ga]["levels"])
            out["paired_by_mechanic"] = {m: {"games": len(xs), "delta_levels_mean": round(sum(xs) / len(xs), 3)} for m, xs in sorted(mech.items())}
            print("  B-A levels by mechanic: " + "; ".join("%s %+.2f (%d)" % (m, v["delta_levels_mean"], v["games"]) for m, v in out["paired_by_mechanic"].items()))
        else:
            print("no game is in both arms")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
