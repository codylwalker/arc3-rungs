"""The replay's ceiling: what a second play of each won level would score.

The gateway scores a game by its best run, and a new run starts every level's action count at zero.
So a replay that repeats only the successful attempt of each won level (the actions after the level's
last RESET) drops every failed attempt's actions, and a replay at the human baseline scores 100 a
level. This reads a harness run's artifacts/*_events.jsonl (one per game and pass) and reports, per
game and overall, the score as played, the score if every won level were replayed at its successful
attempt's length, and the score at the baseline. Levels not won are unchanged (0) in every column.

Usage: python -I tools/replay_headroom.py RUN_DIR [RUN_DIR ...]
           (--env-dir <environment_files> | --set <generated set dir> [...]) [--json out.json]
  --env-dir: the public games' metadata.json (baseline_actions per level), looked up by the game's
             4-character id;
  --set:     a generated set (metadata.json baseline_actions; cert.json optimal per level), which adds
             the same three columns rescored with the oracle optimum as the baseline.
A game with several passes is scored by its best pass (most levels, then score), as the gateway does.
Actions are counted as taaf counts them (RESET included); the "now" column is checked against each
pass's viewer_data final_score when present. Standard library only; run with -I on downloaded files.
"""
import argparse
import glob
import json
import os
import sys


def level_score(baseline, actions):
    if not actions or not baseline:
        return 0.0
    return min(115.0, (baseline / actions) ** 2 * 100.0)


def game_score(baselines, actions_by_level, won):
    """taaf's _compute_final_score: 1-indexed level weights, capped at the won levels' weight share."""
    n = len(baselines)
    total = weights = max_w = 0.0
    for i in range(n):
        w = i + 1
        weights += w
        s = level_score(baselines[i], actions_by_level[i]) if won[i] else 0.0
        if s > 0:
            max_w += w
        total += s * w
    if weights == 0:
        return 0.0
    return min(total / weights, max_w / weights * 100.0)


def read_events(path):
    """Per level (1-based index -> record): actions as counted, the successful attempt's length,
    resets, game overs, won.

    An action is charged to the level it was taken on: one more than the levels completed BEFORE
    it (the event's `score` is the count after it; its `level` field has already advanced on the
    completing action, so it is not used)."""
    levels = {}
    attempt = {}
    completed = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("type") != "action":
                continue
            lv = completed + 1
            rec = levels.setdefault(lv, {"actions": 0, "attempt": 0, "resets": 0, "game_overs": 0, "won": False})
            rec["actions"] += 1
            name = str(e.get("action_name") or "")
            if name == "RESET":
                rec["resets"] += 1
                attempt[lv] = 0
                continue
            attempt[lv] = attempt.get(lv, 0) + 1
            if e.get("game_over"):
                rec["game_overs"] += 1
            after = e.get("score")
            won_now = (int(after) > completed) if isinstance(after, (int, float)) else bool(
                e.get("level_completed") or e.get("done"))
            if won_now:
                rec["won"] = True
                rec["attempt"] = attempt[lv]
                completed = int(after) if isinstance(after, (int, float)) else completed + 1
    return levels


def game_id_of(path):
    base = os.path.basename(path)
    return base.split("_p")[0]


def pass_of(path):
    base = os.path.basename(path)
    try:
        return int(base.split("_p")[1].split("_")[0])
    except (IndexError, ValueError):
        return 0


def viewer_score(events_path):
    vd = events_path.replace("_events.jsonl", "_viewer_data.json")
    if not os.path.isfile(vd):
        return None
    with open(vd, encoding="utf-8", errors="replace") as f:
        v = json.load(f)
    return v.get("final_score"), list(v.get("actions_per_level") or [])


def load_baselines(env_dirs, set_dirs):
    """{id: {"baseline": [...], "optimal": [...] or None}}, keyed by the full id and its 4-char prefix."""
    out = {}
    for d in env_dirs:
        for mp in glob.glob(os.path.join(d, "*", "*", "metadata.json")):
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            rec = {"baseline": list(m.get("baseline_actions") or []), "optimal": None}
            out[m["game_id"]] = rec
            out[m["game_id"][:4]] = rec
    for d in set_dirs:
        for mp in glob.glob(os.path.join(d, "*", "*", "metadata.json")):
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            rec = {"baseline": list(m.get("baseline_actions") or []), "optimal": None}
            cp = os.path.join(os.path.dirname(mp), "cert.json")
            if os.path.isfile(cp):
                with open(cp, encoding="utf-8") as f:
                    rec["optimal"] = [lv["optimal"] for lv in json.load(f)["levels"]]
            out[m["game_id"]] = rec
            out[m["game_id"][:4]] = rec
    return out


def score_pass(levels, baselines):
    n = len(baselines)
    won = [levels.get(i + 1, {}).get("won", False) for i in range(n)]
    now = [levels.get(i + 1, {}).get("actions", 0) for i in range(n)]
    att = [levels.get(i + 1, {}).get("attempt", 0) for i in range(n)]
    return {
        "won": sum(won),
        "levels": n,
        "now": game_score(baselines, now, won),
        "attempt": game_score(baselines, att, won),
        "baseline": game_score(baselines, list(baselines), won),
        "deaths": sum(levels.get(i + 1, {}).get("game_overs", 0) for i in range(n)),
        "levels_with_resets": sum(1 for i in range(n) if won[i] and levels.get(i + 1, {}).get("resets", 0)),
        "won_at_attempt_ratio": [att[i] / baselines[i] for i in range(n) if won[i] and baselines[i]],
        "won_now_ratio": [now[i] / baselines[i] for i in range(n) if won[i] and baselines[i]],
    }


def analyse(run_dirs, info):
    games = {}
    checks = []
    for d in run_dirs:
        for p in sorted(glob.glob(os.path.join(d, "**", "artifacts", "*_events.jsonl"), recursive=True)):
            gid = game_id_of(p)
            rec = info.get(gid) or info.get(gid[:4])
            if rec is None or not rec["baseline"]:
                checks.append("%s: no baseline known, skipped" % gid)
                continue
            levels = read_events(p)
            res = score_pass(levels, rec["baseline"])
            res["pass"] = pass_of(p)
            if rec["optimal"]:
                opt = score_pass(levels, rec["optimal"])
                res["opt_now"], res["opt_attempt"], res["opt_baseline"] = opt["now"], opt["attempt"], opt["baseline"]
            vs = viewer_score(p)
            if vs is not None and vs[0] is not None:
                if abs(float(vs[0]) - res["now"]) > 0.05:
                    checks.append("%s p%d: now %.2f but viewer_data final_score %.2f" % (gid, res["pass"], res["now"], vs[0]))
            games.setdefault(gid, []).append(res)
    best = {}
    for gid, passes in games.items():
        best[gid] = max(passes, key=lambda r: (r["won"], r["now"]))
    return best, games, checks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--env-dir", action="append", default=[])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if not a.env_dir and not a.set:
        ap.error("give --env-dir or --set for the baselines")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    info = load_baselines(a.env_dir, a.set)
    best, games, checks = analyse(a.run_dirs, info)
    if not best:
        print("no events files found under", a.run_dirs)
        return 1
    has_opt = any("opt_now" in r for r in best.values())
    head = "%-16s %5s %4s | %7s %8s %9s" % ("game", "won", "pass", "now", "attempt", "baseline")
    if has_opt:
        head += " | %7s %8s %9s" % ("@opt", "att@opt", "opt")
    head += " | deaths resets"
    print(head)
    for gid in sorted(best):
        r = best[gid]
        line = "%-16s %2d/%-2d %4d | %7.2f %8.2f %9.2f" % (gid, r["won"], r["levels"], r["pass"], r["now"], r["attempt"], r["baseline"])
        if has_opt:
            line += " | %7.2f %8.2f %9.2f" % (r.get("opt_now", 0), r.get("opt_attempt", 0), r.get("opt_baseline", 0))
        line += " | %6d %6d" % (r["deaths"], r["levels_with_resets"])
        print(line)
    n = len(best)
    mean = lambda k: sum(r[k] for r in best.values()) / n  # noqa: E731
    print("\n%d games (best pass of %d game-passes): mean score now %.2f, at the successful attempt %.2f, at the baseline %.2f"
          % (n, sum(len(v) for v in games.values()), mean("now"), mean("attempt"), mean("baseline")))
    if has_opt:
        print("  rescored with the oracle optimum as the baseline: now %.2f, attempt %.2f, optimum %.2f"
              % (mean("opt_now"), mean("opt_attempt"), mean("opt_baseline")))
    won_levels = sum(r["won"] for r in best.values())
    with_resets = sum(r["levels_with_resets"] for r in best.values())
    ratios_now = [x for r in best.values() for x in r["won_now_ratio"]]
    ratios_att = [x for r in best.values() for x in r["won_at_attempt_ratio"]]
    print("  won levels %d, of which %d had a RESET (the replay drops those attempts); actions ÷ baseline on won levels: now %.2f, successful attempt %.2f"
          % (won_levels, with_resets, sum(ratios_now) / max(1, len(ratios_now)), sum(ratios_att) / max(1, len(ratios_att))))
    at_or_under = sum(1 for x in ratios_att if x <= 1.0)
    print("  won levels whose successful attempt is already at or under the baseline: %d of %d" % (at_or_under, len(ratios_att)))
    for c in checks:
        print("  check:", c)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"best": best, "passes": games, "checks": checks}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
