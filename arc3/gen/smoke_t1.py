"""T1 smoke over a generated set: every game x every T1 policy x N actions, through arc_agi's offline
Arcade (the same path the harness uses). Passes when nothing raises and the blind policies (one fixed
action forever, a fixed click) clear no level. Also reports how many levels the other memoryless
policies clear: a first, cheap non-triviality reading (Phase B's gate is stricter: 3 x the summed
baselines, 3 seeds, random must clear nothing).

Usage (from tools/arc3, the .venv-arc3 interpreter):
    python gen/smoke_t1.py gen/out/set-a [--actions 300] [--seeds 1]
Writes <set>/smoke_t1.json; exit 1 on an exception or a blind clear.
"""
import argparse
import json
import logging
import os
import random
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import t1_probe  # noqa: E402
from arc_agi import Arcade, OperationMode  # noqa: E402

BLIND = ("blind_", "click_center")


def games(set_dir):
    out = []
    for root, _dirs, files in os.walk(set_dir):
        if "metadata.json" in files:
            with open(os.path.join(root, "metadata.json"), encoding="utf-8") as f:
                meta = json.load(f)
            out.append((meta["game_id"].split("-")[0], meta))
    return sorted(out, key=lambda g: g[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir")
    ap.add_argument("--actions", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    logging.disable(logging.INFO)  # arc_agi logs every make and scorecard
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=a.set_dir)
    rows, errors = [], []
    t0 = time.perf_counter()
    gs = games(a.set_dir)
    for gid, meta in gs:
        for seed in range(a.seeds):
            for pol in t1_probe.policies(random.Random(seed)):
                try:
                    r = t1_probe.run(arc, gid, pol, seed, a.actions)
                except Exception:  # noqa: BLE001 - the smoke records every failure
                    errors.append({"game": gid, "policy": pol.name, "seed": seed, "error": traceback.format_exc(limit=4)})
                    continue
                if r["steps"] == 0:
                    continue  # the policy does not apply (a click policy on a keyboard game, or the reverse)
                rows.append({"game": gid, "mechanics": meta.get("mechanics"), "policy": pol.name, "seed": seed,
                             "levels": r["levels"], "win_levels": r["win_levels"], "steps": r["steps"],
                             "resets": r["resets"], "first_level_at": r["first_level_at"]})
    blind_clears = [r for r in rows if r["policy"].startswith(BLIND) and r["levels"] > 0]
    by_pol = {}
    for r in rows:
        p = by_pol.setdefault(r["policy"], {"runs": 0, "games_with_a_clear": 0, "levels": 0})
        p["runs"] += 1
        p["levels"] += r["levels"]
        p["games_with_a_clear"] += r["levels"] > 0
    per_game = {}
    for r in rows:
        g = per_game.setdefault(r["game"], {"mechanics": r["mechanics"], "best_levels": 0, "best_policy": None})
        if r["levels"] > g["best_levels"]:
            g["best_levels"], g["best_policy"] = r["levels"], r["policy"]
    total_levels = sum(len(m.get("baseline_actions", [])) for _g, m in gs)
    summary = {"set": a.set_dir, "games": len(gs), "levels": total_levels, "actions": a.actions, "seeds": a.seeds,
               "runs": len(rows), "errors": len(errors), "blind_clears": len(blind_clears), "by_policy": by_pol,
               "levels_cleared_by_any_policy": sum(g["best_levels"] for g in per_game.values()),
               "secs": round(time.perf_counter() - t0, 1)}
    with open(os.path.join(a.set_dir, "smoke_t1.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"summary": summary, "per_game": per_game, "errors": errors, "blind_clears": blind_clears,
                   "rows": rows}, f, indent=1)
    print("games %d, levels %d, runs %d, errors %d, blind clears %d, %.0f s" % (
        len(gs), total_levels, len(rows), len(errors), len(blind_clears), summary["secs"]))
    for name, p in sorted(by_pol.items()):
        print("  %-20s runs %3d  games with a clear %3d  levels %3d" % (name, p["runs"], p["games_with_a_clear"],
                                                                         p["levels"]))
    for gid, g in sorted(per_game.items()):
        if g["best_levels"]:
            print("  cleared by a T1 policy: %s %s %d level(s) (%s)" % (gid, ",".join(g["mechanics"] or []),
                                                                       g["best_levels"], g["best_policy"]))
    for e in errors[:5]:
        print("ERROR", e["game"], e["policy"], e["error"].splitlines()[-1])
    sys.exit(1 if errors or blind_clears else 0)


if __name__ == "__main__":
    main()
