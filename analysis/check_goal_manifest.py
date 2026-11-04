"""Check the goal-rung set's manifest: pairs complete, variants matched, optima shared, every game replayed.

Usage: python -I analysis/check_goal_manifest.py <set dir>
"""
import collections
import json
import os
import sys


def main():
    d = sys.argv[1]
    with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    games = man["games"]
    print("games", len(games), "| top-level keys", sorted(man))
    print("fields of one game:", sorted(games[0]))
    by_pair = collections.defaultdict(list)
    for g in games:
        by_pair[g.get("pair_seed")].append(g)
    bad = []
    for seed, gs in sorted(by_pair.items(), key=lambda kv: str(kv[0])):
        variants = sorted(g.get("variant") for g in gs)
        optima = {json.dumps(g.get("optimal")) for g in gs}
        replay = [g.get("replay_ok") for g in gs]
        goal = {g.get("goal") for g in gs}
        mech = {json.dumps(g.get("mechanics")) for g in gs}
        ok = (variants == ["consequence", "signalled"] and len(optima) == 1 and all(r is True for r in replay)
              and len(goal) == 1 and len(mech) == 1)
        if not ok:
            bad.append(seed)
        print("pair %s | %s | goal %s | mech %s | optimal %s | replay %s | ids %s | %s" % (
            seed, variants, goal, mech, optima, replay, [g["game_id"] for g in gs], "OK" if ok else "BAD"))
    print("pairs", len(by_pair), "| bad", bad)
    # each game's folder holds its emitted files
    for g in games:
        id4 = g["game_id"].split("-")[0]
        p = os.path.join(d, id4)
        if not os.path.isdir(p):
            print("MISSING folder", p)
    return 1 if bad or len(by_pair) != 10 or len(games) != 20 else 0


if __name__ == "__main__":
    sys.exit(main())
