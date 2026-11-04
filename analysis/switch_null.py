"""The switch pattern under an exact null: does one mechanic hold every failure?

python -I analysis/switch_null.py <harness_report json with per-game mechanics>

v2 left three of 41 games unfinished, all three with switch (3 of 11 switch games, 0 of 30 others). Under the
null that the failures fall on the games at random, P(a given mechanic m holds all k failures) = C(n_m, k) / C(41, k)
(counted exactly, no simulation). The pattern was found
by looking across all 13 mechanics, so the relevant number is P(SOME mechanic holds all k), counted exactly over
every k-subset of the 41 games. Also for the other runs' unfinished sets.
"""
import itertools
import json
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")
with open(sys.argv[1], encoding="utf-8") as f:
    rep = json.load(f)


def find_games(obj):
    """[(game, [mechanics])] from the report's per-game rows, wherever they sit."""
    if isinstance(obj, dict) and isinstance(obj.get("games"), dict):
        rows = obj["games"]
        if rows and all(isinstance(v, dict) and "mechanics" in v for v in rows.values()):
            return [(g, v["mechanics"]) for g, v in rows.items()]
    if isinstance(obj, dict):
        for key in ("per_game", "games", "rows"):
            if isinstance(obj.get(key), list) and obj[key] and isinstance(obj[key][0], dict):
                rows = obj[key]
                if any("mechanics" in r for r in rows):
                    return [(r.get("game") or r.get("game_id"), r["mechanics"]) for r in rows]
        for v in obj.values():
            got = find_games(v)
            if got:
                return got
    return None


games = find_games(rep)
if not games:
    raise SystemExit("no per-game mechanics in %s; keys %s" % (sys.argv[1], list(rep)[:12]))
mech = {g: set(m if isinstance(m, list) else str(m).split(",")) for g, m in games}
names = sorted(mech)
n = len(names)
allm = sorted(set().union(*mech.values()))
print("%d games, %d mechanics: %s" % (n, len(allm), ", ".join("%s %d" % (m, sum(1 for g in names if m in mech[g])) for m in allm)))
switch_n = sum(1 for g in names if "switch" in mech[g])
print("P(all 3 failures are switch games) = C(%d,3)/C(%d,3) = %.4f" % (switch_n, n, math.comb(switch_n, 3) / math.comb(n, 3)))
for k in (3, 4):
    hits = total = 0
    for trip in itertools.combinations(names, k):
        total += 1
        if set.intersection(*(mech[g] for g in trip)):
            hits += 1
    print("P(some one of the %d mechanics holds all %d failures) = %d / %d = %.4f" % (len(allm), k, hits, total, hits / total))
import re  # noqa: E402

full = {g[:4]: g for g in names}
LINE = re.compile(r"^\s+(g\w{3})-\w+: score=[\d.]+, levels=([\d.]+)/(\d+)")
for arg in sys.argv[2:]:
    run, _, path = arg.partition("=")
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line)
            if m and float(m.group(2)) < float(m.group(3)):
                ids.append(m.group(1))
    sets = [mech[full[i]] for i in ids if i in full]
    shared = sorted(set.intersection(*sets)) if sets else []
    print("  %-3s unfinished %s: mechanics shared by all: %s" % (run, ",".join(ids), shared or "none"))
