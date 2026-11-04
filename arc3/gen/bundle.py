"""Merge generated sets into one environments dir for a harness run (arc_agi's offline Arcade scans one
dir). Checks each set's manifest against its game dirs (one <id>/<hash>/ per manifest entry, nothing
extra), refuses an id present in two sets (arc_agi would resolve it ambiguously), copies the game dirs
and writes a manifest whose entries carry their source set.

Usage (from tools/arc3): python gen/bundle.py --out gen/out/ab-set-a gen/out/set-a gen/out/set-a-click
"""
import argparse
import json
import os
import shutil


def set_games(s):
    with open(os.path.join(s, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    listed = {g["game_id"]: g for g in man["games"]}
    found = {}
    for gid in sorted(os.listdir(s)):
        gdir = os.path.join(s, gid)
        if not os.path.isdir(gdir):
            continue
        for h in sorted(os.listdir(gdir)):
            if os.path.isfile(os.path.join(gdir, h, "metadata.json")):
                found["%s-%s" % (gid, h)] = (gid, h)
    if set(found) != set(listed):
        raise SystemExit("%s: manifest and dirs disagree: only in dirs %s, only in manifest %s" % (
            s, sorted(set(found) - set(listed)), sorted(set(listed) - set(found))))
    return [(full, found[full][0], found[full][1], listed[full]) for full in sorted(found)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit("%s exists; bundle into a new dir" % a.out)
    owner, plan = {}, []
    for s in a.sets:
        for full, gid, h, cert in set_games(s):
            if gid in owner:
                raise SystemExit("id %s is in both %s and %s" % (gid, owner[gid], s))
            owner[gid] = s
            plan.append((s, gid, h, cert))
    os.makedirs(a.out)
    games = []
    for s, gid, h, cert in plan:
        shutil.copytree(os.path.join(s, gid, h), os.path.join(a.out, gid, h))
        games.append(dict(cert, source_set=os.path.basename(os.path.normpath(s))))
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"games": games, "sources": [os.path.normpath(s) for s in a.sets]}, f, indent=1)
    kinds = {}
    for g in games:
        kinds[g["source_set"]] = kinds.get(g["source_set"], 0) + 1
    print("bundled %d games into %s (%s)" % (len(games), a.out, ", ".join("%s %d" % kv for kv in sorted(kinds.items()))))


if __name__ == "__main__":
    main()
