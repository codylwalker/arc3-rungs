"""Build the Kaggle dataset folder of generated games for the quota evaluation (plan 2026-10-27, late).

Usage: python -I kaggle/build_gen_dataset.py <kaggle-username> [--set <bundle dir>] [--out <dir>]
Default set: arc3/gen/out/ab-set-a (41 games, 246 levels). Writes <out> (default
kaggle/arc3-gen-games/) with, per game, ONLY `<id4>/<hash>/<id4>.py` and `metadata.json` (what arc_agi's
OFFLINE Arcade needs: it finds games by rglob("metadata.json")); never cert.json or the manifest's
oracle paths: the harness never sees a certificate. Plus `GEN_GAMES_ROOT` (an empty marker the notebook's pre-check
cell finds under /kaggle/input), `games.json` (id, mechanics, baseline_actions, optimal per level; the
baselines are ceil(1.5 x optimal), so the optima leak nothing new) and a PRIVATE dataset-metadata.json.
Upload from inside <out>: `kaggle datasets create -p . --dir-mode zip` / `kaggle datasets version -p . -m
"<note>" --dir-mode zip`. Without --dir-mode the CLI skips every game folder and uploads only the top-level
files (2026-10-30); with it, Kaggle unpacks the zips into the <id4>/<hash>/ dirs. Evaluating on these games is not training (Kaggle rules 6a does not bind).
"""
import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_SET = os.path.normpath(os.path.join(ROOT, "..", "arc3", "gen", "out", "ab-set-a"))
MARKER = "GEN_GAMES_ROOT"


PAIR_FIELDS = ("goal", "variant", "pair_seed")  # the goal-rung pairs (gen/corpus.py --goal --pair, 2026-10-30)


def build(set_dir, out, user, slug="arc3-gen-games"):
    with open(os.path.join(set_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    games = []
    for g in manifest["games"]:
        gid = g["game_id"]
        id4, h = gid.split("-")
        src = os.path.join(set_dir, id4, h)
        dst = os.path.join(out, id4, h)
        os.makedirs(dst)
        for name in (id4 + ".py", "metadata.json"):
            if not os.path.isfile(os.path.join(src, name)):
                raise SystemExit("%s: missing %s" % (gid, name))
            shutil.copyfile(os.path.join(src, name), os.path.join(dst, name))
        row = {"game_id": gid, "mechanics": g["mechanics"], "baseline_actions": g["baseline_actions"],
               "optimal": [lv["optimal"] for lv in g["levels"]], "source_set": g.get("source_set")}
        row.update({k: g[k] for k in PAIR_FIELDS if k in g})
        games.append(row)
    with open(os.path.join(out, "games.json"), "w", encoding="utf-8") as f:
        json.dump({"set": os.path.basename(os.path.normpath(set_dir)), "games": games}, f, indent=1)
    with open(os.path.join(out, MARKER), "w", encoding="utf-8") as f:
        f.write("generated ARC-AGI-3 games, %d games; arc_agi OFFLINE environments_dir is this directory\n" % len(games))
    with open(os.path.join(out, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"title": slug, "id": "%s/%s" % (user, slug),
                   # our own emitted games and their data (LICENSE, LICENSE-DATA.md); uploaded private
                   "licenses": [{"name": "CC0-1.0"}]}, f, indent=2)
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("--set", default=DEFAULT_SET)
    ap.add_argument("--out", default=None, help="default kaggle/<slug>/")
    ap.add_argument("--slug", default="arc3-gen-games",
                    help="the dataset's slug (the goal-rung pairs: arc3-gen-goal, beside the 41-game set, never over it)")
    args = ap.parse_args()
    args.out = args.out or os.path.join(HERE, args.slug)
    games = build(args.set, os.path.abspath(args.out), args.user, args.slug)
    levels = sum(len(g["optimal"]) for g in games)
    print("wrote %s: %d games, %d levels, from %s" % (args.out, len(games), levels, args.set))


if __name__ == "__main__":
    main()
