"""Build the generated-games evaluation kernel from a D' FP8 kernel (plan 2026-10-27, late).

Usage: python -I kaggle/build_gen_variant.py <kaggle-username> [a|b] [--minutes 40] [--passes 1]
Arm a: from kernels/arc3-twin-a-fp8-dprime (twin tools wired, off). Arm b: from arc3-twin-b-fp8-dprime
(build_fp8_variant.py b --dprime first). Writes kernels/arc3-twin-<arm>-fp8-dprime-gen/: the source
unchanged except, in the Save Version demo only,
- a pre-check cell after the bundled-source cell: find the arc3-gen-games dataset (its GEN_GAMES_ROOT
  marker) under /kaggle/input, scan it with arc_agi's OFFLINE Arcade, print the ids and raise if the count
  differs from its games.json (a minute of CPU before the 15-35 min model load);
- the settings cell: every game (demo_excluded_games = []), --passes passes, --minutes a game on both the
  demo line and the priority branch, concurrency = <games> x passes (the admission gate holds the rest
  behind the 10 server slots, as in the contest);
- the benchmark cell: the offline game list comes from the dataset root instead of the competition's
  environment files.
The competition-rerun branch is checked intact. Attach the arc3-gen-games dataset (Add Input) besides the
kernel's usual inputs. Save Version only: this kernel is never submitted (a submission would play the live
gateway, which is pointless and spends the day's slot).
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"a": "arc3-twin-a-fp8-dprime", "b": "arc3-twin-b-fp8-dprime"}
SOURCE_CELL_MARKER = "def _source_path_entries("
OLD_EXCLUDED = "demo_excluded_games = [] if TRUE_SUBMISSION else ["
OLD_RUNTIME = "bm.solver.max_runtime_s_per_game = 48*60 #7920.0\n"
OLD_DEMO_PRIORITY_RUNTIME = "        bm.solver.max_runtime_s_per_game = 60*60\n"
OLD_DEMO_CONCURRENCY = "        bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes\n"
OLD_PASSES = re.compile(r"\nbm\.n_passes = \d+\nbm\.game_weights = None\n")  # the a kernel says 2, the b kernel 1
CONTEST_BRANCH = "if TRUE_SUBMISSION:  # overrides for contest run\n    bm.solver.save_request_logs = False\n    bm.n_passes = 1\n"
OLD_ENV = ('    competition_env_files = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")\n'
           "    bm.games = _offline_games(competition_env_files)\n")
NEW_ENV = ("    # generated-games evaluation: the arc3-gen-games dataset found by the pre-check cell\n"
           "    bm.games = _offline_games(GEN_GAMES_DIR)\n")
TWIN_FLAG_LINE = 'os.environ["ARC3_TWIN_TOOLS"] = TWIN_TOOLS\n'
PRECHECK_CELL = """\
# ## 4b. Generated games: find the arc3-gen-games dataset and check it before the model loads
# The dataset carries GEN_GAMES_ROOT at its root, one <id4>/<hash>/{<id4>.py, metadata.json} per game and
# games.json. arc_agi's OFFLINE Arcade finds games by rglob("metadata.json"); the count must match.
import arc_agi

_gen_marker = next(Path("/kaggle/input").rglob("GEN_GAMES_ROOT"), None)
if _gen_marker is None:
    raise RuntimeError("attach the arc3-gen-games dataset (Add Input -> Your Work -> Datasets)")
GEN_GAMES_DIR = str(_gen_marker.parent)
with open(_gen_marker.parent / "games.json", encoding="utf-8") as _f:
    GEN_GAMES = json.load(_f)["games"]
_gen_arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=GEN_GAMES_DIR)
_gen_ids = sorted(e.game_id for e in _gen_arcade.available_environments)
print("generated games:", len(_gen_ids), "of", len(GEN_GAMES), "expected, under", GEN_GAMES_DIR)
print(", ".join(_gen_ids))
if len(_gen_ids) != len(GEN_GAMES) or set(_gen_ids) != {g["game_id"] for g in GEN_GAMES}:
    raise RuntimeError("the arcade's game list differs from games.json: %s" % sorted(set(_gen_ids) ^ {g["game_id"] for g in GEN_GAMES}))
"""


def find_cell(cells, marker, what):
    hits = [i for i, c in enumerate(cells) if marker in "".join(c.get("source", []))]
    if len(hits) != 1:
        raise SystemExit("expected %s exactly once in the notebook, found %d" % (what, len(hits)))
    return hits[0]


def replace_once(cells, old, new, what):
    i = find_cell(cells, old, what)
    src = "".join(cells[i]["source"])
    if src.count(old) != 1:
        raise SystemExit("expected %s exactly once in its cell" % what)
    cells[i]["source"] = src.replace(old, new).splitlines(keepends=True)
    return cells[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("arm", nargs="?", default="a", choices=sorted(ARMS))
    ap.add_argument("--minutes", type=int, default=40)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--games", type=int, default=None, help="game count (default: from kaggle/arc3-gen-games/games.json)")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="games played at once; the server has 10 slots, so 10 gives each game its whole --minutes "
                         "(the 10-28 pilot used the game count: 41 clocks ran together behind 10 slots and every "
                         "game got about 10 min of model time)")
    ap.add_argument("--suffix", default="",
                    help="appended to the kernel slug (e.g. -c, -m): a separate slug per structural arm, so each "
                         "keeps its own version history and post_run's expected flag")
    ap.add_argument("--dataset", default="arc3-gen-games",
                    help="the generated-games dataset slug the kernel attaches (and kaggle/<slug>/games.json counts): "
                         "arc3-gen-goal for the goal-rung pairs (2026-10-30)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="an ARC3_* flag set in the twin cell after ARC3_TWIN_TOOLS (repeatable): "
                         "ARC3_AUTO_CERTIFY=1 (arm C), ARC3_AUTO_FRAME_DIFF=1 and ARC3_METERS=1 (arm M)")
    args = ap.parse_args()
    env = []
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep or not key.startswith("ARC3_") or not key.replace("_", "").isalnum():
            raise SystemExit("--env wants ARC3_NAME=value, got %r" % item)
        env.append((key, value))
    src_slug = ARMS[args.arm]
    dst_slug = src_slug + "-gen" + args.suffix
    src, dst = os.path.join(HERE, "kernels", src_slug), os.path.join(HERE, "kernels", dst_slug)
    if not os.path.isdir(src):
        raise SystemExit("%s missing: build it first (build_fp8_variant.py %s --dprime)" % (src, args.arm))
    n_games = args.games
    if n_games is None:
        with open(os.path.join(HERE, args.dataset, "games.json"), encoding="utf-8") as f:
            n_games = len(json.load(f)["games"])
    with open(os.path.join(src, src_slug + ".ipynb"), encoding="utf-8") as f:
        nb = json.load(f)
    with open(os.path.join(src, "kernel-metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cells = nb["cells"]
    # the pre-check cell after the bundled-source cell
    s = find_cell(cells, SOURCE_CELL_MARKER, "the bundled-source cell")
    cells.insert(s + 1, {"cell_type": "code", "source": PRECHECK_CELL.splitlines(keepends=True), "metadata": {},
                         "outputs": [], "execution_count": None})
    # the settings cell
    i = find_cell(cells, OLD_EXCLUDED, "the demo's excluded-games line")
    lines = "".join(cells[i]["source"]).splitlines(keepends=True)
    idx = [k for k, l in enumerate(lines) if l.startswith(OLD_EXCLUDED)]
    if len(idx) != 1:
        raise SystemExit("expected one excluded-games line")
    lines[idx[0]] = "demo_excluded_games = []  # generated-games evaluation: every game in the dataset\n"
    cells[i]["source"] = lines
    settings = replace_once(cells, OLD_RUNTIME, "bm.solver.max_runtime_s_per_game = %d*60  # generated games\n" % args.minutes,
                            "the demo's per-game runtime")
    replace_once(cells, OLD_DEMO_PRIORITY_RUNTIME, "        bm.solver.max_runtime_s_per_game = %d*60  # generated games\n" % args.minutes,
                 "the demo's priority-branch runtime")
    replace_once(cells, OLD_DEMO_CONCURRENCY, "        bm.solver.concurrency = %d  # generated games: one server slot per running game\n" % args.concurrency,
                 "the demo's priority-branch concurrency")
    src_settings = "".join(settings["source"])
    if len(OLD_PASSES.findall(src_settings)) != 1:
        raise SystemExit("expected the demo's n_passes line exactly once in the settings cell")
    settings["source"] = OLD_PASSES.sub("\nbm.n_passes = %d\nbm.game_weights = None\n" % args.passes, src_settings).splitlines(keepends=True)
    if CONTEST_BRANCH not in "".join(settings["source"]):
        raise SystemExit("the competition-rerun branch changed; refusing to build on it")
    # the benchmark cell
    replace_once(cells, OLD_ENV, NEW_ENV, "the offline environment-files line")
    # the twin cell: the arm's extra flags, right after ARC3_TWIN_TOOLS
    if env:
        extra = "".join("os.environ[%r] = %r  # build_gen_variant --env\n" % (k, v) for k, v in env)
        replace_once(cells, TWIN_FLAG_LINE, TWIN_FLAG_LINE + extra, "the twin cell's ARC3_TWIN_TOOLS line")
    head = "".join(cells[0]["source"])
    note = ("\n**Generated-games evaluation (kaggle/build_gen_variant.py, arm %s%s): the Save Version plays the %d generated "
            "games of the %s dataset, %d pass(es), %d at a time, up to %d min a game%s. Save Version only; "
            "never submit this kernel.**\n" % (args.arm, args.suffix, n_games, args.dataset, args.passes, args.concurrency,
                                               args.minutes, ("; flags " + ", ".join("%s=%s" % kv for kv in env)) if env else ""))
    cells[0]["source"] = (head + note).splitlines(keepends=True)
    # the title must resolve to the slug (Kaggle refuses a push whose title names another kernel: 409)
    meta.update(id="%s/%s" % (args.user, dst_slug), title=meta["title"] + " gen" + args.suffix.replace("-", " "),
                code_file=dst_slug + ".ipynb")
    ds = "%s/%s" % (args.user, args.dataset)
    if ds not in meta["dataset_sources"]:
        meta["dataset_sources"].append(ds)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, dst_slug + ".ipynb"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    with open(os.path.join(dst, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
    print("wrote", dst, "| games", n_games, "| passes", args.passes, "| concurrency", args.concurrency, "| minutes/game", args.minutes,
          "| pre-check cell at", s + 1, "| datasets", meta["dataset_sources"])


if __name__ == "__main__":
    main()
