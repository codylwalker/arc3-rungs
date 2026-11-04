"""Build the two A/B kernels (notebook + kernel-metadata.json) from Franzen's live notebook.

Usage: python -I kaggle/build_kernels.py <live.ipynb> <live kernel-metadata.json> <kaggle-username> [out-dir]

Arm A, arc3-twin-a-control: Franzen's notebook + the twin_tools cell with TWIN_TOOLS = "0".
Arm B, arc3-twin-b-on:      the same with TWIN_TOOLS = "1".
Changes to Franzen's notebook, and nothing else:
- a header cell crediting him and Tufa Labs and naming the arm;
- the twin_tools cell (kaggle/notebook_cell.py) right after the cell that applies his harness patch;
- in the run-settings cell, the NON-submission branch goes back to his quick demo set (10 games, 1 pass)
  at up to 60 min a game, instead of the live full demo (25 games x 4 passes, ~9 h); the
  competition-rerun branch is asserted identical to his.
Metadata copies his sources, docker image (the pinned Python 3.12 environment) and machine shape
(NvidiaRtxPro6000), adds our private dataset, and keeps the kernels private.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

QUICK_DEMO_SETTINGS = '''# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.
bm.solver.max_runtime_s_per_game = 48*60 #7920.0
bm.solver.analyzer_timeout = 900.0
bm.solver.concurrency = 10
bm.solver.max_actions_per_game = None
bm.solver.save_request_logs = True
bm.n_passes = 1
bm.game_weights = None

# A Save Version plays Franzen's quick demo set (10 games, 1 pass) for up to 60 min a game (his was
# 25; the model load alone took 30+ min on 2026-10-22); the competition rerun below is unchanged.
demo_excluded_games = [] if TRUE_SUBMISSION else ['bp35', 'cd82', 'cn04', 'dc22', 'g50t', 'ka59', 'lf52', 'ls20', 'm0r0', 's5i5', 'sk48', 'sp80', 'su15', 'tn36', 'wa30']

if TRUE_SUBMISSION:  # overrides for contest run
    bm.solver.save_request_logs = False
    bm.n_passes = 1
    if USE_PRIORITY_SCHEDULING:
        bm.solver.concurrency = 120  # game count should be 110
        bm.solver.max_runtime_s_per_game = 532*60
else:
    if USE_PRIORITY_SCHEDULING:
        bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes
        bm.solver.max_runtime_s_per_game = 60*60
    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:
        raise RuntimeError(
            f'Expected the 32400-second notebook budget, got {target.max_runtime_s!r}.'
        )

# show config
print('benchmark.solver:', bm.solver)
'''

CONTEST_BRANCH = '''if TRUE_SUBMISSION:  # overrides for contest run
    bm.solver.save_request_logs = False
    bm.n_passes = 1
    if USE_PRIORITY_SCHEDULING:
        bm.solver.concurrency = 120  # game count should be 110
        bm.solver.max_runtime_s_per_game = 532*60
'''

ARMS = [
    ("arc3-twin-a-control", "arc3 twin a control", "0",
     "**Arm A (control):** the twin_tools wiring is applied but `ARC3_TWIN_TOOLS` is off, so the run is Franzen's."),
    ("arc3-twin-b-on", "arc3 twin b on", "1",
     "**Arm B:** `ARC3_TWIN_TOOLS` is on: the model's Python tool gains check_model, check_goal, plan, run_plan "
     "and best_probe, and the system prompt documents them."),
]


def src(cell):
    return "".join(cell.get("source", []))


def as_source(text):
    lines = text.splitlines(keepends=True)
    return lines


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    nb_path, meta_path, user = sys.argv[1], sys.argv[2], sys.argv[3]
    out_root = os.path.abspath(sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, "kernels"))
    with open(nb_path, encoding="utf-8") as f:
        base = json.load(f)
    with open(meta_path, encoding="utf-8") as f:
        base_meta = json.load(f)
    with open(os.path.join(HERE, "notebook_cell.py"), encoding="utf-8") as f:
        twin_cell = f.read()
    cells = base["cells"]
    patch_at = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "harness-changes.patch" in src(c)
                and "BUNDLE_DIR = Path(" in src(c)]
    settings_at = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and "demo_excluded_games = " in src(c)
                   and "bm.n_passes" in src(c)]
    if len(patch_at) != 1 or len(settings_at) != 1:
        raise SystemExit("expected one patch cell and one settings cell, found %s and %s" % (patch_at, settings_at))
    if CONTEST_BRANCH not in src(cells[settings_at[0]]) or CONTEST_BRANCH not in QUICK_DEMO_SETTINGS:
        raise SystemExit("the competition-rerun branch of the settings cell is not Franzen's; refusing")
    for slug, title, flag, note in ARMS:
        nb = copy.deepcopy(base)
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                c["outputs"] = []
                c["execution_count"] = None
        nb["cells"][settings_at[0]]["source"] = as_source(QUICK_DEMO_SETTINGS)
        cell_text = twin_cell.replace('TWIN_TOOLS = "1"', 'TWIN_TOOLS = "%s"' % flag)
        assert ('TWIN_TOOLS = "%s"' % flag) in cell_text
        nb["cells"].insert(patch_at[0] + 1, {"cell_type": "code", "execution_count": None, "metadata": {},
                                             "outputs": [], "source": as_source(cell_text)})
        header = ("# s2ar: twin_tools A/B on Daniel Franzen's Milestone-2 solution\n\n"
                  "%s\n\n"
                  "Everything below is Daniel Franzen's notebook "
                  "([kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution]"
                  "(https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution), Apache-2.0), itself built "
                  "on Tufa Labs' Duck harness; all credit for the harness, the serving stack and the model choice "
                  "is theirs. Our changes: the cell after section 1, which wires twin_tools from the attached "
                  "`arc3-twin-tools` dataset into the patched harness, and the quick demo for a Save Version in "
                  "section 7 (the competition rerun is unchanged).\n" % note)
        nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": as_source(header)})
        meta = {
            "id": "%s/%s" % (user, slug),
            "title": title,
            "code_file": "%s.ipynb" % slug,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": [],
            "dataset_sources": list(base_meta["dataset_sources"]) + ["%s/arc3-twin-tools" % user],
            "kernel_sources": [],
            "competition_sources": list(base_meta["competition_sources"]),
            "model_sources": list(base_meta["model_sources"]),
            "docker_image": base_meta["docker_image"],
            "machine_shape": base_meta["machine_shape"],
        }
        out = os.path.join(out_root, slug)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, meta["code_file"]), "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        with open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(meta, f, indent=2)
        print("wrote", out, "| cells", len(nb["cells"]), "| twin cell at", patch_at[0] + 2, "| TWIN_TOOLS", flag)


if __name__ == "__main__":
    main()
