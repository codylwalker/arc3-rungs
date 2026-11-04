# What a game asks for: the ARC-AGI-3 rungs, measured

Code, games and certificates for the paper *What a Game Asks For: A Certified Generator Measures Where an LLM
Harness Stops on ARC-AGI-3* (Cody Lee Walker, ARC Prize 2026 Paper Track).

The paper decomposes an ARC-AGI-3 game into rungs (perception, dynamics, goal, exploration, planning and
efficiency), builds a generator whose every level carries a certified optimum, and measures where an open LLM
harness stops. Everything needed to rebuild the games, re-certify them, and re-read a run is here. The runs
themselves are Kaggle notebooks; their outputs are not included.

## Contents

| folder | what it is |
|---|---|
| `arc3/gen/` | The generator, on ARCEngine's public API: the composer (`compose.py`), the movement and click runtimes, the breadth-first certifier (`certify.py`), emission, the corpus driver (`corpus.py`), the harness reports and their tests. |
| `arc3/gen/out/ab-set-a/` | The paper's 41 generated games (246 levels). Each game folder holds the emitted program, its `metadata.json` and its certificate `cert.json`; `manifest.json` lists the set. |
| `arc3/gen/out/goal-set-a/` | The goal-rung pairs: 10 certified plans, each emitted twice, signalled and consequence (20 games, 120 levels). |
| `arc3/` | The symbolic agent: object tracking (`arcobj.py`), rule induction (`arcrules.py`) and its stages `s0` to `s4c`. Also the memoryless probe `t1_probe.py` and the replay seal `t3_seal.py`. |
| `entry/` | What we changed in the harness we ran: the model-checking toolkit (`twin_tools.py`), its prompt section, the unasked certificate, the diff and meters (`meters.py`), the D' slot priority (`priority_dprime.py`) and the patcher that wires them in (`harness/apply_twin.py`). Also the Kaggle builders, the run-reading tools and their tests. See `entry/README.md`. |
| `analysis/` | The scripts behind the paper's statistics: the exact nulls, the paired cost per completed level, the run-to-run statistics and the public games' budget bars. See `analysis/README.md`. |

## Licence

- **Code** (every `.py` file, including the emitted game programs): MIT-0. See `LICENSE`.
- **Data** (`cert.json`, `manifest.json`, `metadata.json`, `games.json`): CC0 1.0. See `LICENSE-DATA.md`.
- **Third-party lines:** a few anchor lines matched against the Apache-2.0 harness and notebook stay under
  Apache-2.0. See `NOTICE`.

## Requirements

- Python 3.12 with `arc-agi` 0.9.9, which brings `arcengine` 0.9.3, and numpy.
- The public ARC-AGI-3 games are competition data and are not included. `arc-agi` downloads them into
  `environment_files/` the first time a probe runs with `--fetch`.
- The Kaggle steps need the Kaggle CLI and an account entered in the ARC-AGI-3 competition.

## The generator

From `arc3/`:

```
python gen/tests/test_gen.py            # and every other gen/tests/test_*.py
python gen/corpus.py --out gen/out/my-set --n 5 --seed 1000 --k 2 --max-nodes 50000 --max-secs 120
python gen/corpus.py --out gen/out/my-goal-set --n 5 --seed 1000 --k 2 --goal keys --pair
python gen/render.py gen/out/ab-set-a/g04u/d976eddd --cert --level 1 --out shot.png
```

The scripts under `arc3/` import their neighbours, so run them without `-I`.

`corpus.py` keeps a game only when every level certifies, and replays every emitted file against its
certified paths before keeping it. `--goal keys|plates --pair` emits each plan twice, with the goal on the HUD
(`signalled`) and without it (`consequence`).

## The symbolic agent

From `arc3/`, after `python t1_probe.py --fetch --actions 300 --seeds 2` has written `arc3_t1.json` (the
memoryless baselines and the human baselines the stages read):

```
python s0_census.py
python s1_track.py
python s2_induce.py
python s2b_agent.py
python s2c_roles.py
python s3_explore.py
python s4_contact.py
python s4b_sequence.py
python s4c_click.py
```

Each stage writes its `arc3_s*.json` and prints one line per game.

## The Kaggle runs

The harness is Daniel Franzen's ARC-AGI-3 Milestone-2 solution (github.com/da-fr/arc-agi-3-solution,
Apache-2.0), itself built on Tufa Labs' Duck harness. Our kernels are built from his live notebook. From
`entry/`:

```
kaggle kernels pull dfranzen/arc-agi-3-milestone-2-solution -p live -m
python -I kaggle/build_dataset.py <user>                      # the tools dataset the notebook attaches
python -I kaggle/build_kernels.py live/<notebook>.ipynb live/kernel-metadata.json <user>
python -I kaggle/build_fp8_variant.py <user> a --dprime       # the submitted configuration
python -I kaggle/build_gen_dataset.py <user> --set ../arc3/gen/out/ab-set-a
python -I kaggle/build_gen_variant.py <user> a --passes 1 --concurrency 10 --minutes 40
```

Upload the datasets private, from inside each folder: `kaggle datasets create -p .`, and for a games dataset
`kaggle datasets create -p . --dir-mode zip`, since without it the CLI skips the game folders. Push the
generated-games kernel with `kaggle kernels push -p .`; the push runs it as a Save Version. Never submit a
generated-games kernel: it would play the competition's games, not ours. The goal-rung run uses
`build_gen_dataset.py <user> --set ../arc3/gen/out/goal-set-a --slug arc3-gen-goal` and
`build_gen_variant.py <user> a --suffix=-goal --dataset arc3-gen-goal --passes 5 --concurrency 10 --minutes 40`.

## Reading a run

From `entry/`:

```
python -I kaggle/post_run.py --out out/gen <user>/<kernel slug>
python -I tools/file_access.py out/gen/<slug>/output           # the model's code never read the games' files
python ../arc3/gen/harness_report.py out/gen/<slug>/output --set ../arc3/gen/out/ab-set-a --json report.json
python -I tools/analyze_run.py out/gen/<slug>/output --json analysis.json
python -I tools/score_predictions.py --report report.json --analysis analysis.json
python -I tools/goal_pairs.py out/gen/<goal slug>/output --set ../arc3/gen/out/goal-set-a
```

The statistics across runs are in `analysis/`.

## Where each result comes from

| paper | script |
|---|---|
| 3.2, the symbolic agent's tracking, induction, levels | `arc3/s1_track.py`, `arc3/s2_induce.py`, `arc3/s4_contact.py` (the stages' own reports) |
| 3.2, the two solvers' failures against chance, and the symbolic agent's mean score | `analysis/two_solvers.py` |
| 3.3, memoryless policies: 20 of 246 generated levels, 3 of 183 public | `arc3/t1_probe.py` |
| 3.3, the generator and its certificates | `arc3/gen/corpus.py`, `arc3/gen/certify.py`; the sets under `arc3/gen/out/` |
| 4.1, the pilot row of the runs table | `entry/tools/analyze_run.py` on the pilot's output |
| 4.1, generated characters by level (Figure 2; its medians) | `entry/tools/analyze_run.py` (`gen_chars_by_level`); `analysis/make_figures.py` |
| 4.2, two identical runs (Figure 3) | `entry/tools/arm_agreement.py`; `analysis/make_figures.py` |
| 4.4, the goal pairs' cost ratios with their intervals (Figure 4) | `analysis/goal_t_interval.py`; `analysis/make_figures.py` |
| 3.3 / 4.4, the goal pair (Figure 1) | `writeup/make_figure.py --paper` in the paper's folder, over `arc3/gen/out/goal-set-a` |
| 4.5, the hidden set | the Kaggle leaderboard; `entry/kaggle/poll_scores.py` logs the draws |
| 4.1, levels, games won, mean score, the optimum | `arc3/gen/harness_report.py`, `entry/tools/analyze_run.py`; each run's optimality, `analysis/retest_stats.py`; the score with every baseline at the optimum, `analysis/score_at_optimum.py` |
| 4.1, the six predictions | `entry/tools/score_predictions.py` |
| 4.1, the model's own search | `entry/tools/search_code.py` |
| 4.2, agreement between runs, with intervals and on the score's ceiling | `entry/tools/arm_agreement.py`, `analysis/retest_stats.py` |
| 4.2, the exact nulls | `analysis/repeat_null.py`, `analysis/switch_null.py` |
| 4.3, cost per completed level and its placebo | `analysis/paired_level_cost.py`, `analysis/demo_cost.py` |
| 4.3, the certificate's lines | `entry/tools/analyze_run.py`, `analysis/certify_context.py` |
| 3.4, the public games' budget bars | `analysis/public_bars.py` |
| 3.5, the file-access check | `entry/tools/file_access.py` |
| 4.4, the goal-rung pairs | `entry/tools/goal_pairs.py`, `analysis/check_goal_manifest.py`; the cost ratio's t-interval, `analysis/goal_t_interval.py` |

## Authorship

Cody Lee Walker. AI coding and writing tools were used throughout, at the author's direction and under the
author's review.
