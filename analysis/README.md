# analysis: the statistics behind the paper

Each script reads run outputs (`kaggle/post_run.py` downloads them) or the reports the run tools write, and
prints its numbers. Run them from the repository root; `paired_level_cost.py` and `demo_cost.py` import
`entry/tools/analyze_run.py`, and `public_bars.py` imports `entry/twin_tools.py`.

| script | what it computes | paper |
|---|---|---|
| `repeat_null.py` | Do unfinished games repeat across runs? The exact distribution of the number of distinct games that k unfinished game-runs fall in, by dynamic programming over runs. | 4.2 |
| `switch_null.py` | The chance that some one mechanic holds every failure of a run, counted exactly over every subset of games. | 4.2 |
| `retest_stats.py` | Three runs side by side: which games fail, whether failures repeat, and how well one run's tokens rank another run's failures (AUC). With `--json`, the ceiling split of the two identical runs: tokens and scores by rank over all games, over the games both runs scored 100 and over those both scored under 100, with 95% game-bootstrap intervals (4,000 resamples, seed 11); the chance that two runs share no unfinished game; each run's optimality on the levels it solved (the share at the certified optimum, the median moves over it); and the token agreement with each game's size (the sum of its certified optima) partialled out. | 4.1, 4.2 |
| `two_solvers.py` | Do the symbolic agent and the LLM harness fail the same public games? The overlap against chance (exact hypergeometric tails), and the harness's mean score on the games the symbolic agent passed against the rest (exact permutation); also the symbolic agent's mean score on the harness's 0-100 scale, by the competition toolkit's scoring rule. Reads the stage-4 reports (`arc3/arc3_s4b.json`, `arc3/arc3_s4c.json`, written by the stages) and the harness author's run log (the Kaggle notebook's output log, not included). | 3.2 |
| `paired_level_cost.py` | Generated characters per completed level, paired by game and level, treatment against base, with a game-cluster bootstrap and the level difference beside it. Run it on two identical base runs for the placebo. | 4.3 |
| `demo_cost.py` | Per run: generated characters per completed level, and the share of generation spent on the level left unfinished. | 4.3 |
| `certify_context.py` | The unasked certificate's lines in their context: which games saw them, what they said and what the model did next. | 4.3 |
| `public_bars.py` | Plays each public game at random through level 1 and reports where its budget bar sits and whether the toolkit's mask finds it. Run it from `arc3/`, where `t1_probe.py --fetch` put the public games: `python -I ../analysis/public_bars.py ../entry`. | 3.4 |
| `check_goal_manifest.py` | Checks a goal-rung set: each pair has both variants, one goal kind, one mechanic list, identical optima, and every game replayed. | 4.4 |
| `goal_t_interval.py` | The goal-rung pairs' cost ratio (P-G2) with a t-interval over the pairs beside the pre-registered bootstrap, and by goal kind; it first reproduces `entry/tools/goal_pairs.py`'s ratio and bootstrap exactly. | 4.4 |
| `score_at_optimum.py` | The generated games' mean score with every baseline at the certified optimum, by the competition toolkit's scoring rule, after checking that its scorer reproduces every recorded game score at the real baselines; and the two runs' agreement on the rescored scores, where no game reaches 100. | 4.1, 4.2 |
| `make_figures.py` | Figures 2, 3 and 4, and the medians by level that Table 5 states (`chars-by-level.json`). Needs matplotlib. | 4.1, 4.2, 4.4 |

Examples:

```
python -I analysis/repeat_null.py v2=<run>/output/summary.txt v3=<run>/output/summary.txt ...
python -I analysis/switch_null.py report-a.json
python -I analysis/retest_stats.py report-a.json report-a3.json report-b.json --json ceiling-split.json
python -I analysis/two_solvers.py <run.log.json> arc3/arc3_s4b.json arc3/arc3_s4c.json --json two-solvers.json
python -I analysis/paired_level_cost.py --base <base run> <base run> --treat <treated run> --label M
python -I analysis/check_goal_manifest.py arc3/gen/out/goal-set-a
python -I analysis/goal_t_interval.py <goal run>/output --set arc3/gen/out/goal-set-a --json goal-t.json
python -I analysis/score_at_optimum.py report-a.json report-a3.json --json score-at-optimum.json
python -I analysis/make_figures.py <folder with report-a.json, report-a3.json, analysis-a.json, analysis-a3.json> out/
```
