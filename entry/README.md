# entry: the harness changes we ran on ARC-AGI-3

The harness is Daniel Franzen's ARC-AGI-3 Milestone-2 solution (github.com/da-fr/arc-agi-3-solution,
Apache-2.0), itself built on Tufa Labs' Duck harness. It lets a model play by writing Python in a sandboxed
REPL. Nothing of that harness is included here; `harness/apply_twin.py` patches a copy of it. Our leaderboard
submission is that notebook with three changes: the D' slot priority (`priority_dprime.py`), a smaller
quantisation of the same model that loads faster, and the keep-alive (`twin_keepalive.py`). Every other change
below is wired in and switched off by default.

## The toolkit (`twin_tools.py`, behind `ARC3_TWIN_TOOLS=1`)

| tool | what it does |
|---|---|
| `check_model(predict)` | replays the model's `predict(frame, action)` over every logged transition of the level; reports how many it reproduces and the first mismatch, cell by cell |
| `check_goal(goal)` | evaluates the model's win test on every logged board; a correct one is False on all of them |
| `plan(predict, goal)` | breadth-first search inside the model's own `predict` for the shortest action list that makes `goal` true |
| `run_plan(actions, predict)` | executes a plan one action at a time through the harness's `action()`, stopping at the first board the model did not foresee |
| `best_probe([predict_a, predict_b, ...])` | the one action whose outcome best tells rival hypotheses apart |

The tools mask timer and budget bars before comparing boards. A bar is judged anywhere on the board by its
behaviour: a colour whose count moves one way on at least 90% of its changes, on at least 30% of the level's
actions. An explicit `ignore` given to `check_model` carries to `run_plan` and `best_probe` in the same snippet.

## The other changes

- **The unasked certificate** (`ARC3_AUTO_CERTIFY=1`, with the tools on): after a snippet runs, every
  world-model-like function it defined is replayed over the level's last 60 logged transitions, and one
  `[certify]` line goes into the snippet's output.
- **The diff and meters** (`meters.py`, `ARC3_METERS=1`, with the harness's own frame diff on through
  `ARC3_AUTO_FRAME_DIFF=1`): a `meter:` line in the level's opener for each budget bar or clock, and a `far:`
  note in each action's result for objects that appeared, vanished, recoloured or resized away from the move.
- **The D' slot priority** (`priority_dprime.py`): the scheduler's priority after the formula the team
  Affectify published.
- **The keep-alive** (`twin_keepalive.py`, `ARC3_KEEPALIVE_SECONDS`): the competition's gateway closes a
  scorecard that sees no action for 15 minutes, and a large model can take longer than that to load. While
  the model server does not answer its health check, the warmup game resets at the set interval; it stops as
  soon as the server answers or the game acts.
- `twin_prompt.py` documents the tools in the system prompt.

## Files

- `harness/apply_twin.py <copy>/ARC3-Inference`: wires everything into a copy of the harness. Every anchor
  must match exactly once or nothing is written; a second run reports nothing new.
- `kaggle/`: the builders. `build_dataset.py` writes the tools dataset the notebook attaches;
  `build_kernels.py` builds the kernels from Franzen's live notebook; `build_fp8_variant.py`,
  `build_m_variant.py` and `build_gen_variant.py` derive the variants; `build_gen_dataset.py` packs a
  generated set for Kaggle (programs and metadata only, never certificates); `post_run.py` fetches and checks
  a finished run; `notebook_cell.py` is the cell the builders insert.
- `tools/`: reading a run. `analyze_run.py` (tool calls, certificate lines, generated characters by level),
  `file_access.py` (the leak check), `goal_pairs.py` (the goal-rung predictions),
  `score_predictions.py` (the five predictions of the dynamics run), `search_code.py` (the model's own
  search code), `arm_agreement.py`, `replay_headroom.py`, `wall_signal.py`, `loss_accounting.py`,
  `log_summary.py`, and `traj_to_sft.py` (a run's trajectories as supervised records; the tools dataset
  carries it, and no run in the paper used it).
- `tests/`: unit tests (`test_*.py`) and integration checks that need a copy of the harness
  (`sandbox_integration.py`, `prompt_check.py`, `dprime_integration.py`) or the public games (`real_games.py`,
  `snapshot_size.py`). `bench_plan.py` times the planner on full-size boards.

## Tests

```
python -I tests/test_twin_tools.py          # and every other tests/test_*.py
python -I harness/apply_twin.py <copy>/ARC3-Inference
python -I tests/sandbox_integration.py <copy>/ARC3-Inference
python -I tests/real_games.py --env-dir <environment_files> ls20 tu93
```
