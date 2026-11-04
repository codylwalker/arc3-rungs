# twin_prompt: the system-prompt section that documents twin_tools to the model.
#
# Kept apart from twin_tools.py because the sandbox pastes that file into its bootstrap,
# which travels as one command-line argument (Linux caps one at 128 KiB); the prompt
# text is only needed on the host. Licence: MIT-0 (see LICENSE).
#
# The worked example is real code: tests/test_twin_tools.py runs it against a toy game, so the
# prompt cannot teach a predict that does not work.

TWIN_TOOLS_PROMPT = """\
World-model tools (pure CPU: they spend no actions and no tokens):
- Write `predict(frame, action)` returning the board you expect after `action` from `frame`: the full board \
(a list of row strings like `frame.rows`, or ascii text) or a dict `{(row, col): 'X'}` of only the cells that \
change (colour letters). Return None where you do not know. `frame` has `.rows` (tuple of row strings), \
`.ascii`, `.shape`, `.level`, `.step`, `.segmentation`, `.cell(row, col)`. `action` is 'UP', 'DOWN', 'LEFT', \
'RIGHT', 'SPACE' or 'MOUSE(row=R, col=C)'; `parse_action(action)` returns (name, row, col).
- `check_model(predict)` replays predict over every logged transition of this level and returns how many it \
reproduces and the first mismatch as cells [row, col, predicted, actual]. Timer cells (changing on most \
actions) are ignored by default; `ignore=None` is strict, `ignore='border'` skips the four-cell border.
- `check_goal(goal)` evaluates `goal(frame)` on every logged board of this level. A correct win test is False \
on all of them (unless winning needs one more action).
- `plan(predict, goal)` searches YOUR predict from `current_frame` for the shortest action list reaching \
`goal(frame)`. Options: `moves=[...]`, `clicks=[(row, col), ...]` or `clicks='objects'`, `max_nodes`, \
`time_limit` (seconds, at most 12). Returns `status` ('found', 'exhausted', 'node_limit', 'time_limit') and \
`actions`.
- `run_plan(actions, predict)` executes the actions one at a time and stops at the first board your predict \
did not foresee, so a wrong model costs one action, not the whole plan. It stops on game over and level \
completion too.
- `best_probe([predict_a, predict_b, ...])`: when several hypotheses fit what you have seen, returns the one \
action from `current_frame` whose outcome best tells them apart (`action`, `bits`, and `split`: which \
hypotheses predict which board). One informative action beats many exploratory ones.
Example predict for a sprite of colour 'R' that moves one cell per arrow over floor 'B' and stops at walls 'G':
def predict(frame, action):
    d = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}.get(parse_action(action)[0])
    if d is None:
        return None
    cells = [(r, c) for r, row in enumerate(frame.rows) for c, ch in enumerate(row) if ch == 'R']
    moved = [(r + d[0], c + d[1]) for r, c in cells]
    if any(frame.rows[r][c] == 'G' for r, c in moved):
        return {}
    out = {cell: 'B' for cell in cells}
    out.update({cell: 'R' for cell in moved})
    return out
Use them when you have a hypothesis about the mechanics: predict -> check_model until it matches -> goal -> \
check_goal -> plan -> run_plan; with rival hypotheses, best_probe first. `plan` calls predict thousands of \
times, so read `frame.rows` there rather than `frame.segmentation`, which is far slower. Every action you take \
counts against your score; computation does not.
"""

# Appended when ARC3_AUTO_CERTIFY is set: the sandbox replays a predict-like function the snippet
# defined over the level's log without being asked, and prints one [certify] line.
TWIN_CERTIFY_NOTE = """\
Whenever a snippet defines a world-model function (named like predict/step/simulate, or taking a frame \
and an action), the sandbox replays it over this level's logged transitions and prints one line \
`[certify] <name> reproduces N of M logged transitions ...` with the first mismatch. A model that \
reproduces all of them is safe to plan with; a mismatch names the transition to look at.
"""
