"""Wire twin_tools into an ARC3-Inference checkout (Franzen's Milestone-2 harness,
github.com/da-fr/arc-agi-3-solution at 10882e3, Apache-2.0).

Run it on a COPY of the checkout; it edits files in place. Every anchor must occur exactly once,
and nothing is written until every edit has been applied in memory, so a failed run leaves the
tree untouched. Running it twice is a no-op.

Usage: python -I harness/apply_twin.py <path/to/ARC3-Inference>

What it changes, all behind ARC3_TWIN_TOOLS (off unless set to 1/on/true/yes):
- copies twin_tools.py and twin_prompt.py to inference/utils/;
- python_tool_sandbox.py: pastes the module's source into the sandbox bootstrap, registers
  check_model / check_goal / plan / run_plan / best_probe / parse_action / TwinFrame in the
  snippet's globals when the host says so, and has the host say so from the environment variable;
- tool_agent.py: appends TWIN_TOOLS_PROMPT to the system prompt when the variable is set;
- framework/solver.py: after the warmup RESET, starts twin_keepalive (repeat RESETs every
  ARC3_KEEPALIVE_SECONDS while the model server loads; off unless that variable is set).

And behind ARC3_PRIORITY_FORM=dprime (off unless set), priority_dprime: the slot priority of
Franzen's admission gate becomes Affectify's D' (priority_dprime.py) - tool_agent.py tracks each
game's tokens per cleared level and scores snapshots by D', priority_scheduler.py's PrioritySnapshot
carries that figure, and solver.py queues an unstarted game with a level-1 snapshot instead of the
untrimmed top band.
"""
import os
import shutil
import sys

MARKER = "twin_tools_enabled"

SANDBOX_EDITS = [
    (
        "from inference.utils import frame_diff as _frame_diff_module\n",
        "from inference.utils import frame_diff as _frame_diff_module\n"
        "from inference.utils import twin_tools as _twin_tools_module\n",
    ),
    (
        "    __FRAME_DIFF_SOURCE__\n",
        "    __FRAME_DIFF_SOURCE__\n\n    __TWIN_TOOLS_SOURCE__\n",
    ),
    (
        "        runtime_globals[\"action\"] = action\n",
        "        runtime_globals[\"action\"] = action\n"
        "        if initial.get(\"twin_tools_enabled\"):\n"
        "            # twin_tools: the model's predict/goal checked against the logged game, and\n"
        "            # searched for the shortest plan, on the CPU; none of it spends an action\n"
        "            def check_model(predict, level=None, ignore=\"auto\", include_terminal=False):\n"
        "                return twin_check_model(predict, runtime_globals.get(\"transitions\") or [],\n"
        "                                        level=level, ignore=ignore,\n"
        "                                        include_terminal=include_terminal)\n"
        "\n"
        "            def check_goal(goal, level=None):\n"
        "                return twin_check_goal(goal, runtime_globals.get(\"history\") or [], level=level)\n"
        "\n"
        "            def plan(predict, goal, start=None, moves=None, clicks=None, max_nodes=20000,\n"
        "                     time_limit=8.0, ignore=\"auto\", max_depth=None):\n"
        "                return twin_plan(\n"
        "                    predict, goal,\n"
        "                    start if start is not None else runtime_globals.get(\"current_frame\"),\n"
        "                    moves=moves, clicks=clicks, max_nodes=max_nodes, time_limit=time_limit,\n"
        "                    ignore=ignore, max_depth=max_depth,\n"
        "                    valid_actions=runtime_globals.get(\"valid_actions\"),\n"
        "                    transitions=runtime_globals.get(\"transitions\") or [])\n"
        "\n"
        "            def run_plan(actions, predict=None, ignore=\"auto\"):\n"
        "                return twin_run_plan(actions, predict, runtime_globals[\"action\"],\n"
        "                                     lambda: runtime_globals.get(\"current_frame\"),\n"
        "                                     ignore=ignore,\n"
        "                                     transitions=runtime_globals.get(\"transitions\") or [])\n"
        "\n"
        "            def best_probe(predicts, start=None, moves=None, clicks=None, ignore=\"auto\",\n"
        "                           weights=None):\n"
        "                return twin_best_probe(\n"
        "                    predicts,\n"
        "                    start if start is not None else runtime_globals.get(\"current_frame\"),\n"
        "                    moves=moves, clicks=clicks, ignore=ignore, weights=weights,\n"
        "                    valid_actions=runtime_globals.get(\"valid_actions\"),\n"
        "                    transitions=runtime_globals.get(\"transitions\") or [])\n"
        "\n"
        "            runtime_globals.update(check_model=check_model, check_goal=check_goal, plan=plan,\n"
        "                                   run_plan=run_plan, best_probe=best_probe,\n"
        "                                   parse_action=twin_parse_action, TwinFrame=TwinFrame)\n",
    ),
    (
        "    \"__FRAME_DIFF_SOURCE__\\n\", inspect.getsource(_frame_diff_module)\n)\n",
        "    \"__FRAME_DIFF_SOURCE__\\n\", inspect.getsource(_frame_diff_module)\n"
        ").replace(\n"
        "    \"__TWIN_TOOLS_SOURCE__\\n\", inspect.getsource(_twin_tools_module)\n)\n",
    ),
    (
        "def _sandbox_env() -> dict[str, str]:\n",
        "def _twin_tools_enabled() -> bool:\n"
        "    return os.environ.get(\"ARC3_TWIN_TOOLS\", \"\").strip().lower() in {\"1\", \"on\", \"true\", \"yes\"}\n"
        "\n"
        "\n"
        "def _sandbox_env() -> dict[str, str]:\n",
    ),
    (
        "                \"color_chars\": ARC_COLOR_CHARS,\n",
        "                \"color_chars\": ARC_COLOR_CHARS,\n"
        "                \"twin_tools_enabled\": _twin_tools_enabled(),\n",
    ),
    (
        "            return self._segmentation\n\n        def __str__(self):\n            rows, cols = self.shape\n",
        "            return self._segmentation\n\n"
        "        @property\n"
        "        def rows(self):\n"
        "            # twin_tools: TwinFrame's surface on the harness's own frames too (on 2026-10-23 a model\n"
        "            # called current_frame.rows and its snippet died before its run_plan)\n"
        "            return tuple(self.ascii.split(\"\\n\"))\n\n"
        "        def cell(self, row, col):\n"
        "            return self.ascii.split(\"\\n\")[row][col]\n\n"
        "        def __str__(self):\n            rows, cols = self.shape\n",
    ),
]

CERTIFY = "twin_auto_certify"

CERTIFY_SANDBOX_EDITS = [
    (
        "                \"twin_tools_enabled\": _twin_tools_enabled(),\n",
        "                \"twin_tools_enabled\": _twin_tools_enabled(),\n"
        "                \"auto_certify\": _auto_certify_enabled(),\n",
    ),
    (
        "def _twin_tools_enabled() -> bool:\n",
        "def _auto_certify_enabled() -> bool:\n"
        "    # twin_auto_certify: ARC3_AUTO_CERTIFY=1 replays a predict the snippet defined over the level's\n"
        "    # log, unasked, and prints one line (needs the twin tools on)\n"
        "    return os.environ.get(\"ARC3_AUTO_CERTIFY\", \"\").strip().lower() in {\"1\", \"on\", \"true\", \"yes\"}\n"
        "\n"
        "\n"
        "def _twin_tools_enabled() -> bool:\n",
    ),
    (
        "            _send(\n"
        "                {\n"
        "                    \"type\": \"final\",\n"
        "                    **_collect_retention_payload(),\n",
        "            if initial.get(\"twin_tools_enabled\") and initial.get(\"auto_certify\"):\n"
        "                # twin_auto_certify: every world-model-like function this snippet defined, replayed\n"
        "                # over the level's logged transitions on the CPU; one line into the snippet's stdout\n"
        "                try:\n"
        "                    _tw_names = [_n.name for _n in _tree.body if isinstance(_n, ast.FunctionDef)]\n"
        "                    _tw_line = twin_auto_certify(_tw_names, runtime_globals,\n"
        "                                                 runtime_globals.get(\"transitions\") or [])\n"
        "                    if _tw_line:\n"
        "                        _tw_text = stdout.getvalue()\n"
        "                        stdout.write((\"\" if not _tw_text or _tw_text.endswith(\"\\n\") else \"\\n\") + _tw_line + \"\\n\")\n"
        "                except Exception:\n"
        "                    pass\n"
        "            _send(\n"
        "                {\n"
        "                    \"type\": \"final\",\n"
        "                    **_collect_retention_payload(),\n",
    ),
]

CERTIFY_AGENT_EDITS = [
    (
        "        prompt += TWIN_TOOLS_PROMPT\n",
        "        prompt += TWIN_TOOLS_PROMPT\n"
        "        if _get_env_bool(\"ARC3_AUTO_CERTIFY\", False):\n"
        "            # twin_auto_certify: the [certify] line the sandbox appends unasked\n"
        "            from inference.utils.twin_prompt import TWIN_CERTIFY_NOTE\n"
        "            prompt += TWIN_CERTIFY_NOTE\n",
    ),
]

METERS = "meters_note"

METERS_AGENT_EDITS = [
    (
        "        lines.extend(self._prepare_auto_diff(current_frame, previous_step_summary))\n"
        "        state_line = f\"Current state: step {current_step}, level {current_level}\"\n",
        "        lines.extend(self._prepare_auto_diff(current_frame, previous_step_summary))\n"
        "        if _get_env_bool(\"ARC3_METERS\", False):\n"
        "            # meters: the level's budget bars and clocks named for the model (inference.utils.meters;\n"
        "            # the far-change line of each action travels in its verdict, see meters_note in solver.py)\n"
        "            try:\n"
        "                from inference.utils.meters import meters_lines as _meters_lines\n"
        "                lines.extend(_meters_lines(history_entries, current_level))\n"
        "            except Exception:\n"
        "                pass\n"
        "        state_line = f\"Current state: step {current_step}, level {current_level}\"\n",
    ),
]

METERS_SOLVER_EDITS = [
    (
        "def _guards_active(level: Any) -> bool:\n",
        "def _meters_enabled() -> bool:\n"
        "    # meters_note: ARC3_METERS=1 adds the far-change line to every executed action's verdict\n"
        "    return os.environ.get(\"ARC3_METERS\", \"\").strip().lower() in {\"1\", \"on\", \"true\", \"yes\"}\n"
        "\n"
        "\n"
        "def _guards_active(level: Any) -> bool:\n",
    ),
    (
        "    vocabulary for the model to learn.\"\"\"\n"
        "    if item.get(\"game_over\"):\n",
        "    vocabulary for the model to learn.\"\"\"\n"
        "    note = item.get(\"meters_note\")\n"
        "    if note:\n"
        "        # meters_note: consequences far from the move, with the verdict in echo and trace alike\n"
        "        return _action_verdict_base(item) + \" | \" + str(note)\n"
        "    return _action_verdict_base(item)\n"
        "\n"
        "\n"
        "def _action_verdict_base(item: dict) -> str:\n"
        "    if item.get(\"game_over\"):\n",
    ),
    (
        "        gameplay_changed = _interior_gameplay_changed(previous_grid, new_grid)\n"
        "        no_op = gameplay_changed is False\n",
        "        gameplay_changed = _interior_gameplay_changed(previous_grid, new_grid)\n"
        "        no_op = gameplay_changed is False\n"
        "        meters_note = None\n"
        "        if _meters_enabled() and board_changed and action.id.name != \"RESET\":\n"
        "            # meters_note: objects that appeared, vanished or recoloured far from what the action\n"
        "            # moved (or from the click), on the CPU (inference.utils.meters)\n"
        "            try:\n"
        "                from inference.utils.meters import action_note as _meters_action_note\n"
        "                from inference.utils.meters import harness_diff_fn as _meters_diff_fn\n"
        "                _m_origin = None\n"
        "                if action.id == arcengine.GameAction.ACTION6:\n"
        "                    _m_origin = (int(action.data.get(\"y\", 0)), int(action.data.get(\"x\", 0)))\n"
        "                meters_note = _meters_action_note(previous_grid, new_grid, _meters_diff_fn(), _m_origin)\n"
        "            except Exception:\n"
        "                meters_note = None\n",
    ),
    (
        "            \"action_display\": action_display,\n"
        "            \"batch_index\": batch_index,\n",
        "            \"action_display\": action_display,\n"
        "            \"meters_note\": meters_note,\n"
        "            \"batch_index\": batch_index,\n",
    ),
    (
        "            \"animation\", \"automatic\",\n"
        "        )\n",
        "            \"animation\", \"automatic\", \"meters_note\",\n"
        "        )\n",
    ),
]

AGENT_EDITS = [
    (
        "from inference.utils.retained_functions import function_signature\n",
        "from inference.utils.retained_functions import function_signature\n"
        "from inference.utils.twin_prompt import TWIN_TOOLS_PROMPT\n",
    ),
    (
        "        prompt += STEP_VERIFICATION_ADDENDUM\n",
        "        prompt += STEP_VERIFICATION_ADDENDUM\n"
        "    if _get_env_bool(\"ARC3_TWIN_TOOLS\", False):\n"
        "        # twin_tools: check_model / check_goal / plan / run_plan / best_probe (python_tool_sandbox)\n"
        "        prompt += TWIN_TOOLS_PROMPT\n",
    ),
]


def edit(text, edits, name):
    for anchor, replacement in edits:
        n = text.count(anchor)
        if n != 1:
            raise SystemExit("%s: anchor found %d times, expected once:\n%s" % (name, n, anchor))
        text = text.replace(anchor, replacement)
    return text


SOLVER_EDITS = [
    (
        "            self._execute_auto_reset()\n"
        "        try:\n"
        "            retry_analysis_step: int | None = None\n",
        "            self._execute_auto_reset()\n"
        "            # twin_keepalive: repeat the warmup RESET every ARC3_KEEPALIVE_SECONDS while the model\n"
        "            # server loads, so the gateway's 15-minute idle limit cannot close the scorecard\n"
        "            from inference.utils.twin_keepalive import start_keepalive as _twin_keepalive\n"
        "            _twin_keepalive(self, log=log)\n"
        "        try:\n"
        "            retry_analysis_step: int | None = None\n",
    ),
]

DPRIME = "priority_dprime"

DPRIME_AGENT_EDITS = [
    (
        "class _PriorityGate:\n",
        "def _priority_form() -> str:\n"
        "    \"\"\"priority_dprime: ARC3_PRIORITY_FORM=dprime swaps the slot priority for Affectify's D'.\"\"\"\n"
        "    return os.environ.get(\"ARC3_PRIORITY_FORM\", \"\").strip().lower()\n"
        "\n"
        "\n"
        "class _PriorityGate:\n",
    ),
    (
        "        return priority_value(\n"
        "            snapshot, endgame=endgame, tail_fraction=tail_fraction,\n",
        "        if _priority_form() == \"dprime\":\n"
        "            from inference.utils.priority_dprime import dprime_priority\n"
        "            value = dprime_priority(snapshot, tail_fraction)\n"
        "            if not getattr(self, \"_dprime_announced\", False):\n"
        "                self._dprime_announced = True\n"
        "                log.warning(\n"
        "                    \"priority_dprime: D' slot priority active; first snapshot level %d, %d actions, \"\n"
        "                    \"%d tokens, %s levels, p %s, fade %s -> %d\", snapshot.level, snapshot.actions,\n"
        "                    int(snapshot.tokens), getattr(snapshot, \"total_levels\", None),\n"
        "                    getattr(snapshot, \"tokens_per_level\", None), tail_fraction, value)\n"
        "            return value\n"
        "        return priority_value(\n"
        "            snapshot, endgame=endgame, tail_fraction=tail_fraction,\n",
    ),
    (
        "    def acquire(self, priority: int) -> None:\n"
        "        with self._cond:\n"
        "            self._enqueue_and_wait(priority)\n",
        "    def acquire(self, priority: int, snapshot: PrioritySnapshot | None = None) -> None:\n"
        "        # priority_dprime: an unstarted game may bring a snapshot, so the refreshed queue prices it\n"
        "        with self._cond:\n"
        "            self._enqueue_and_wait(priority, snapshot)\n",
    ),
    (
        "            self._actions_at_level_start = _priority_action_count(summary)\n"
        "            self._tokens_at_level_start = self._session_generated_tokens\n",
        "            # priority_dprime: D''s p, the mean generated tokens per level cleared in this game\n"
        "            _dp_cleared = max(1, int(summary.get(\"level\") or 2) - 1)\n"
        "            if _dp_cleared > getattr(self, \"_dprime_last_cleared\", 0):\n"
        "                self._dprime_last_cleared = _dp_cleared\n"
        "                self._dprime_cleared = getattr(self, \"_dprime_cleared\", 0) + 1\n"
        "                self._dprime_cleared_tokens = getattr(self, \"_dprime_cleared_tokens\", 0) + max(\n"
        "                    0, self._session_generated_tokens - getattr(self, \"_tokens_at_level_start\", 0))\n"
        "            self._actions_at_level_start = _priority_action_count(summary)\n"
        "            self._tokens_at_level_start = self._session_generated_tokens\n",
    ),
    (
        "            snapshot = PrioritySnapshot(level, actions, tokens, pace, total_levels)\n",
        "            snapshot = PrioritySnapshot(level, actions, tokens, pace, total_levels)\n"
        "            if _priority_form() == \"dprime\":\n"
        "                # priority_dprime: D' needs the level count and p in every snapshot\n"
        "                _dp_n = getattr(self, \"_dprime_cleared\", 0)\n"
        "                snapshot = PrioritySnapshot(\n"
        "                    level, actions, tokens, pace, self._priority_total_levels(),\n"
        "                    tokens_per_level=(getattr(self, \"_dprime_cleared_tokens\", 0) / _dp_n) if _dp_n else None)\n",
    ),
]

DPRIME_SCHEDULER_EDITS = [
    (
        "    total_levels: int | None = None\n",
        "    total_levels: int | None = None\n"
        "    # priority_dprime: D''s p, mean generated tokens per cleared level; None before the first clear\n"
        "    tokens_per_level: float | None = None\n",
    ),
]

DPRIME_SOLVER_EDITS = [
    (
        "            gate.acquire(_PRIORITY_UNTRIMMED_BASE - self._queue_position())\n",
        "            if os.environ.get(\"ARC3_PRIORITY_FORM\", \"\").strip().lower() == \"dprime\":\n"
        "                # priority_dprime: an unstarted game is priced by the same formula as the rest\n"
        "                # instead of jumping the queue (Affectify's D', its second change)\n"
        "                from inference.agent.priority_scheduler import PrioritySnapshot as _DpSnapshot\n"
        "                from inference.agent.priority_scheduler import parse_total_levels as _dp_levels\n"
        "                from inference.utils.priority_dprime import dprime_priority as _dp_priority\n"
        "                _dp_snapshot = _DpSnapshot(\n"
        "                    1, 0, 0.0, 1.0, _dp_levels(getattr(self.game, \"number_of_levels\", None)) or 10)\n"
        "                gate.acquire(_dp_priority(_dp_snapshot, None), _dp_snapshot)\n"
        "            else:\n"
        "                gate.acquire(_PRIORITY_UNTRIMMED_BASE - self._queue_position())\n",
    ),
]

MODULES = ("twin_tools.py", "twin_prompt.py", "twin_keepalive.py", "priority_dprime.py", "meters.py")

# (relative path, [(edits, marker meaning "this group is already applied"), ...]); a file's groups are
# applied in order, in memory, so a later group edits the text the earlier ones produced
TARGETS = [
    (os.path.join("inference", "agent", "python_tool_sandbox.py"),
     [(SANDBOX_EDITS, MARKER), (CERTIFY_SANDBOX_EDITS, CERTIFY)]),
    (os.path.join("inference", "agent", "tool_agent.py"),
     [(AGENT_EDITS, "TWIN_TOOLS_PROMPT"), (DPRIME_AGENT_EDITS, DPRIME), (CERTIFY_AGENT_EDITS, CERTIFY),
      (METERS_AGENT_EDITS, METERS)]),
    (os.path.join("inference", "agent", "priority_scheduler.py"), [(DPRIME_SCHEDULER_EDITS, DPRIME)]),
    (os.path.join("inference", "framework", "solver.py"),
     [(SOLVER_EDITS, "twin_keepalive"), (DPRIME_SOLVER_EDITS, DPRIME), (METERS_SOLVER_EDITS, METERS)]),
]


def read(path):
    """Text with LF newlines, and the newline the file uses on disk (a Windows checkout has CRLF)."""
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8").replace("\r\n", "\n"), ("\r\n" if b"\r\n" in raw else "\n")


def module_dir():
    """Beside this script (the flat Kaggle dataset), or one level up (the repo's harness/ dir)."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (here, os.path.dirname(here)):
        if all(os.path.isfile(os.path.join(candidate, name)) for name in MODULES):
            return candidate
    raise SystemExit("%s not found beside %s or one level up" % (", ".join(MODULES), here))


def copy_modules(root):
    source = module_dir()
    for name in MODULES:
        shutil.copyfile(os.path.join(source, name), os.path.join(root, "inference", "utils", name))


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = os.path.abspath(sys.argv[1])
    pending = []
    for rel, groups in TARGETS:
        path = os.path.join(root, rel)
        text, nl = read(path)
        new = text
        for edits, marker in groups:
            if marker in new:
                print("already applied:", rel, marker)
                continue
            new = edit(new, edits, rel)
        if new == text:
            continue
        compile(new, path, "exec")
        pending.append((path, new, nl))
    copy_modules(root)
    for path, text, nl in pending:
        tmp = path + ".twin-tmp"
        with open(tmp, "w", encoding="utf-8", newline=nl) as f:
            f.write(text)
        os.replace(tmp, path)
    print("applied to", root, "| edited:", [os.path.basename(p) for p, _, _ in pending] or "nothing new")


if __name__ == "__main__":
    main()
