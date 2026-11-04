"""twin_tools inside the patched harness sandbox (python_tool_sandbox.run_sandboxed_python),
driven by the toy maze from test_twin_tools acting as the host.

Checks: the patched bootstrap fits in one Linux argument (128 KiB; the sandbox passes it with
`-c`); the tools exist in a snippet only when ARC3_TWIN_TOOLS is set; a model-style snippet can
go predict -> check_model -> check_goal -> plan -> run_plan through the real action() path, with
the harness's own guards and terminal latch in the loop.

On Windows a command line is capped at 32,767 characters, so here the bootstrap runs from a temp
file instead of `-c`; Kaggle runs it as `-c` on Linux.

Usage: python -I tests/sandbox_integration.py <path/to/patched/ARC3-Inference>
"""
import os
import subprocess
import sys
import tempfile
import types

ROOT = os.path.abspath(sys.argv[1])
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import inference  # noqa: E402,F401  (a trivial package; inference.agent's __init__ pulls the whole agent)

_agent_pkg = types.ModuleType("inference.agent")
_agent_pkg.__path__ = [os.path.join(ROOT, "inference", "agent")]
sys.modules["inference.agent"] = _agent_pkg

from inference.agent import python_tool_sandbox as sb  # noqa: E402
from test_twin_tools import LEVEL1, LEVEL2, Maze  # noqa: E402
import twin_tools as tw  # noqa: E402

LINUX_ARG_MAX = 131072
_REAL_POPEN = subprocess.Popen


def _popen_via_file(args, **kwargs):
    if os.name == "nt" and len(args) >= 5 and args[3] == "-c":
        fd, path = tempfile.mkstemp(prefix="twin_bootstrap_", suffix=".py")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(args[4])
        args = list(args[:3]) + [path]
    return _REAL_POPEN(args, **kwargs)


class Host:
    """The solver side of the sandbox protocol over a toy game."""

    def __init__(self, game):
        self.game = game
        self.history = [{"action": "", "frame": self.payload(game.rows(), game.level, game.steps), "result": {}}]
        self.last = {}

    @staticmethod
    def payload(rows, level, step):
        return {"ascii": "\n".join(rows), "step": step, "level": level, "shape": [len(rows), len(rows[0])],
                "grid": [[tw.TWIN_CHARS.index(ch) for ch in row] for row in rows]}

    def state(self):
        return {"current_frame": self.history[-1]["frame"], "history": self.history,
                "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT"], "last_action_call_result": self.last}

    def handler(self, actions, stale_after=None):
        executed = []
        flags = {}
        board_changed = gameplay_changed = False
        for spec in actions:
            before = self.game.rows()
            result, shown = self.game.play(spec)
            text = tw.twin_action_text(spec)
            self.history.append({"action": text, "frame": self.payload(shown, self.game.level_shown, self.game.steps),
                                 "result": result})
            if result.get("game_over"):
                self.history.append({"action": "RESET", "frame": self.payload(self.game.rows(), self.game.level,
                                                                               self.game.steps),
                                     "result": {"automatic": True}})
            executed.append(text)
            diff = [(r, c) for r, (x, y) in enumerate(zip(before, shown)) for c in range(len(x)) if x[c] != y[c]]
            board_changed = board_changed or bool(diff)
            gameplay_changed = gameplay_changed or any(4 <= r < len(before) - 4 and 4 <= c < len(before[0]) - 4
                                                       for r, c in diff)
            flags = {k: v for k, v in result.items() if v}
            if flags:
                break
        self.last = dict(flags, executed=bool(executed), executed_actions=executed,
                         board_changed=board_changed, gameplay_changed=gameplay_changed)
        return {"action_result": self.last, "state": self.state()}


PROBE = """
names = dir()
print(sorted(n for n in ('check_model', 'check_goal', 'plan', 'run_plan', 'best_probe', 'parse_action',
                         'TwinFrame') if n in names))
"""

MODEL_SNIPPET = """
def predict(frame, action):
    name, _, _ = parse_action(action)
    moves = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
    if name not in moves:
        return None
    pos = [(r, row.index('R')) for r, row in enumerate(frame.rows) if 'R' in row]
    if not pos:
        return None
    r, c = pos[0]
    dr, dc = moves[name]
    if frame.rows[r + dr][c + dc] in 'GS':
        return {}
    return {(r, c): 'B', (r + dr, c + dc): 'R'}

def two(frame, action):
    name, _, _ = parse_action(action)
    moves = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
    if name not in moves:
        return None
    r, c = [(r, row.index('R')) for r, row in enumerate(frame.rows) if 'R' in row][0]
    dr, dc = moves[name]
    if frame.rows[r + dr][c + dc] in 'GS' or frame.rows[r + 2 * dr][c + 2 * dc] in 'GS':
        return {}
    return {(r, c): 'B', (r + 2 * dr, c + 2 * dc): 'R'}

def goal(frame):
    return frame.rows[6][14] == 'R'

for a in ['DOWN', 'UP', 'RIGHT', 'LEFT']:
    action([a])
probe = best_probe([predict, two])
print('probe', probe['bits'], probe['action'])
print('frameview rows', len(current_frame.rows), current_frame.cell(0, 0), len(history[0].frame.rows))
print(check_model(predict)['summary'])
print(check_model(two)['summary'])
print(check_goal(goal)['summary'])
p = plan(predict, goal)
print('plan', p['status'], p['length'])
out = run_plan(p['actions'], predict)
print('run', out['stopped'], out['executed'])
"""


def run(code, host):
    return sb.run_sandboxed_python(code=code, timeout_seconds=30, initial_state=host.state(),
                                   action_handler=host.handler)


def main():
    checks = []
    size = len(sb._SANDBOX_BOOTSTRAP.encode("utf-8"))
    checks.append(("bootstrap %d bytes < %d (one Linux argument)" % (size, LINUX_ARG_MAX), size < LINUX_ARG_MAX))
    compile(sb._SANDBOX_BOOTSTRAP, "<bootstrap>", "exec")
    checks.append(("bootstrap compiles", True))
    sb.subprocess.Popen = _popen_via_file
    try:
        os.environ.pop("ARC3_TWIN_TOOLS", None)
        off = run(PROBE, Host(Maze([LEVEL1, LEVEL2])))
        checks.append(("tools absent when ARC3_TWIN_TOOLS is unset: %r" % off.get("stdout", "").strip(),
                       off.get("stdout", "").strip() == "[]" and not off.get("error")))
        os.environ["ARC3_TWIN_TOOLS"] = "1"
        on = run(PROBE, Host(Maze([LEVEL1, LEVEL2])))
        checks.append(("tools present when set: %r" % on.get("stdout", "").strip(),
                       "check_model" in on.get("stdout", "") and not on.get("error")))
        host = Host(Maze([LEVEL1, LEVEL2]))
        out = run(MODEL_SNIPPET, host)
        lines = out.get("stdout", "").strip().splitlines()
        print("\n".join("  | " + line for line in lines))
        if out.get("error"):
            print("  error:", out["error"])
        checks.append(("best_probe inside the sandbox: one bit, first move", "probe 1.0 UP" in lines))
        checks.append(("the harness's own frames have .rows and .cell", "frameview rows 20 G 20" in lines))
        checks.append(("check_model inside the sandbox", any("reproduces 4 of 4" in s for s in lines)))
        checks.append(("check_model rejects the rival", any("reproduces 0 of 4" in s for s in lines)))
        checks.append(("check_goal inside the sandbox", any("False on all 5" in s for s in lines)))
        checks.append(("plan inside the sandbox finds the 13-move optimum", "plan found 13" in lines))
        checks.append(("run_plan through the real action() completes the level", "run level_completed 13" in lines))
        checks.append(("the host saw the level end", host.game.level == 2))
        checks.append(("no sandbox error", not out.get("error")))
    finally:
        sb.subprocess.Popen = _REAL_POPEN
    for label, ok in checks:
        print("%s  %s" % ("ok  " if ok else "FAIL", label))
    print("PASS" if all(ok for _, ok in checks) else "FAIL")
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
