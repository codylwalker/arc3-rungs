"""priority_dprime wired into the notebook's exact harness tree.

Patches a COPY of the tree, re-patches it (must be a no-op), then drives the real code:
- the real _PriorityGate admits three waiting games in D' order (deep before fresh before stuck;
  Franzen's untrimmed band would have let the fresh game jump the queue);
- the real ToolAgent._maybe_handover hands over a D' snapshot carrying N and p;
- the patched level-up bookkeeping and the patched solver admission block, run on stand-ins;
- with ARC3_PRIORITY_FORM unset, the gate scores by Franzen's priority_value again.

Usage: python -I tests/dprime_integration.py <ARC3-Inference checkout>
       (Franzen's 10882e3 = Duck 7652836 + the notebook's patch; needs `requests`)
"""
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def block(path, start, end):
    """The patched lines from the one starting with `start` up to (not including) `end`, dedented."""
    lines = open(path, encoding="utf-8").read().split("\n")
    i = next(k for k, l in enumerate(lines) if l.strip().startswith(start))
    j = next(k for k in range(i + 1, len(lines)) if lines[k].strip().startswith(end))
    return textwrap.dedent("\n".join(lines[i:j]))


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    tree = os.path.join(tempfile.mkdtemp(prefix="dprime_tree_"), "ARC3-Inference")
    shutil.copytree(os.path.abspath(sys.argv[1]), tree)
    apply = [sys.executable, "-I", os.path.join(ROOT, "harness", "apply_twin.py"), tree]
    first = subprocess.run(apply, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    print(first.stdout.strip().splitlines()[-1])
    second = subprocess.run(apply, capture_output=True, text=True)
    assert second.returncode == 0 and "nothing new" in second.stdout, second.stdout + second.stderr
    print("re-apply:", second.stdout.strip().splitlines()[-1])

    os.environ.update(ARC3_PRIORITY_FORM="dprime", ARC3_PRIORITY_REFRESH_QUEUE="1", ARC3_PRIORITY_TAIL_FADE="1",
                      ARC3_PRIORITY_TAIL_FADE_FRACTION="0.4", ARC3_PRIORITY_SCORE_NORMALIZATION="1",
                      ARC3_PRIORITY_TAIL_LOOKUP="remaining", ARC3_MAX_ACTIVE_STREAMS="10")
    sys.path.insert(0, tree)
    from inference.agent import tool_agent as ta
    from inference.agent.priority_scheduler import PrioritySnapshot
    from inference.utils.priority_dprime import dprime_priority

    # 1. the real gate, one slot, three waiters
    gate = ta._PriorityGate(1)
    t0 = time.monotonic()
    gate.configure_clock(t0, t0 + 3600.0)
    snaps = {"fresh": PrioritySnapshot(1, 0, 0.0, 1.0, 7),
             "deep": PrioritySnapshot(3, 20, 30000.0, 1.0, 7, tokens_per_level=25000.0),
             "stuck": PrioritySnapshot(1, 90, 150000.0, 1.0, 7)}
    now = time.monotonic()
    for name, s in snaps.items():  # an hour left of a 40% (24-min) fade window: phi = 1
        assert gate._snapshot_priority(s, now) == dprime_priority(s, 1.0), name
    expected = sorted(snaps, key=lambda n: -dprime_priority(snaps[n], 1.0))
    assert expected == ["deep", "fresh", "stuck"], expected
    gate.acquire(1)
    order = []

    def wait(name):
        gate.acquire(dprime_priority(snaps[name], None), snaps[name])
        order.append(name)
        gate.release()

    threads = []
    for name in ("stuck", "fresh", "deep"):  # arrival order is the reverse of D' order
        threads.append(threading.Thread(target=wait, args=(name,)))
        threads[-1].start()
        time.sleep(0.05)
    gate.release()
    for th in threads:
        th.join(5)
    assert order == expected, order
    print("gate: admitted", order, "| priorities", {n: dprime_priority(s, 1.0) for n, s in snaps.items()})

    # 2. the real _maybe_handover on a stand-in agent
    calls = []

    class RecordingGate(ta._PriorityGate):
        def handover(self, priority, snapshot=None):
            calls.append((priority, snapshot))

    ta._PRIORITY_GATE = RecordingGate(10)
    ta._PRIORITY_GATE.configure_clock(t0, t0 + 3600.0)
    agent = types.SimpleNamespace(
        _context_was_trimmed=True, _last_step_summary={"level": 3, "end_action_num": 50},
        _actions_at_level_start=30, _session_generated_tokens=100000, _tokens_at_level_start=70000,
        _dprime_cleared=2, _dprime_cleared_tokens=50000, _priority_total_levels=lambda: 7,
        _in_endgame=lambda: False)
    ta.ToolAgent._maybe_handover(agent)
    (priority, snap), = calls
    assert (snap.level, snap.actions, snap.tokens, snap.total_levels, snap.tokens_per_level) == (3, 20, 30000, 7, 25000.0), snap
    assert priority == dprime_priority(snap, 1.0) and agent._priority_current == priority, (priority, agent)
    print("handover: snapshot", snap, "->", priority)

    # 3. the patched level-up bookkeeping (tool_agent) on a stand-in
    agent_py = os.path.join(tree, "inference", "agent", "tool_agent.py")
    code = block(agent_py, "# priority_dprime: D''s p", "self._actions_at_level_start")
    me = types.SimpleNamespace(_session_generated_tokens=80000, _tokens_at_level_start=30000)
    for _ in range(2):  # the same transition seen twice counts once
        exec(code, {"max": max, "int": int, "getattr": getattr}, {"self": me, "summary": {"level": 2}})
    me._tokens_at_level_start, me._session_generated_tokens = 80000, 100000
    exec(code, {"max": max, "int": int, "getattr": getattr}, {"self": me, "summary": {"level": 3}})
    assert (me._dprime_cleared, me._dprime_cleared_tokens, me._dprime_last_cleared) == (2, 70000, 2), vars(me)
    print("bookkeeping: 2 clears, p =", me._dprime_cleared_tokens / me._dprime_cleared)

    # 4. the patched solver admission block on a stand-in
    solver_py = os.path.join(tree, "inference", "framework", "solver.py")
    code = block(solver_py, "if os.environ.get(\"ARC3_PRIORITY_FORM\"", "try:")
    acquired = []
    fake_gate = types.SimpleNamespace(acquire=lambda priority, snapshot=None: acquired.append((priority, snapshot)))
    session = types.SimpleNamespace(game=types.SimpleNamespace(number_of_levels=8), _queue_position=lambda: 300)
    env = {"os": os, "gate": fake_gate, "self": session, "_PRIORITY_UNTRIMMED_BASE": 2_000_000}
    exec(code, dict(env))
    (priority, snap), = acquired
    assert (snap.level, snap.actions, snap.tokens, snap.total_levels) == (1, 0, 0.0, 8) and priority == dprime_priority(snap, None)
    os.environ["ARC3_PRIORITY_FORM"] = ""
    exec(code, dict(env))
    assert acquired[-1] == (2_000_000 - 300, None), acquired[-1]
    print("solver: D' admission", priority, "| off: untrimmed band", acquired[-1][0])

    # 5. off again: Franzen's priority_value scores the gate
    s = snaps["deep"]
    franzen = gate._snapshot_priority(s, time.monotonic())
    assert franzen != dprime_priority(s, 1.0) and franzen >= 1
    print("off: Franzen's priority for the deep game", franzen)
    print("PASS")


if __name__ == "__main__":
    main()
