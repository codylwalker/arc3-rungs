"""tools/search_code.py on synthetic events files in the harness's v2 transcript format.

Run: python -I tests/test_search_code.py
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import search_code  # noqa: E402

# the prompt documents the tools and names a search; only tool-call blocks may count
SYSTEM = ("[SYSTEM PROMPT]\nUse Python to search or score candidate actions. `plan(predict, goal)` searches "
          "YOUR predict with a BFS; `run_plan(actions, predict)` executes it.\n[USER PROMPT]\nCurrent grid image:\n")


def call(code, result="ok"):
    return "[TOOL CALL: python]\n%s\n[TOOL RESULT: python]\n%s\n" % (code, result)


def write(d, game, steps):
    with open(os.path.join(d, "%s_p0_events.jsonl" % game), "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "initial", "level": 1}) + "\n")
        for step in steps:
            f.write(json.dumps({"type": "analysis", "level": 1, "transcript": SYSTEM + step}) + "\n")


class SearchCode(unittest.TestCase):
    def test_own_search_planner_calls_and_neither(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "own", [call("from collections import deque\ndef solve(start):\n    q = deque([start])")])
            write(d, "tool", [call("import heapq\nst = plan(predict, goal)"), call("out = run_plan(acts, predict)")])
            write(d, "none", [call("print(current_frame.segmentation)")])
            # the model's own `plan` is not the tool's; its BFS is a search
            write(d, "ownplan", [call("from collections import deque\ndef plan(fr):\n    return bfs(fr)\n"
                                      "seq = plan(current_frame)")])
            games = search_code.scan(d)
        self.assertEqual(sorted(games), ["none_p0", "own_p0", "ownplan_p0", "tool_p0"])
        self.assertEqual(games["own_p0"]["search"], {"deque", "def solve/search"})
        self.assertEqual(games["own_p0"]["planner_calls"], 0)
        self.assertEqual((games["tool_p0"]["search"], games["tool_p0"]["planner_calls"]), ({"heapq"}, 2))
        self.assertEqual((games["none_p0"]["search"], games["none_p0"]["planner_calls"]), (set(), 0))
        self.assertEqual(games["ownplan_p0"]["search"], {"deque", "bfs"})
        self.assertEqual((games["ownplan_p0"]["planner_calls"], games["ownplan_p0"]["defined"]), (0, ["plan"]))

    def test_the_prompt_alone_counts_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "g", ["[ANALYZER STATUS]\nok\n"])
            games = search_code.scan(d)
        self.assertEqual((games["g_p0"]["search"], games["g_p0"]["planner_calls"]), (set(), 0))


if __name__ == "__main__":
    unittest.main(verbosity=1)
