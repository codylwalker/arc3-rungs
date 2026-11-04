"""twin_auto_certify on the toy maze: which functions it treats as world models, what it prints for
a right one, a wrong one, one with the model's own (grid, action) signature, one that abstains, and
nothing with too little logged. Run: python -I tests/test_auto_certify.py (standard library only).

The sandbox-level check (the [certify] line arriving in a snippet's stdout through the patched
harness) is tests/sandbox_integration.py's territory and needs a copy of the harness:
run it there, or let the Kaggle pre-check do it.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import twin_tools as tw  # noqa: E402
from test_twin_tools import LEVEL1, LEVEL2, MOVES, Maze, Runtime, maze_predict, two_cells, walk  # noqa: E402


def step(grid, action):
    """The model's own style: a grid of colour indices in, a full grid out (R = index of 'R')."""
    name = tw.twin_parse_action(action)[0]
    if name not in MOVES:
        return None
    red, floor = tw.TWIN_CHARS.index("R"), tw.TWIN_CHARS.index("B")
    out = [list(row) for row in grid]
    pos = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == red]
    if not pos:
        return None
    r, c = pos[0]
    dr, dc = MOVES[name]
    if grid[r + dr][c + dc] != floor:
        return out
    out[r][c] = floor
    out[r + dr][c + dc] = red
    return out


def abstain(frame, action):
    return None


def helper(x):
    return x


class Candidates(unittest.TestCase):
    def test_by_name_or_by_parameters(self):
        self.assertTrue(tw._tw_certify_candidate("predict", maze_predict))
        self.assertTrue(tw._tw_certify_candidate("my_model", maze_predict))
        self.assertTrue(tw._tw_certify_candidate("simulate_step", step))
        self.assertTrue(tw._tw_certify_candidate("f", lambda board, move: None))
        self.assertFalse(tw._tw_certify_candidate("helper", helper))
        self.assertFalse(tw._tw_certify_candidate("bfs", lambda start, goal: None))
        self.assertFalse(tw._tw_certify_candidate("predict", 3))


class Lines(unittest.TestCase):
    def setUp(self):
        tw._TW_LAST_IGNORE[0] = None
        self.rt = Runtime(Maze([LEVEL1, LEVEL2]))
        walk(self.rt, ["DOWN", "RIGHT", "RIGHT", "UP", "LEFT", "DOWN", "DOWN"])
        self.ns = {"predict": maze_predict, "wrong": two_cells, "step": step, "abstain": abstain, "helper": helper}

    def test_right_wrong_and_grid_signature(self):
        text = tw.twin_auto_certify(["helper", "predict", "wrong", "step", "abstain"], self.ns, self.rt.transitions)
        lines = text.splitlines()
        self.assertEqual(len(lines), 3, text)
        self.assertTrue(lines[0].startswith("[certify] predict reproduces all 7 logged transitions"), lines[0])
        self.assertIn("timer or bar cells ignored", lines[0])
        self.assertTrue(lines[1].startswith("[certify] wrong reproduces 0 of 7"), lines[1])
        self.assertIn("first mismatch at transitions[0] DOWN", lines[1])
        self.assertTrue(lines[2].startswith("[certify] step reproduces all 7"), lines[2])  # adapted to (grid, action)

    def test_order_follows_the_snippet(self):
        text = tw.twin_auto_certify(["wrong", "predict"], self.ns, self.rt.transitions)
        self.assertTrue(text.splitlines()[0].startswith("[certify] wrong"))

    def test_nothing_without_candidates_or_history(self):
        self.assertIsNone(tw.twin_auto_certify(["helper"], self.ns, self.rt.transitions))
        short = Runtime(Maze([LEVEL1, LEVEL2]))
        walk(short, ["DOWN", "UP"])
        self.assertIsNone(tw.twin_auto_certify(["predict"], self.ns, short.transitions))

    def test_a_raising_function_is_skipped(self):
        def predict(frame, action):
            raise RuntimeError("boom")
        self.assertIsNone(tw.twin_auto_certify(["predict"], {"predict": predict}, self.rt.transitions))

    def test_budget_stops_a_slow_model(self):
        import time

        def predict(frame, action):
            time.sleep(0.05)
            return maze_predict(frame, action)
        self.assertIsNone(tw.twin_auto_certify(["predict"], {"predict": predict}, self.rt.transitions, budget=0.01))


if __name__ == "__main__":
    unittest.main(verbosity=2)
