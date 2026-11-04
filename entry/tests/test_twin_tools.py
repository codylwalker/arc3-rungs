"""Unit tests for twin_tools on toy games, through a stand-in for the harness runtime.

Run: python -I tests/test_twin_tools.py   (standard library only)
"""
import os
import sys
import unittest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import twin_tools as tw  # noqa: E402

MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


# ---------------------------------------------------------------------------------------------
# a stand-in for the harness's frame views, history and action()


class FV:
    def __init__(self, rows, level, step):
        self.ascii = "\n".join(rows)
        self.level = level
        self.step = step
        self.shape = (len(rows), len(rows[0]))


class HE:
    def __init__(self, action, frame, result):
        self.action = action
        self.frame = frame
        self.result = dict(result)


class TV:
    def __init__(self, action, before, after, result):
        self.action = action
        self.before_frame = before
        self.after_frame = after
        self.frame = after
        self.result = dict(result)


class Runtime:
    """history / transitions / action() with the harness's shapes: a death appends the fatal
    board, then an automatic RESET entry with the restarted level."""

    def __init__(self, game):
        self.game = game
        self.history = [HE("", FV(game.rows(), game.level, game.steps), {})]

    def frame(self):
        return self.history[-1].frame

    @property
    def transitions(self):
        out = []
        for i, entry in enumerate(self.history):
            if entry.action:
                out.append(TV(entry.action, self.history[i - 1].frame if i else None, entry.frame, entry.result))
        return out

    def act(self, actions):
        last = {}
        for spec in actions:
            text = tw.twin_action_text(spec)
            result, shown = self.game.play(spec)
            self.history.append(HE(text, FV(shown, self.game.level_shown, self.game.steps), result))
            if result.get("game_over"):
                self.history.append(HE("RESET", FV(self.game.rows(), self.game.level, self.game.steps),
                                       {"automatic": True}))
            last = result
            if any(result.get(f) for f in ("game_over", "level_completed", "run_complete")):
                break
        return dict(last)


# ---------------------------------------------------------------------------------------------
# toy games


class Maze:
    """20x20: walls G, player R, goal Y, a budget bar of S cells in the top wall row that loses
    one cell per action (game over when empty). Optional hidden teleporter: (cell, destination)."""

    H = W = 20

    def __init__(self, layouts, budget=18, teleport=None):
        self.layouts = layouts
        self.budget_full = budget
        self.teleport = teleport
        self.level = 1
        self.level_shown = 1
        self.steps = 0
        self._load()

    def _load(self):
        player, goal, walls = self.layouts[self.level - 1]
        self.player, self.goal, self.walls = player, goal, set(walls)
        self.budget = self.budget_full

    def rows(self):
        grid = [["B"] * self.W for _ in range(self.H)]
        for r in range(self.H):
            grid[r][0] = grid[r][self.W - 1] = "G"
        for c in range(self.W):
            grid[0][c] = grid[self.H - 1][c] = "G"
        for r, c in self.walls:
            grid[r][c] = "G"
        for i in range(self.budget):
            grid[0][1 + i] = "S"
        grid[self.goal[0]][self.goal[1]] = "Y"
        grid[self.player[0]][self.player[1]] = "R"
        return ["".join(row) for row in grid]

    def blocked(self, r, c):
        return r <= 0 or c <= 0 or r >= self.H - 1 or c >= self.W - 1 or (r, c) in self.walls

    def play(self, spec):
        name = tw.twin_parse_action(spec)[0]
        self.steps += 1
        dr, dc = MOVES.get(name, (0, 0))
        r, c = self.player[0] + dr, self.player[1] + dc
        if not self.blocked(r, c):
            self.player = (r, c)
        if self.teleport and self.player == self.teleport[0]:
            self.player = self.teleport[1]
        self.budget -= 1
        self.level_shown = self.level
        if self.player == self.goal:
            if self.level == len(self.layouts):
                return {"run_complete": True, "level_completed": True}, self.rows()
            self.level += 1
            self.level_shown = self.level
            self._load()
            return {"level_completed": True}, self.rows()
        if self.budget <= 0:
            shown = self.rows()
            self._load()
            return {"game_over": True}, shown
        return {}, self.rows()


WALL = [(r, 10) for r in range(4, 16) if r != 8]
LEVEL1 = ((6, 5), (6, 14), WALL)
LEVEL2 = ((14, 3), (5, 16), [(r, 8) for r in range(3, 17) if r != 15])


def maze_predict(frame, action):
    """The model's hypothesis: R moves one cell unless the target is wall (G, or the bar in the
    wall row); nothing else changes (the bar is left to the ignore mask)."""
    name, _, _ = tw.twin_parse_action(action)
    if name not in MOVES:
        return None
    pos = [(r, row.index("R")) for r, row in enumerate(frame.rows) if "R" in row]
    if not pos:
        return None
    r, c = pos[0]
    dr, dc = MOVES[name]
    if frame.rows[r + dr][c + dc] in "GS":
        return {}
    return {(r, c): "B", (r + dr, c + dc): "R"}


def maze_goal_at(goal_cell):
    return lambda frame: frame.rows[goal_cell[0]][goal_cell[1]] == "R"


def bfs_distance(start, goal, walls, h=20, w=20):
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), d = queue.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in MOVES.values():
            n = (r + dr, c + dc)
            if 0 < n[0] < h - 1 and 0 < n[1] < w - 1 and n not in walls and n not in seen:
                seen.add(n)
                queue.append((n, d + 1))
    return None


class Clicker:
    """16x16 black board with three 2x2 P blocks; a click on a P block turns it Y. The level
    completes when no P is left."""

    def __init__(self):
        self.level = self.level_shown = 1
        self.steps = 0
        self.grid = [["B"] * 16 for _ in range(16)]
        for r0, c0 in ((5, 5), (5, 10), (10, 7)):
            for r in (r0, r0 + 1):
                for c in (c0, c0 + 1):
                    self.grid[r][c] = "P"

    def rows(self):
        return ["".join(row) for row in self.grid]

    def play(self, spec):
        name, r, c = tw.twin_parse_action(spec)
        self.steps += 1
        if name == "MOUSE" and self.grid[r][c] == "P":
            for y, x in flood(self.rows(), r, c):
                self.grid[y][x] = "Y"
        if not any("P" in row for row in self.grid):
            return {"run_complete": True, "level_completed": True}, self.rows()
        return {}, self.rows()


def flood(rows, r, c):
    colour = rows[r][c]
    seen = {(r, c)}
    stack = [(r, c)]
    while stack:
        y, x = stack.pop()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < len(rows) and 0 <= nx < len(rows[0]) and (ny, nx) not in seen and rows[ny][nx] == colour:
                seen.add((ny, nx))
                stack.append((ny, nx))
    return seen


def click_predict(frame, action):
    name, r, c = tw.twin_parse_action(action)
    if name != "MOUSE":
        return None
    if frame.rows[r][c] != "P":
        return {}
    return {cell: "Y" for cell in flood(frame.rows, r, c)}


# ---------------------------------------------------------------------------------------------


def walk(rt, names):
    for name in names:
        rt.act([name])


class Basics(unittest.TestCase):
    def test_rows_actions(self):
        self.assertEqual(tw.twin_rows("ab\ncd\n"), ("ab", "cd"))
        self.assertEqual(tw.twin_rows([[0, 1], [15, 3]]), ("Ww", "pG"))
        self.assertEqual(tw.twin_parse_action("MOUSE(row=4, col=7)"), ("MOUSE", 4, 7))
        self.assertEqual(tw.twin_parse_action({"action": "mouse", "row": 1, "col": 2}), ("MOUSE", 1, 2))
        self.assertEqual(tw.twin_parse_action("left"), ("LEFT", None, None))
        self.assertEqual(tw.twin_action_spec("MOUSE(row=4, col=7)"), {"action": "MOUSE", "row": 4, "col": 7})
        self.assertEqual(tw.twin_action_text({"action": "MOUSE", "row": 3, "col": 9}), "MOUSE(row=3, col=9)")

    def test_keyer_ignores_masked_cells(self):
        key = tw._tw_keyer(frozenset({(0, 1), (0, 2), (1, 0)}))
        self.assertEqual(key(("abcd", "efgh")), key(("aXYd", "Zfgh")))
        self.assertNotEqual(key(("abcd", "efgh")), key(("abcX", "efgh")))

    def test_apply_delta_and_full(self):
        before = ("BBB", "BRB")
        self.assertEqual(tw._tw_apply(before, {(1, 1): "B", (0, 1): "R"}), ("BRB", "BBB"))
        self.assertEqual(tw._tw_apply(before, "BBB\nBBR"), ("BBB", "BBR"))
        self.assertIsNone(tw._tw_apply(before, None))
        with self.assertRaises(IndexError):
            tw._tw_apply(before, {(5, 0): "R"})


class CheckModel(unittest.TestCase):
    def setUp(self):
        tw._TW_LAST_IGNORE[0] = None  # the ignore carry is per sandbox process; here one process runs every test
        self.rt = Runtime(Maze([LEVEL1, LEVEL2]))
        walk(self.rt, ["DOWN", "RIGHT", "RIGHT", "UP", "LEFT", "DOWN", "DOWN"])

    def test_correct_model_matches_with_timer_masked(self):
        out = tw.twin_check_model(maze_predict, self.rt.transitions)
        self.assertEqual((out["checked"], out["matched"], out["mismatched"]), (7, 7, 0), out["summary"])
        self.assertGreater(out["ignored_cells"], 0)

    def test_strict_mode_shows_the_timer(self):
        out = tw.twin_check_model(maze_predict, self.rt.transitions, ignore=None)
        self.assertEqual(out["mismatched"], 7)
        self.assertEqual(out["first_mismatch"]["bbox"][0], 0)  # the budget bar's row

    def test_border_mode_also_hides_the_timer(self):
        out = tw.twin_check_model(maze_predict, self.rt.transitions, ignore="border")
        self.assertEqual(out["matched"], 7, out["summary"])

    def test_wrong_model_reports_first_mismatch(self):
        def two_cells(frame, action):
            d = maze_predict(frame, action)
            if not d:
                return d
            (r0, c0), (r1, c1) = [cell for cell, v in d.items() if v == "B"][0], \
                [cell for cell, v in d.items() if v == "R"][0]
            return {(r0, c0): "B", (2 * r1 - r0, 2 * c1 - c0): "R"}

        out = tw.twin_check_model(two_cells, self.rt.transitions)
        self.assertEqual(out["mismatched"], 7)
        first = out["first_mismatch"]
        self.assertEqual((first["transition"], first["action"]), (0, "DOWN"))
        self.assertEqual(first["n_cells"], 2)
        self.assertIn("first mismatch at transitions[0] DOWN", out["summary"])

    def test_errors_are_reported_not_raised(self):
        def broken(frame, action):
            raise KeyError("player")

        out = tw.twin_check_model(broken, self.rt.transitions)
        self.assertEqual(out["errors"], 7)
        self.assertIn("KeyError", out["summary"])

    def test_abstain(self):
        out = tw.twin_check_model(lambda f, a: None, self.rt.transitions)
        self.assertEqual((out["checked"], out["abstained"]), (0, 7))

    def test_death_and_reset_are_skipped(self):
        rt = Runtime(Maze([LEVEL1, LEVEL2], budget=5))
        walk(rt, ["DOWN", "UP", "DOWN", "UP", "DOWN"])  # the fifth action empties the budget
        actions = [t.action for t in rt.transitions]
        self.assertEqual(actions[-1], "RESET")
        out = tw.twin_check_model(maze_predict, rt.transitions)
        self.assertEqual(out["checked"], 4, out["summary"])  # four live moves; death and RESET skipped
        self.assertEqual(out["mismatched"], 0, out["summary"])


class CheckGoal(unittest.TestCase):
    def test_goal_false_on_logged_boards(self):
        rt = Runtime(Maze([LEVEL1, LEVEL2]))
        walk(rt, ["DOWN", "RIGHT", "UP"])
        out = tw.twin_check_goal(maze_goal_at(LEVEL1[1]), rt.history)
        self.assertTrue(out["ok"])
        self.assertEqual(out["checked"], 4)

    def test_loose_goal_is_flagged(self):
        rt = Runtime(Maze([LEVEL1, LEVEL2]))
        walk(rt, ["DOWN", "RIGHT"])
        out = tw.twin_check_goal(lambda f: "R" in f.ascii, rt.history)
        self.assertFalse(out["ok"])
        self.assertEqual(out["true_on"], [0, 1, 2])


class Plan(unittest.TestCase):
    def test_shortest_path_matches_bfs(self):
        rt = Runtime(Maze([LEVEL1, LEVEL2]))
        out = tw.twin_plan(maze_predict, maze_goal_at(LEVEL1[1]), rt.frame(), transitions=rt.transitions)
        self.assertEqual(out["status"], "found", out)
        self.assertEqual(out["length"], bfs_distance(LEVEL1[0], LEVEL1[1], set(WALL)))
        self.assertEqual(out["length"], 13)

    def test_node_limit_and_no_path(self):
        rt = Runtime(Maze([LEVEL1, LEVEL2]))
        out = tw.twin_plan(maze_predict, maze_goal_at(LEVEL1[1]), rt.frame(), max_nodes=10)
        self.assertEqual(out["status"], "node_limit")
        boxed = Runtime(Maze([((6, 5), (6, 14), [(r, 10) for r in range(1, 19)]), LEVEL2]))
        out = tw.twin_plan(maze_predict, maze_goal_at((6, 14)), boxed.frame())
        self.assertEqual(out["status"], "exhausted")

    def test_predict_errors_stop_the_search(self):
        rt = Runtime(Maze([LEVEL1, LEVEL2]))

        def broken(frame, action):
            raise ValueError("bad hypothesis")

        out = tw.twin_plan(broken, maze_goal_at(LEVEL1[1]), rt.frame())
        # five errors at the root (UP DOWN LEFT RIGHT SPACE, the default moves), nothing to expand
        self.assertEqual(out["status"], "exhausted")
        self.assertEqual(out["errors"], 5)
        self.assertIn("ValueError", out["first_error"]["error"])

    def test_clicks_on_objects(self):
        rt = Runtime(Clicker())
        out = tw.twin_plan(click_predict, lambda f: "P" not in f.ascii, rt.frame(), moves=[], clicks="objects")
        self.assertEqual((out["status"], out["length"]), ("found", 3), out)
        self.assertTrue(all(a["action"] == "MOUSE" for a in out["actions"]))


class RunPlan(unittest.TestCase):
    def setUp(self):
        tw._TW_LAST_IGNORE[0] = None

    def test_explicit_ignore_carries_to_run_plan(self):
        # check_model told to ignore the teleporter's two cells; run_plan says nothing and inherits it
        warm = ["DOWN", "UP", "RIGHT", "LEFT"]
        rt0 = Runtime(Maze([LEVEL1, LEVEL2]))
        walk(rt0, warm)
        planned = tw.twin_plan(maze_predict, maze_goal_at(LEVEL1[1]), rt0.frame(), transitions=rt0.transitions)
        cells = [LEVEL1[0]]
        for spec in planned["actions"]:
            dr, dc = MOVES[spec]
            cells.append((cells[-1][0] + dr, cells[-1][1] + dc))
        rt = Runtime(Maze([LEVEL1, LEVEL2], teleport=(cells[4], (17, 17))))
        walk(rt, warm)
        bar = [(0, c) for c in range(20)]
        tw.twin_check_model(maze_predict, rt.transitions, ignore=set(bar + [cells[4], (17, 17)]))
        out = tw.twin_run_plan(planned["actions"], maze_predict, rt.act, rt.frame, transitions=rt.transitions)
        self.assertNotEqual(out["stopped"], "mismatch", out)  # the surprise is in the carried ignore set
        tw._TW_LAST_IGNORE[0] = None

    def test_plan_then_run_completes_the_level(self):
        rt = Runtime(Maze([LEVEL1, LEVEL2]))
        walk(rt, ["DOWN", "UP", "RIGHT", "LEFT"])  # some history, so the timer mask exists
        planned = tw.twin_plan(maze_predict, maze_goal_at(LEVEL1[1]), rt.frame(), transitions=rt.transitions)
        self.assertEqual(planned["length"], 13)
        # the budget bar has 14 cells left after 4 actions; the plan needs 13
        out = tw.twin_run_plan(planned["actions"], maze_predict, rt.act, rt.frame, transitions=rt.transitions)
        self.assertEqual((out["stopped"], out["executed"]), ("level_completed", 13), out)
        self.assertEqual(rt.game.level, 2)

    def test_first_surprise_stops_the_plan(self):
        # plan on the maze the model believes in, then put a teleporter it does not know about
        # on the fourth cell of that route
        warm = ["DOWN", "UP", "RIGHT", "LEFT"]
        rt0 = Runtime(Maze([LEVEL1, LEVEL2]))
        walk(rt0, warm)
        planned = tw.twin_plan(maze_predict, maze_goal_at(LEVEL1[1]), rt0.frame(), transitions=rt0.transitions)
        cells = [LEVEL1[0]]
        for spec in planned["actions"]:
            dr, dc = MOVES[spec]
            cells.append((cells[-1][0] + dr, cells[-1][1] + dc))
        rt = Runtime(Maze([LEVEL1, LEVEL2], teleport=(cells[4], (17, 17))))
        walk(rt, warm)
        out = tw.twin_run_plan(planned["actions"], maze_predict, rt.act, rt.frame, transitions=rt.transitions)
        self.assertEqual(out["stopped"], "mismatch", out)
        self.assertEqual(out["executed"], 4)  # one surprised action, then the plan stops
        self.assertEqual(out["mismatch"]["n_cells"], 2)
        self.assertEqual(rt.game.player, (17, 17))

    def test_without_predict_it_runs_through(self):
        rt = Runtime(Maze([LEVEL1, LEVEL2]))
        out = tw.twin_run_plan(["DOWN", "DOWN"], None, rt.act, rt.frame)
        self.assertEqual((out["stopped"], out["executed"]), ("done", 2))


def two_cells(frame, action):
    """A rival hypothesis: the sprite moves two cells per arrow."""
    name = tw.twin_parse_action(action)[0]
    if name not in MOVES:
        return None
    r, c = [(r, row.index("R")) for r, row in enumerate(frame.rows) if "R" in row][0]
    dr, dc = MOVES[name]
    if frame.rows[r + 2 * dr][c + 2 * dc] in "GS" or frame.rows[r + dr][c + dc] in "GS":
        return {}
    return {(r, c): "B", (r + 2 * dr, c + 2 * dc): "R"}


def sideways(frame, action):
    """Another rival: LEFT and RIGHT are swapped."""
    swap = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
    name = tw.twin_parse_action(action)[0]
    return maze_predict(frame, swap.get(name, name))


class BestProbe(unittest.TestCase):
    def setUp(self):
        tw._TW_LAST_IGNORE[0] = None
        self.frame = Runtime(Maze([LEVEL1, LEVEL2])).frame()

    def test_two_rivals_one_bit(self):
        out = tw.twin_best_probe([maze_predict, two_cells], self.frame, moves=list(MOVES))
        self.assertEqual(out["bits"], 1.0)
        self.assertEqual(sorted(sorted(g) for g in out["split"]), [["maze_predict"], ["two_cells"]])
        self.assertIn(out["action"], list(MOVES))

    def test_three_rivals_prefers_the_three_way_split(self):
        # UP/DOWN cannot tell maze_predict from sideways; LEFT/RIGHT split all three apart
        out = tw.twin_best_probe({"one": maze_predict, "two": two_cells, "swap": sideways}, self.frame,
                                 moves=list(MOVES))
        self.assertIn(out["action"], ("LEFT", "RIGHT"))
        self.assertAlmostEqual(out["bits"], 1.585, places=3)
        self.assertEqual(len(out["split"]), 3)

    def test_agreeing_hypotheses_give_no_probe(self):
        out = tw.twin_best_probe([maze_predict, lambda f, a: maze_predict(f, a)], self.frame, moves=list(MOVES))
        self.assertIsNone(out["action"])
        self.assertEqual(out["bits"], 0.0)
        self.assertIn("no single action", out["summary"])

    def test_weights_are_the_prior(self):
        out = tw.twin_best_probe([maze_predict, two_cells], self.frame, moves=list(MOVES), weights=[0.9, 0.1])
        self.assertAlmostEqual(out["bits"], 0.469, places=3)

    def test_a_raising_hypothesis_sits_out(self):
        def broken(frame, action):
            raise KeyError("sprite")

        out = tw.twin_best_probe([maze_predict, two_cells, broken], self.frame, moves=list(MOVES))
        self.assertEqual(out["bits"], 1.0)
        self.assertIn("broken", out["errors"])

    def test_needs_two(self):
        self.assertIsNone(tw.twin_best_probe([maze_predict], self.frame)["action"])

    def test_clicks_on_objects(self):
        frame = Runtime(Clicker()).frame()

        def toggles_all(f, a):
            name, r, c = tw.twin_parse_action(a)
            if name != "MOUSE" or f.rows[r][c] != "P":
                return {} if name == "MOUSE" else None
            return {(y, x): "Y" for y, row in enumerate(f.rows) for x, ch in enumerate(row) if ch == "P"}

        out = tw.twin_best_probe([click_predict, toggles_all], frame, moves=[], clicks="objects")
        self.assertEqual(out["action"]["action"], "MOUSE")
        self.assertEqual(out["bits"], 1.0)


class PromptExample(unittest.TestCase):
    def test_the_documented_predict_works(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, here)
        from twin_prompt import TWIN_TOOLS_PROMPT

        lines = TWIN_TOOLS_PROMPT.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("def predict("))
        end = next(i for i in range(start + 1, len(lines)) if not lines[i].startswith(" "))
        namespace = {"parse_action": tw.twin_parse_action}
        exec("\n".join(lines[start:end]), namespace)
        rt = Runtime(Maze([LEVEL1, LEVEL2]))
        walk(rt, ["DOWN", "RIGHT", "RIGHT", "UP", "LEFT", "DOWN", "DOWN"])
        out = tw.twin_check_model(namespace["predict"], rt.transitions)
        self.assertEqual((out["checked"], out["matched"]), (7, 7), out["summary"])
        planned = tw.twin_plan(namespace["predict"], maze_goal_at(LEVEL1[1]), Runtime(Maze([LEVEL1, LEVEL2])).frame())
        self.assertEqual(planned["length"], 13)


class AutoMask(unittest.TestCase):
    def test_bar_masked_moving_sprite_not(self):
        rt = Runtime(Maze([LEVEL1, LEVEL2]))
        walk(rt, ["DOWN", "RIGHT", "RIGHT", "UP", "LEFT", "DOWN"])
        mask = tw._tw_mask("auto", (20, 20), rt.transitions, 1)
        self.assertTrue(all((0, 1 + i) in mask for i in range(18)))  # every cell the bar ever held
        self.assertFalse(any(cell in mask for cell in ((6, 5), (7, 5), (7, 6), (7, 7), (6, 7))))

    def test_too_few_actions_means_strict(self):
        rt = Runtime(Maze([LEVEL1, LEVEL2]))
        walk(rt, ["DOWN", "RIGHT"])
        self.assertIsNone(tw._tw_mask("auto", (20, 20), rt.transitions, 1))

    @staticmethod
    def interior_bar(shrink_on, n=6, size=20):
        """n transitions on a 20x20 board: a sprite R walks right along row 15 every action; a bar of
        Y on row 10, cols 2-17, loses its rightmost cell on the actions in shrink_on (sb26's shape:
        a bar inside the board, not in the edge band)."""
        def board(sprite_col, bar_end):
            rows = ["B" * size for _ in range(size)]
            rows[10] = "BB" + "Y" * (bar_end - 2) + "B" * (size - bar_end)
            rows[15] = "B" * sprite_col + "R" + "B" * (size - sprite_col - 1)
            return rows
        out = []
        col, end = 1, 18
        for i in range(n):
            before = board(col, end)
            col += 1
            if i in shrink_on:
                end -= 1
            out.append(TV("RIGHT", FV(before, 1, i), FV(board(col, end), 1, i + 1), {}))
        return out

    def test_interior_bar_masked_whole_at_a_third_of_the_actions(self):
        mask = tw._tw_mask("auto", (20, 20), self.interior_bar({0, 3}), 1)  # 2 of 6 actions
        self.assertIsNotNone(mask)
        self.assertTrue(all((10, c) in mask for c in range(2, 18)), sorted(mask))  # the whole bar
        self.assertFalse(any(r != 10 for r, _ in mask), sorted(mask))  # nothing of the sprite's row

    def test_interior_bar_below_the_rate_is_not_masked(self):
        self.assertIsNone(tw._tw_mask("auto", (20, 20), self.interior_bar({0}), 1))  # 1 of 6

    def test_moving_sprite_alone_is_not_masked(self):
        self.assertIsNone(tw._tw_mask("auto", (20, 20), self.interior_bar(set()), 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
