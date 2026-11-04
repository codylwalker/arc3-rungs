"""meters on synthetic grids: a shrinking bar is a meter with a rate and an estimate, a growing one
has no estimate, a sprite alone is not, a bar below the rate is not; far changes from a hand-made
object diff, with and without a moved object or a click. Run: python -I tests/test_meters.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meters  # noqa: E402


class Frame:
    def __init__(self, grid, level=1):
        self.grid = tuple(tuple(r) for r in grid)
        self.level = level


class Entry:
    def __init__(self, action, grid, level=1, result=None):
        self.action = action
        self.frame = Frame(grid, level)
        self.result = result or {}


def board(size=20, sprite=(10, 2), bar_end=19, bar_row=19, bar_colour=9, grow=False):
    g = [[0] * size for _ in range(size)]
    g[sprite[0]][sprite[1]] = 5
    for c in range(1, bar_end):
        g[bar_row][c] = bar_colour
    return g


def history(shrink_on, n=6, grow=False):
    """n RIGHT moves of the sprite; the bar loses (or gains) its rightmost cell on the actions in shrink_on."""
    end = 19 if not grow else 8
    out = [Entry("", board(sprite=(10, 2), bar_end=end))]
    col = 2
    for i in range(n):
        col += 1
        if i in shrink_on:
            end += 1 if grow else -1
        out.append(Entry("RIGHT", board(sprite=(10, col), bar_end=end)))
    return out


class Meters(unittest.TestCase):
    def test_shrinking_bar_is_a_meter_with_an_estimate(self):
        found = meters.find_meters(meters.level_transitions(history({0, 2, 4}), 1))
        self.assertEqual(len(found), 1, found)
        m = found[0]
        self.assertEqual((m["colour"], m["direction"], m["changes"], m["actions"]), (9, "down", 3, 6))
        self.assertEqual((m["rows"], m["start"], m["now"]), ((19, 19), 18, 15))
        self.assertAlmostEqual(m["per_action"], -0.5)
        self.assertEqual(m["left"], 30)
        line = meters.meter_line(m)
        self.assertTrue(line.startswith("meter: colour 9 along row 19 (cols 16-18): 18 -> 15 cells, -0.5 per action, about 30 actions left"), line)

    def test_growing_bar_has_no_estimate(self):
        found = meters.find_meters(meters.level_transitions(history({0, 1, 2, 3}, grow=True), 1))
        self.assertEqual(len(found), 1, found)
        self.assertEqual(found[0]["direction"], "up")
        self.assertIsNone(found[0]["left"])
        self.assertNotIn("left", meters.meter_line(found[0]))

    def test_sprite_alone_and_slow_bar_are_not_meters(self):
        self.assertEqual(meters.find_meters(meters.level_transitions(history(set()), 1)), [])
        self.assertEqual(meters.find_meters(meters.level_transitions(history({0}), 1)), [])  # 1 of 6

    def test_the_bars_background_is_not_a_second_meter(self):
        # arm M (2026-10-28): the HUD's own background colour fills in behind the shrinking bar, so it grows
        # one-for-one on the same strip and passed the one-way rule; only the shrinking colour is the meter
        hist = history({0, 2, 4})
        for entry in hist:
            g = [list(r) for r in entry.frame.grid]
            for c in range(1, 19):
                if g[19][c] != 9:
                    g[19][c] = 3  # the strip's background, not the board's majority colour 0
            entry.frame = Frame(g)
        found = meters.find_meters(meters.level_transitions(hist, 1))
        self.assertEqual([m["colour"] for m in found], [9], found)

    def test_resets_and_other_levels_are_skipped(self):
        h = history({0, 2, 4})
        h.insert(3, Entry("RESET", board(), result={"automatic": True}))
        h.append(Entry("RIGHT", board(), level=2))
        trans = meters.level_transitions(h, 1)
        # the move INTO the RESET entry is skipped, the move after it is real, the level-2 move is not level 1
        self.assertEqual(len(trans), 6)
        self.assertEqual(len(meters.meters_lines(h, 1)), 1)
        self.assertEqual(meters.meters_lines(h, 2), [])

    def test_too_few_actions(self):
        self.assertEqual(meters.meters_lines(history({0, 1}, n=3), 1), [])


class Far(unittest.TestCase):
    def test_far_appeared_object_is_named_near_one_is_not(self):
        diff = {"changed_cell_count": 9,
                "moved": [{"color": 5, "pixels": 1, "from": [10, 2], "to": [10, 3]}],
                "appeared": [{"color": 7, "pixels": 4, "at": [40, 50]}, {"color": 7, "pixels": 1, "at": [11, 4]}],
                "disappeared": [], "changed_color": [], "resized": [], "rotated": []}
        line = meters.far_line(diff)
        self.assertEqual(line, "far: colour 7 appeared at (40, 50) 4px, 44 cells from the moved object", line)

    def test_click_origin(self):
        diff = {"changed_cell_count": 3, "moved": [], "rotated": [],
                "changed_color": [{"before_color": 2, "after_color": 3, "at": [30, 30], "pixels": 3}],
                "appeared": [], "disappeared": [], "resized": []}
        self.assertIsNone(meters.far_line(diff, origin=(31, 29)))
        line = meters.far_line(diff, origin=(5, 5))
        self.assertEqual(line, "far: colour 2->3 recoloured at (30, 30) 3px, 24 cells from the click", line)

    def test_a_resize_under_a_quarter_is_not_far(self):
        # arm M (2026-10-28): the bar and its background resize by a cell on every action, far from the move
        base = {"changed_cell_count": 2, "moved": [{"color": 5, "pixels": 1, "from": [10, 2], "to": [10, 3]}],
                "appeared": [], "disappeared": [], "changed_color": [], "rotated": []}
        tick = dict(base, resized=[{"color": 9, "before_pixels": 118, "after_pixels": 117, "at": [40, 50]}])
        self.assertIsNone(meters.far_line(tick))
        opened = dict(base, resized=[{"color": 9, "before_pixels": 12, "after_pixels": 4, "at": [40, 50]}])
        self.assertEqual(meters.far_line(opened), "far: colour 9 resized 12->4px at (40, 50), 44 cells from the moved object")

    def test_tiny_objects_are_not_far(self):
        # the D' + M demo (2026-10-29): most far entries on the public ten were 1-2 px sprite fragments and
        # bar ticks; an object under MIN_FAR_PIXELS (a resize under MIN_RESIZE_PIXELS) is not a consequence
        base = {"changed_cell_count": 2, "moved": [{"color": 5, "pixels": 1, "from": [10, 2], "to": [10, 3]}],
                "appeared": [], "disappeared": [], "changed_color": [], "resized": [], "rotated": []}
        tiny = dict(base, appeared=[{"color": 7, "pixels": 2, "at": [40, 50]}],
                    disappeared=[{"color": 8, "pixels": 1, "at": [50, 50]}],
                    changed_color=[{"before_color": 2, "after_color": 3, "at": [30, 30], "pixels": 2}],
                    resized=[{"color": 9, "before_pixels": 6, "after_pixels": 3, "at": [20, 50]}])
        self.assertIsNone(meters.far_line(tiny))
        kept = dict(base, appeared=[{"color": 7, "pixels": 3, "at": [40, 50]}])
        self.assertEqual(meters.far_line(kept), "far: colour 7 appeared at (40, 50) 3px, 45 cells from the moved object")

    def test_nothing_moved_and_no_click(self):
        diff = {"changed_cell_count": 2, "moved": [], "rotated": [], "changed_color": [],
                "appeared": [{"color": 1, "pixels": 4, "at": [3, 3]}], "disappeared": [], "resized": []}
        self.assertEqual(meters.far_line(diff), "far: colour 1 appeared at (3, 3) 4px (nothing moved)")

    def test_limit_and_more(self):
        diff = {"changed_cell_count": 9, "moved": [{"color": 5, "pixels": 1, "from": [0, 0], "to": [0, 1]}],
                "appeared": [{"color": c, "pixels": 4, "at": [50, 50 + c]} for c in range(5)],
                "disappeared": [], "changed_color": [], "resized": [], "rotated": []}
        line = meters.far_line(diff)
        self.assertTrue(line.endswith("; +2 more"), line)
        self.assertEqual(line.count("appeared"), 3)

    def test_action_note_uses_the_diff_fn(self):
        a = [[0] * 7 for _ in range(7)]
        b = [[0] * 7 for _ in range(7)]
        b[6][6] = 7
        calls = []

        def diff_fn(x, y):
            calls.append((x, y))
            return {"changed_cell_count": 1, "moved": [], "rotated": [], "changed_color": [], "resized": [],
                    "appeared": [{"color": 7, "pixels": 4, "at": [6, 6]}], "disappeared": []}
        self.assertIsNone(meters.action_note(a, a, diff_fn))
        # 6 cells from the click less the object's radius (2 for a 4px object)
        self.assertEqual(meters.action_note(a, b, diff_fn, origin=(0, 0)), "far: colour 7 appeared at (6, 6) 4px, 4 cells from the click")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
