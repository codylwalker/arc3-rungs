"""priority_dprime against the published D' definition, written out again by hand here, plus the
properties the definition implies.

Run: python -I tests/test_priority_dprime.py
"""
import os
import sys
import unittest
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import priority_dprime as d  # noqa: E402


def reference(l, N, a, t, p, phi):
    """Affectify's text, transcribed independently of the module."""
    N = min(10, max(6, N))
    norm = 55 / (N * (N + 1) / 2)
    A = (1 + 0.5 * (l - 1)) * norm * (300 / (300 + a)) ** 2.5
    M = 1.0 if p is None else min(4, max(0.25, (30000 / p) ** 0.4))
    T = 225000 * (min(2, max(0.5, (30000 if p is None else p) / 30000))) ** 0.5
    C = 0.1 * max(0.1, 1 - a / 115) + 0.9 * max(0.1, 1 - t / T)
    rem = N - l
    B = 16 if rem >= 3 else {2: 14, 1: 10}.get(rem, 0)
    return A * M * C + B * phi


CASES = [  # level, N, actions, tokens, p, phi
    (1, 7, 0, 0, None, 1.0),
    (3, 7, 20, 30000, 25000, 1.0),
    (1, 7, 40, 59000, None, 1.0),
    (6, 7, 10, 5000, 40000, 0.5),
    (7, 7, 5, 1000, 20000, 1.0),
    (2, 12, 300, 400000, 90000, 0.0),
    (4, 4, 0, 0, None, 1.0),
    (5, 9, 200, 250000, 5000, 0.3),
]

Snap = namedtuple("Snap", "level actions tokens cost_multiplier total_levels tokens_per_level")


class Formula(unittest.TestCase):
    def test_matches_the_published_definition(self):
        for l, N, a, t, p, phi in CASES:
            with self.subTest(case=(l, N, a, t, p, phi)):
                self.assertAlmostEqual(d.dprime_value(l, a, t, N, p, phi), reference(l, N, a, t, p, phi), places=9)

    def test_hand_values(self):
        # a fresh 7-level game: A = 55/28, M = C = 1, B = 16
        self.assertAlmostEqual(d.dprime_value(1, 0, 0, 7, None, 1.0), 55 / 28 + 16, places=9)
        # the last level has no room to grow, so the fade does not matter
        self.assertAlmostEqual(d.dprime_value(7, 0, 0, 7, None, 1.0), d.dprime_value(7, 0, 0, 7, None, 0.0))
        # missing level count = 10, as the harness falls back to
        self.assertEqual(d.dprime_value(2, 3, 100, None, None, 1.0), d.dprime_value(2, 3, 100, 10, None, 1.0))

    def test_properties(self):
        base = d.dprime_value(3, 20, 30000, 8, 30000, 1.0)
        self.assertLess(d.dprime_value(3, 60, 30000, 8, 30000, 1.0), base)       # more actions
        self.assertLess(d.dprime_value(3, 20, 90000, 8, 30000, 1.0), base)       # more tokens
        self.assertGreater(d.dprime_value(4, 20, 30000, 8, 30000, 1.0), base)    # deeper, same room
        self.assertGreater(d.dprime_value(3, 20, 30000, 8, 15000, 1.0), base)    # faster pace
        self.assertLess(d.dprime_value(3, 20, 30000, 8, 30000, 0.0), base - 15)  # fade removes B=16
        # the patience floor: an exhausted level keeps 0.1 of A*M, never zero
        self.assertGreater(d.dprime_value(1, 10 ** 6, 10 ** 9, 6, None, 0.0), 0.0)

    def test_priority_scale_and_snapshot(self):
        snap = Snap(3, 20, 30000.0, 1.0, 7, 25000.0)
        self.assertEqual(d.dprime_priority(snap, 1.0), round(100 * reference(3, 7, 20, 30000, 25000, 1.0)))
        old = namedtuple("Old", "level actions tokens cost_multiplier")(1, 0, 0.0, 1.0)  # no new fields
        self.assertEqual(d.dprime_priority(old, None), round(100 * reference(1, 10, 0, 0, None, 1.0)))
        self.assertEqual(d.dprime_priority(Snap(10, 10 ** 6, 10 ** 9, 1.0, 10, 10 ** 7), 0.0), 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
