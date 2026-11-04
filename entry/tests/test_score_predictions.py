"""tools/score_predictions.py on a synthetic harness_report / analyze_run pair with hand-worked answers.

Run: python -I tests/test_score_predictions.py
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import score_predictions as S  # noqa: E402


def game(mechanics, solved, optimal, tokens=6000):
    return {"mechanics": mechanics, "n_mechanics": len(mechanics), "levels": float(solved), "total_levels": 6,
            "actions_per_level": [2 * o for o in optimal[:solved]] + [5] * (6 - solved), "optimal": optimal,
            "tokens": tokens, "tokens_per_level_solved": tokens // max(1, solved), "score": 100.0 * solved / 6,
            "t1_best_level": 0}


class ScorePredictions(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        rising = [2, 3, 4, 5, 6, 7]
        self.a = {"g1": game(["click_move"], 6, rising), "g2": game(["hazard"], 5, [3, 4, 5, 6, 7, 8]),
                  "g3": game(["ice", "keydoor", "switch", "wrap"], 1, rising),
                  "g4": game(["hazard", "keydoor", "pushable", "switch"], 2, rising)}
        an = [{"game": k, "gen_chars_by_level": {str(i + 1): 100 * (i + 1) for i in range(int(g["levels"]))}}
              for k, g in self.a.items()]
        self.ra, self.aa = self.write("a.json", {"games": self.a}), self.write("a_an.json", {"per_game": an})
        b = dict(self.a, g4=game(["hazard", "keydoor", "pushable", "switch"], 4, rising))
        self.rb = self.write("b.json", {"games": b})

    def tearDown(self):
        self.d.cleanup()

    def write(self, name, obj):
        path = os.path.join(self.d.name, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        return path

    def test_verdicts(self):
        games = S.load_games(self.ra, self.aa)
        r1 = S.p1(games)  # 1 mechanic: 11/12; 4 mechanics: 3/12
        self.assertAlmostEqual(r1["rate_by_count"][1], 11 / 12)
        self.assertAlmostEqual(r1["rate_by_count"][4], 0.25)
        self.assertEqual(r1["verdict"], "HIT")
        r2 = S.p2(games)  # A: hazard 7/12, ice 1/6, wrap 1/6; B: keydoor 3/12, pushable 2/6, switch 3/12
        self.assertAlmostEqual(r2["A"]["hazard"], 7 / 12)
        self.assertIsNone(r2["A"]["oneway"])
        self.assertAlmostEqual(r2["mean_A_minus_mean_B"], (7 / 12 + 2 / 6) / 3 - (0.25 + 1 / 3 + 0.25) / 3)
        r3 = S.p3(games)  # every solved level took 2x optimal: in the band, no length trend (rho undefined)
        self.assertAlmostEqual(r3["pooled_ratio"], 2.0)
        self.assertEqual((r3["verdict_band"], r3["verdict_longer"], r3["rho_optimal_vs_ratio"]), ("HIT", "MISS", None))
        r4 = S.p4(games)  # chars rise with optimal inside g1 and g2 (the games with >= 3 solved levels)
        self.assertAlmostEqual(r4["within_game_rho_optimal_vs_chars"], 1.0)
        self.assertEqual(r4["median_chars_by_level_index"]["1"], 100)
        self.assertAlmostEqual(r4["level1_over_later_levels_median"], 100 / 350.0)
        r6 = S.p6(games)
        self.assertTrue(r6["verdict"].startswith("UNTESTABLE"))
        self.assertEqual(r6["first_level_clears"], "4 of 4")

    def test_paired_twin_arm(self):
        r5 = S.p5(S.load_games(self.ra), S.load_games(self.rb))  # only g4 changes: 2 -> 4 levels
        self.assertEqual(r5["pairs"], 4)
        self.assertAlmostEqual(r5["levels_b_minus_a"], 0.5)
        self.assertAlmostEqual(r5["levels_diff_contact"], 2 / 3)  # g2, g3 (keydoor), g4
        self.assertAlmostEqual(r5["levels_diff_non_contact"], 1.0)  # g3, g4
        self.assertEqual(r5["verdict_shape"], "MISS")  # the contact gain does not exceed the snake/switch gain

    def test_spearman_ties(self):
        self.assertAlmostEqual(S.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(S.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(S.spearman([1, 1, 1, 1], [1, 2, 3, 4]))
        self.assertAlmostEqual(S.ranks([5, 1, 5, 2]), [3.5, 1.0, 3.5, 2.0])


if __name__ == "__main__":
    unittest.main(verbosity=1)
