"""tools/arm_agreement.py on two small synthetic harness reports.

Run: python -I tests/test_arm_agreement.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import arm_agreement  # noqa: E402


def game(levels, score, ratio, tokens, mechanics=("ice",)):
    return {"mechanics": list(mechanics), "levels": levels, "total_levels": 6, "score": score,
            "actions_over_optimal": ratio, "tokens": tokens, "tokens_per_level_solved": tokens // max(1, int(levels))}


class ArmAgreement(unittest.TestCase):
    def test_disjoint_failures_and_rank_agreement(self):
        a = {"g1": game(6, 100.0, 1.1, 60000), "g2": game(4, 40.0, 2.5, 150000, ("switch",)),
             "g3": game(6, 80.0, 1.6, 90000), "g4": game(6, 90.0, 1.3, 70000)}
        b = {"g1": game(6, 100.0, 1.2, 65000), "g2": game(6, 70.0, 2.0, 120000, ("switch",)),
             "g3": game(5, 50.0, 1.9, 140000), "g4": game(6, 95.0, 1.2, 72000), "only_b": game(6, 100.0, 1.0, 1)}
        r = arm_agreement.agreement(a, b)
        self.assertEqual(r["games"], 4)  # paired games only
        self.assertEqual((r["unwon_a"], r["unwon_b"], r["unwon_both"]), (["g2"], ["g3"], []))
        self.assertEqual(r["fields"]["levels"]["identical"], 2)
        self.assertAlmostEqual(r["fields"]["tokens"]["rho"], 0.8, places=6)
        w = r["won_in_both"]
        self.assertEqual((w["games"], w["both_100"]), (2, 1))  # g1 and g4
        self.assertAlmostEqual(w["ratio_mean_a"], 1.2)


if __name__ == "__main__":
    unittest.main(verbosity=1)
