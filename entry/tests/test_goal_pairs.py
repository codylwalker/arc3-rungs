"""tools/goal_pairs.py on synthetic runs in the harness's event format: a planted effect, a null, a failed control.

Run: python -I tests/test_goal_pairs.py
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import goal_pairs  # noqa: E402


def write_pass(d, name, gid, levels, chars):
    """One game-pass: its viewer data and one analysis event per level played, carrying the generated characters."""
    with open(os.path.join(d, name + "_viewer_data.json"), "w", encoding="utf-8") as f:
        json.dump({"game_id": gid, "levels_completed": levels, "total_levels": 6, "final_score": 0}, f)
    with open(os.path.join(d, name + "_events.jsonl"), "w", encoding="utf-8") as f:
        for k in range(1, min(levels + 1, 6) + 1):
            meta = "[MODEL RESPONSE META]\nreasoning_chars: %d\n" % chars
            f.write(json.dumps({"type": "analysis", "level": k, "transcript": meta}) + "\n")


def build(n_pairs, s_levels, c_levels, s_chars, c_chars, passes=3):
    """A set dir (manifest pairing) and a run dir: n_pairs pairs, each variant played `passes` times."""
    root = tempfile.mkdtemp(prefix="goalpairs_")
    set_dir, run_dir = os.path.join(root, "set"), os.path.join(root, "run")
    os.makedirs(set_dir)
    os.makedirs(run_dir)
    games = []
    for p in range(n_pairs):
        for variant, levels, chars, tag in (("signalled", s_levels, s_chars, "s"), ("consequence", c_levels, c_chars, "c")):
            gid = "g%s%02d-%08x" % (tag, p, p)
            games.append({"game_id": gid, "pair_seed": 1000 + p, "variant": variant, "goal": "keys"})
            for k in range(passes):
                write_pass(run_dir, "%s_p%d" % (gid.split("-")[0], k), gid, levels, chars)
    games.append({"game_id": "gzzz-00000000", "seed": 5})  # an unpaired game: ignored
    with open(os.path.join(set_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"games": games}, f)
    return set_dir, run_dir


def run(set_dir, run_dir):
    return goal_pairs.score(goal_pairs.rows_by_pair(run_dir, goal_pairs.load_pairs(set_dir)), reps=500, seed=3)


class GoalPairs(unittest.TestCase):
    def test_planted_effect_reads_as_hits(self):
        res = run(*build(4, 6, 3, 1000, 2000))
        self.assertEqual(res["pairs"], 4)
        self.assertEqual(res["P-G5"]["verdict"], "OK")
        self.assertEqual(res["P-G1"]["verdict"], "HIT")
        self.assertAlmostEqual(res["P-G1"]["mean_levels_diff"], 3.0)
        self.assertEqual(res["P-G2"]["verdict"], "HIT")
        self.assertAlmostEqual(res["P-G2"]["ratio_consequence_over_signalled"], 2.0)
        self.assertEqual(res["P-G4"]["verdict"], "HIT")  # 4 of 4 pairs, p = 1/16
        self.assertAlmostEqual(res["P-G4"]["sign_p"], 1 / 16)
        # level 1 is cleared by both variants here, so the level-1 gap (0) is below the later levels' (0.6): MISS
        self.assertEqual(res["P-G3"]["verdict"], "MISS")
        self.assertAlmostEqual(res["P-G3"]["levels2_6_diff"], 0.6)

    def test_null_reads_as_misses(self):
        res = run(*build(4, 6, 6, 1000, 1000))
        self.assertEqual(res["P-G1"]["verdict"], "MISS")
        self.assertEqual(res["P-G2"]["verdict"], "MISS (not shown)")
        self.assertEqual(res["P-G4"]["verdict"], "MISS")

    def test_failed_control_makes_the_rest_uninformative(self):
        res = run(*build(4, 3, 1, 1000, 1500))
        self.assertEqual(res["P-G5"]["verdict"], "FAILED")
        for key in ("P-G1", "P-G2", "P-G3", "P-G4"):
            self.assertTrue(res[key]["verdict"].startswith("UNINFORMATIVE"), res[key])

    def test_by_goal_split_is_labelled_and_partitions_the_pairs(self):
        set_dir, run_dir = build(4, 6, 3, 1000, 2000)
        man = os.path.join(set_dir, "manifest.json")
        with open(man, encoding="utf-8") as f:
            data = json.load(f)
        for g in data["games"]:
            if g.get("pair_seed") in (1002, 1003):
                g["goal"] = "plates"
        with open(man, "w", encoding="utf-8") as f:
            json.dump(data, f)
        out = os.path.join(tempfile.mkdtemp(prefix="goalpairs_json_"), "res.json")
        argv = sys.argv
        sys.argv = ["goal_pairs.py", run_dir, "--set", set_dir, "--reps", "200", "--by-goal", "--json", out]
        try:
            self.assertEqual(goal_pairs.main(), 0)
        finally:
            sys.argv = argv
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
        split = res["exploratory_by_goal"]
        self.assertEqual(sorted(split), ["keys", "plates"])
        self.assertEqual(split["keys"]["pairs"] + split["plates"]["pairs"], res["pairs"])
        self.assertEqual(res["P-G1"]["verdict"], "HIT")  # the scored reading is untouched by the split

    def test_sign_test(self):
        self.assertAlmostEqual(goal_pairs.sign_p(4, 4), 1 / 16)
        self.assertAlmostEqual(goal_pairs.sign_p(0, 0), 1.0)
        self.assertAlmostEqual(goal_pairs.sign_p(3, 4), 5 / 16)


if __name__ == "__main__":
    unittest.main(verbosity=1)
