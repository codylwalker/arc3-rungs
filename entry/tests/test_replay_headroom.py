"""replay_headroom on a synthetic run: a level won after a death and a RESET, a clean level, and a
level never won; the best of two passes; the public (env-dir) and generated (set) baseline sources."""
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

import replay_headroom as rh  # noqa: E402


def action(level, name, **flags):
    e = {"type": "action", "level": level, "action_name": name, "board_ascii": ""}
    e.update(flags)
    return e


def write_events(path, events):
    """As the harness writes them: `score` is the levels completed after the action, and `level`
    has already advanced on the completing action."""
    score = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "initial"}) + "\n")
        for e in events:
            e = dict(e)
            if e.get("level_completed") or e.get("done"):
                score += 1
                e["level"] = score + 1
            e["score"] = score
            f.write(json.dumps(e) + "\n")


class ReplayHeadroomTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="replay_headroom_")
        self.run = os.path.join(self.tmp, "run", "artifacts")
        os.makedirs(self.run)
        self.env = os.path.join(self.tmp, "env", "ab12", "deadbeef")
        os.makedirs(self.env)
        with open(os.path.join(self.env, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"game_id": "ab12-deadbeef", "baseline_actions": [6, 4, 10]}, f)
        # pass 0: level 1 won on the second attempt (3 actions, death, RESET, 3 actions), level 2 won
        # in 2, level 3 never won
        p0 = ([action(1, "ACTION1")] * 2 + [action(1, "ACTION1", game_over=True), action(1, "RESET")]
              + [action(1, "ACTION2")] * 2 + [action(1, "ACTION2", level_completed=True)]
              + [action(2, "ACTION3"), action(2, "ACTION3", level_completed=True)]
              + [action(3, "ACTION4")] * 5)
        write_events(os.path.join(self.run, "ab12-deadbeef_p0_events.jsonl"), p0)
        # pass 1: level 1 won in 12 actions, nothing more
        p1 = [action(1, "ACTION1")] * 11 + [action(1, "ACTION1", level_completed=True)]
        write_events(os.path.join(self.run, "ab12-deadbeef_p1_events.jsonl"), p1)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_levels_and_attempts(self):
        levels = rh.read_events(os.path.join(self.run, "ab12-deadbeef_p0_events.jsonl"))
        self.assertEqual(levels[1]["actions"], 7)   # RESET counted, as taaf counts it
        self.assertEqual(levels[1]["attempt"], 3)
        self.assertEqual(levels[1]["resets"], 1)
        self.assertEqual(levels[1]["game_overs"], 1)
        self.assertTrue(levels[1]["won"])
        self.assertEqual(levels[2]["attempt"], 2)
        self.assertFalse(levels[3]["won"])
        self.assertEqual(levels[3]["attempt"], 0)

    def test_scores_and_best_pass(self):
        info = rh.load_baselines([os.path.join(self.tmp, "env")], [])
        best, games, checks = rh.analyse([os.path.join(self.tmp, "run")], info)
        self.assertEqual(len(games["ab12-deadbeef"]), 2)
        r = best["ab12-deadbeef"]
        self.assertEqual(r["pass"], 0)
        self.assertEqual(r["won"], 2)
        # now: level 1 (6/7)^2*100 = 73.47, level 2 100 (4/2 capped 115), level 3 0;
        # weights 1,2,3: (73.47 + 230) / 6 = 50.58, cap at (1+2)/6*100 = 50 -> 50
        self.assertAlmostEqual(r["now"], 50.0, places=2)
        self.assertAlmostEqual(r["attempt"], 50.0, places=2)   # both capped by the unwon level
        self.assertAlmostEqual(r["baseline"], 50.0, places=2)
        self.assertEqual(r["levels_with_resets"], 1)
        self.assertEqual(r["deaths"], 1)
        self.assertEqual([round(x, 3) for x in r["won_now_ratio"]], [round(7 / 6, 3), 0.5])
        self.assertEqual([round(x, 3) for x in r["won_at_attempt_ratio"]], [0.5, 0.5])
        self.assertEqual(checks, [])

    def test_uncapped_scores(self):
        # with level 3 won too, the cap lifts and the attempt column beats now
        p = ([action(1, "ACTION1")] * 8 + [action(1, "ACTION1", game_over=True), action(1, "RESET")]
             + [action(1, "ACTION2")] * 5 + [action(1, "ACTION2", level_completed=True)]
             + [action(2, "ACTION3")] * 3 + [action(2, "ACTION3", level_completed=True)]
             + [action(3, "ACTION4")] * 9 + [action(3, "ACTION4", done=True)])
        write_events(os.path.join(self.run, "ab12-deadbeef_p2_events.jsonl"), p)
        info = rh.load_baselines([os.path.join(self.tmp, "env")], [])
        best, _, _ = rh.analyse([os.path.join(self.tmp, "run")], info)
        r = best["ab12-deadbeef"]
        self.assertEqual(r["pass"], 2)
        # level 1: now 16 actions (6/16)^2 = 14.06; attempt 6 -> 100; level 2: 4 -> 100; level 3: 10 -> 100
        self.assertAlmostEqual(r["now"], (14.0625 + 200 + 300) / 6, places=2)
        self.assertAlmostEqual(r["attempt"], 100.0, places=2)
        self.assertAlmostEqual(r["baseline"], 100.0, places=2)

    def test_generated_set_adds_optimum_columns(self):
        s = os.path.join(self.tmp, "set", "ab12", "deadbeef")
        os.makedirs(s)
        with open(os.path.join(s, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"game_id": "ab12-deadbeef", "baseline_actions": [6, 4, 10]}, f)
        with open(os.path.join(s, "cert.json"), "w", encoding="utf-8") as f:
            json.dump({"levels": [{"optimal": 3}, {"optimal": 2}, {"optimal": 7}]}, f)
        info = rh.load_baselines([], [os.path.join(self.tmp, "set")])
        best, _, _ = rh.analyse([os.path.join(self.tmp, "run")], info)
        r = best["ab12-deadbeef"]
        self.assertIn("opt_now", r)
        # level 1 attempt 3 = the optimum -> 100; level 2 attempt 2 -> 100; capped by level 3 at 50
        self.assertAlmostEqual(r["opt_attempt"], 50.0, places=2)
        self.assertLess(r["opt_now"], r["opt_attempt"])


if __name__ == "__main__":
    unittest.main()
