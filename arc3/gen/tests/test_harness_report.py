"""gen/harness_report.py on a synthetic run dir in the harness's summary.txt / viewer_data format, a
synthetic set (metadata.json + cert.json) and a smoke_t1.json.

Run (from tools/arc3): .venv-arc3/Scripts/python.exe -I gen/tests/test_harness_report.py
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import harness_report as hr  # noqa: E402

GAMES = {  # gid: (mechanics, optimal per level, levels solved, actions per level, tokens, score)
    "gaaa-00000001": (["hazard"], [4, 6, 8], 3, [6, 9, 16], 30000, 60.0),
    "gbbb-00000002": (["ice", "keydoor"], [5, 7], 1, [10, 30], 40000, 20.0),
    "gccc-00000003": (["switch", "snake", "hazard"], [6, 8, 10], 0, [50], 50000, 0.0),
}


def write_set(d):
    for gid, (mech, opt, *_rest) in GAMES.items():
        id4, h = gid.split("-")
        p = os.path.join(d, id4, h)
        os.makedirs(p)
        with open(os.path.join(p, "metadata.json"), "w") as f:
            json.dump({"game_id": gid, "mechanics": mech, "baseline_actions": [-(-o * 3 // 2) for o in opt]}, f)
        with open(os.path.join(p, "cert.json"), "w") as f:
            json.dump({"game_id": gid, "levels": [{"level": i + 1, "optimal": o} for i, o in enumerate(opt)]}, f)


def write_run(d, scale=1.0):
    os.makedirs(os.path.join(d, "artifacts"))
    lines = ["benchmark: x", "passes:    1", "", "per-game (mean across passes):"]
    for gid, (mech, opt, done, acts, toks, score) in GAMES.items():
        done_s = min(len(opt), done + (1 if scale > 1 else 0))
        lines.append("  %s: score=%.2f, levels=%.1f/%d, actions=%d, tokens=%d" % (gid, score * scale, done_s, len(opt), sum(acts), toks))
        with open(os.path.join(d, "artifacts", gid + "_p0_viewer_data.json"), "w") as f:
            json.dump({"game_id": gid, "levels_completed": done_s, "total_levels": len(opt), "actions_per_level": acts, "final_score": score}, f)
    with open(os.path.join(d, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


class Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hr_")
        self.set = os.path.join(self.tmp, "set")
        self.run = os.path.join(self.tmp, "run")
        write_set(self.set)
        write_run(self.run)
        self.t1 = os.path.join(self.tmp, "smoke_t1.json")
        with open(self.t1, "w") as f:
            json.dump({"per_game": {"gaaa": {"best_levels": 1}, "gbbb": {"best_levels": 0}}}, f)

    def test_report(self):
        out = os.path.join(self.tmp, "r.json")
        self.assertEqual(hr.main([self.run, "--set", self.set, "--t1", self.t1, "--json", out]), 0)
        with open(out) as f:
            r = json.load(f)
        g = r["games"]["gaaa-00000001"]
        self.assertEqual(g["levels"], 3.0)
        self.assertEqual(g["actions_over_optimal"], 1.67)  # 6/4, 9/6, 16/8 -> 1.5, 1.5, 2.0
        self.assertEqual(g["t1_best_level"], 1)
        self.assertEqual(r["games"]["gbbb-00000002"]["actions_over_optimal"], 2.0)  # only the solved level counts: 10/5
        self.assertIsNone(r["games"]["gccc-00000003"]["actions_over_optimal"])
        self.assertEqual(r["games"]["gccc-00000003"]["tokens_per_level_solved"], None)
        hz = r["by_mechanic"]["hazard"]
        self.assertEqual((hz["games"], hz["levels_solved"], hz["levels_total"]), (2, 3.0, 6))
        self.assertEqual(hz["solve_rate_by_index"], {"1": 0.5, "2": 0.5, "3+": 0.5})
        self.assertEqual(r["by_mechanic_count"]["1 mechanics"]["solve_rate"], 1.0)
        self.assertEqual(r["by_mechanic_count"]["3 mechanics"]["solve_rate"], 0.0)
        self.assertEqual(r["all"]["all"]["levels_total"], 8)

    def test_paired(self):
        b = os.path.join(self.tmp, "run-b")
        write_run(b, scale=1.5)
        out = os.path.join(self.tmp, "p.json")
        hr.main([self.run, "--b", b, "--set", self.set, "--json", out])
        with open(out) as f:
            r = json.load(f)
        self.assertEqual(r["paired"]["games"], 3)
        self.assertEqual(r["paired"]["delta_levels"]["mean"], round(2 / 3, 3))  # gbbb and gccc gain a level, gaaa is full
        self.assertEqual(r["paired_by_mechanic"]["hazard"]["delta_levels_mean"], 0.5)


if __name__ == "__main__":
    unittest.main()
