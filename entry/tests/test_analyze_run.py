"""tools/analyze_run.py on a synthetic run directory in the harness's real transcript format.

The format (labelled blocks, the system prompt first, later steps repeating earlier history) is the one
seen in the first Kaggle run, 2026-10-22 v1.

Run: python -I tests/test_analyze_run.py
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import analyze_run  # noqa: E402

SYSTEM = """\
--- analysis_step=1 | action=1 | 18:01:13 | tool-agent ---
[SYSTEM PROMPT]
You are a coding agent solving a grid-based puzzle game.
- `check_model(predict)` replays predict over every logged transition of this level ...
- `plan(predict, goal)` searches YOUR predict ... - `run_plan(actions, predict)` executes ...
- `best_probe([predict_a, predict_b, ...])`: when several hypotheses fit ...
[USER PROMPT]
Current grid image:
[image attached]
"""

STEP1 = """\
[REASONING]
I will check my model of the mechanics first.
[ASSISTANT TOOL CALL: python]
id: call_1
arguments:
def predict(frame, action):
    return {}
r = check_model(predict)
print(r['summary'])
p = plan(predict, goal)
out = run_plan(p['actions'], predict)
print(out)
[TOOL RESULT: call_1]
predict reproduces 3 of 4 checked transitions; first mismatch at transitions[2] LEFT: 2 cells in rows 7-8
{'status': 'found', 'actions': ['UP'], 'length': 1}
{'executed': 1, 'stopped': 'level_completed', 'actions_run': ['UP']}
[certify] predict reproduces 3 of 4 logged transitions of this level; first mismatch at transitions[2] LEFT: 2 cells in rows 7-8, cols 3-3
"""

STEP2 = """\
[ASSISTANT TOOL CALL: python]
id: call_2
arguments:
print(best_probe([predict, other])['summary'])
[TOOL RESULT: call_2]
UP splits the 2 hypotheses into predict | other (1.00 bits)
"""


class AnalyzeRun(unittest.TestCase):
    def test_counts_only_the_models_calls_once(self):
        with tempfile.TemporaryDirectory() as d:
            art = os.path.join(d, "artifacts")
            os.makedirs(art)
            events = [
                {"type": "initial", "level": 1},
                {"type": "analysis", "transcript": SYSTEM + STEP1},
                {"type": "action", "game_over": False},
                {"type": "analysis", "transcript": SYSTEM + STEP1 + STEP2},  # history repeated
                {"type": "action", "game_over": True},
            ]
            with open(os.path.join(art, "ls20_p0_events.jsonl"), "w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")
            with open(os.path.join(art, "ls20_p0_viewer_data.json"), "w", encoding="utf-8") as f:
                json.dump({"game_id": "ls20-x", "levels_completed": 1, "total_levels": 7, "final_score": 9.5}, f)
            out = analyze_run.summarise(analyze_run.analyse(d))
        self.assertEqual(out["games"], 1)
        self.assertEqual(out["games_using_tools"], 1)
        self.assertEqual(out["calls"], {"check_model": 1, "check_goal": 0, "plan": 1, "run_plan": 1, "best_probe": 1})
        row = out["per_game"][0]
        self.assertEqual((row["game"], row["levels"], row["actions"], row["game_overs"]), ("ls20-x", 1, 2, 1))
        self.assertEqual(row["python_calls"], 2)
        self.assertEqual(row["check_model_match_rate"], 0.75)
        self.assertEqual((row["plan"], row["run_plan"], row["probes"]), ({"found": 1}, {"level_completed": 1}, 1))
        self.assertEqual((row["certify_lines"], row["certify_full"]), (1, 0))  # the unasked line, once

    def test_real_v2_labels_one_step_per_event(self):
        step = ("[THINKING]\nTest the mechanics.\n[ASSISTANT]\n\n[TOOL CALL: python]\n"
                "r = check_model(predict)\nprint(r['summary'])\n"
                "[TOOL RESULT: python]\npredict reproduces 4 of 4 checked transitions\n"
                "[certify] predict reproduces all 4 logged transitions of this level (18 timer or bar cells ignored)\n"
                "[ANALYZER STATUS]\nok\n")
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "g_p0_events.jsonl"), "w", encoding="utf-8") as f:
                for _ in range(2):  # two separate steps with identical code: both count
                    f.write(json.dumps({"type": "analysis", "transcript": SYSTEM + step}) + "\n")
            out = analyze_run.summarise(analyze_run.analyse(d))
        self.assertEqual(out["calls"]["check_model"], 2)
        self.assertEqual(out["per_game"][0]["python_calls"], 2)
        self.assertEqual(out["per_game"][0]["check_model_match_rate"], 1.0)
        self.assertEqual((out["per_game"][0]["certify_lines"], out["per_game"][0]["certify_full"]), (2, 2))

    def test_prompt_alone_counts_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "g_p0_events.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"type": "analysis", "transcript": SYSTEM + "[ANALYZER STATUS]\nrequest_error: refused\n"}) + "\n")
            out = analyze_run.summarise(analyze_run.analyse(d))
        self.assertEqual(out["games_using_tools"], 0)
        self.assertEqual(sum(out["calls"].values()), 0)

    def test_the_models_own_plan_is_not_the_tool(self):
        # 2026-10-28, tools off: the model wrote `def plan(fr):` (a BFS) and called it; a call before its
        # definition is still the tool's, the definition line and every later call are its own
        steps = [
            "[TOOL CALL: python]\nr = plan(predict, goal)\n[TOOL RESULT: python]\n{'status': 'found'}\n",
            "[TOOL CALL: python]\nfrom collections import deque\ndef plan(fr):\n    return bfs(fr)\n"
            "seq = plan(current_frame)\n[TOOL RESULT: python]\nok\n",
            "[TOOL CALL: python]\nprint(plan(current_frame), run_plan(seq, predict))\n[TOOL RESULT: python]\nok\n",
        ]
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "g_p0_events.jsonl"), "w", encoding="utf-8") as f:
                for step in steps:
                    f.write(json.dumps({"type": "analysis", "transcript": SYSTEM + step}) + "\n")
            out = analyze_run.summarise(analyze_run.analyse(d))
        self.assertEqual((out["calls"]["plan"], out["calls"]["run_plan"]), (1, 1))
        self.assertEqual(out["own_calls"]["plan"], 2)
        self.assertEqual(out["games_defining_own"], 1)
        self.assertEqual(out["per_game"][0]["self_defined"], ["plan"])
        self.assertEqual(out["per_game"][0]["distinct_call_lines"], 2)

    def test_generated_chars_by_level(self):
        meta = "[MODEL RESPONSE META]\nfinish_reason: tool_calls\ncontent_chars: 10\nreasoning_chars: 300\n"
        call = "[TOOL CALL: python]\nprint(1)\n[TOOL RESULT: python]\n1\n"  # 8 chars of code
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "g_p0_events.jsonl"), "w", encoding="utf-8") as f:
                for level, step in (("1", meta + call), ("1", meta), ("2", meta + call + meta)):
                    f.write(json.dumps({"type": "analysis", "level": level, "transcript": SYSTEM + step}) + "\n")
            out = analyze_run.summarise(analyze_run.analyse(d))
        self.assertEqual(out["per_game"][0]["gen_chars_by_level"], {"1": 628, "2": 628})


if __name__ == "__main__":
    unittest.main(verbosity=1)
