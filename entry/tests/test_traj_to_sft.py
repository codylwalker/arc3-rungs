"""tools/traj_to_sft.py on a synthetic run in the harness's events format (2026-10-27).

Two games: one completes level 1 in 3 actions and leaves level 2 unfinished; the other completes level 1
in 5 actions (over 1.25 x the best of 3, so dropped) and its second pass completes it in 3. Checks the
level accounting, the rejection rules, the tool-call parse from `raw_tool_calls` and from markup, the
history budget, the image part and the requests.jsonl override.

Run: python -I tests/test_traj_to_sft.py
"""
import gzip
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import traj_to_sft as t  # noqa: E402

SYSTEM = "You are a coding agent solving a grid-based puzzle game.\n" * 20
BOARD = [[9] * 64 for _ in range(64)]


def meta(finish="tool_calls", calls=None, reasoning_chars=10):
    lines = ["finish_reason: %s" % finish, "served_by: (unknown)", "tool_call_count: %d" % len(calls or []),
             "content_chars: 0", "reasoning_chars: %d" % reasoning_chars, "raw_tool_calls:",
             json.dumps(calls or [], indent=2)]
    return "\n".join(lines)


def call(code, i=0):
    return {"id": "call_%d" % i, "index": i, "type": "function",
            "function": {"name": "python", "arguments": json.dumps({"code": code})}}


def transcript(step, action, user, responses):
    """responses: list of (meta body, thinking, tool call markup or None, tool result)."""
    parts = ["--- analysis_step=%d | action=%d | 10:00:00 | tool-agent ---" % (step, action),
             "[SYSTEM PROMPT]", SYSTEM, "[USER PROMPT]", user, "[ANALYZER STATUS]", "model: x"]
    for m, think, markup, result in responses:
        parts += ["[MODEL RESPONSE META]", m, "[THINKING]", think]
        if markup is not None:
            parts += ["[TOOL CALL: python]", markup]
        if result is not None:
            parts += ["[TOOL RESULT: python]", result]
    return "\n".join(parts) + "\n"


def analysis(step, action, level, score, text):
    return {"type": "analysis", "title": "Analysis Step %d" % step, "action_num": action, "analysis_step": step,
            "board": BOARD, "board_ascii": "", "score": score, "state": "NOT_FINISHED", "level": level,
            "run_status": "playing", "transcript": text}


def action(num, level, score, completed=False):
    return {"type": "action", "title": "Action %d" % num, "action_num": num, "analysis_step": None,
            "board": BOARD, "board_ascii": "", "score": score, "state": "NOT_FINISHED", "level": level,
            "run_status": "playing", "action_name": "ACTION1", "action_display": "UP", "reward": 0.0,
            "board_changed": True, "done": False, "level_completed": completed, "game_over": False,
            "run_complete": False, "batch_index": 0, "batch_size": 1}


def game_a():
    """Level 1 in 3 actions (2 turns), level 2 unfinished after 2 more actions (1 turn)."""
    ev = [{"type": "initial", "action_num": 0, "score": 0, "level": 1, "board": BOARD, "state": "NOT_FINISHED"}]
    ev.append(analysis(1, 1, 1, 0, transcript(1, 1, "Current state: step 0, level 1.", [
        (meta(calls=[call("print(1)")]), "think one", "<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>", "1"),
        (meta(calls=[call("action(['UP'])", 1)]), "think two", "<tool_call>\n<function=python>\n<parameter=code>\naction(['UP'])\n</parameter>\n</function>\n</tool_call>", "ok"),
    ])))
    ev.append(action(1, 1, 0))
    ev.append(analysis(2, 2, 1, 0, transcript(2, 2, "Executed actions: UP.", [
        (meta(calls=[call("action(['UP','UP'])")]), "think three", "<tool_call>\n<function=python>\n<parameter=code>\naction(['UP','UP'])\n</parameter>\n</function>\n</tool_call>", "ok"),
    ])))
    ev.append(action(2, 1, 0))
    ev.append(action(3, 2, 1, completed=True))
    ev.append(analysis(3, 4, 2, 1, transcript(3, 4, "You advanced to level 2.", [
        (meta(finish="length", calls=[]), "cut off", None, None),
        (meta(calls=[call("action(['DOWN'])")]), "think four", "<tool_call>\n<function=python>\n<parameter=code>\naction(['DOWN'])\n</parameter>\n</function>\n</tool_call>", "ok"),
    ])))
    ev.append(action(4, 2, 1))
    ev.append(action(5, 2, 1))
    return ev


def game_b(actions_to_finish, markup_only=False):
    """Level 1 completed in `actions_to_finish` actions by one turn."""
    ev = []
    m = meta(calls=None if markup_only else [call("action(['UP']*%d)" % actions_to_finish)])
    if markup_only:
        m = m.replace("raw_tool_calls:\n[]", "")
    ev.append(analysis(1, 1, 1, 0, transcript(1, 1, "Current state: step 0, level 1.", [
        (m, "think", "<tool_call>\n<function=python>\n<parameter=code>\naction(['UP']*%d)\n</parameter>\n</function>\n</tool_call>" % actions_to_finish, "ok"),
    ])))
    for i in range(1, actions_to_finish + 1):
        ev.append(action(i, 1 if i < actions_to_finish else 2, 0 if i < actions_to_finish else 1,
                         completed=(i == actions_to_finish)))
    return ev


def write_run(d, name, events):
    os.makedirs(os.path.join(d, "artifacts"), exist_ok=True)
    with open(os.path.join(d, "artifacts", name + "_events.jsonl"), "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def read(out, name, label="run"):
    with gzip.open(os.path.join(out, "%s__%s" % (label, name)), "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sft_")
        self.run = os.path.join(self.tmp, "run")
        write_run(self.run, "aaaa-00000001_p0", game_a())
        write_run(self.run, "bbbb-00000002_p0", game_b(5))
        write_run(self.run, "bbbb-00000002_p1", game_b(3, markup_only=True))
        self.out = os.path.join(self.tmp, "out")

    def test_level_stats(self):
        st = t.level_stats(game_a())
        self.assertEqual(st[1], {"completed": True, "actions": 3})
        self.assertEqual(st[2], {"completed": False, "actions": 2})

    def test_selection_and_parse(self):
        self.assertEqual(t.main([self.run, "--out", self.out, "--no-images"]), 0)
        a = read(self.out, "aaaa-00000001_p0.jsonl.gz")
        # level 1: 2 responses in turn 1 + 1 in turn 2; level 2 unfinished -> dropped
        self.assertEqual([(r["level"], r["analysis_step"], r["response_index"]) for r in a],
                         [(1, 1, 0), (1, 1, 1), (1, 2, 0)])
        self.assertEqual(a[0]["completion"]["tool_calls"][0]["function"]["name"], "python")
        self.assertEqual(json.loads(a[1]["completion"]["tool_calls"][0]["function"]["arguments"])["code"], "action(['UP'])")
        self.assertEqual(a[0]["completion"]["reasoning_content"], "think one")
        # the second response's context holds the first response and its tool result
        roles = [m["role"] for m in a[1]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool"])
        self.assertEqual(a[1]["messages"][3]["content"], "1")
        self.assertEqual(a[1]["messages"][3]["tool_call_id"], "call_0")
        # turn 2's context holds turn 1 whole
        self.assertEqual([m["role"] for m in a[2]["messages"]], ["system", "user", "assistant", "tool", "assistant", "tool", "user"])
        self.assertEqual(a[2]["level_actions"], 3)
        self.assertEqual(a[2]["best_actions"], 3)
        # game b: pass 0 took 5 actions, over 1.25 x the best (3) -> dropped; pass 1 kept, call from markup
        self.assertEqual(read(self.out, "bbbb-00000002_p0.jsonl.gz"), [])
        b1 = read(self.out, "bbbb-00000002_p1.jsonl.gz")
        self.assertEqual(len(b1), 1)
        self.assertEqual(json.loads(b1[0]["completion"]["tool_calls"][0]["function"]["arguments"])["code"], "action(['UP']*3)")
        with open(os.path.join(self.out, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        self.assertEqual(index["records"], 4)
        self.assertEqual(index["drops"], {"level_unfinished": 2, "actions_over_slack": 1})
        self.assertEqual(index["games"]["bbbb-00000002"]["best_actions"], {"1": 3})

    def test_all_flag_marks_drops(self):
        t.main([self.run, "--out", self.out, "--no-images", "--all"])
        a = read(self.out, "aaaa-00000001_p0.jsonl.gz")
        self.assertEqual([r["drop"] for r in a], [None, None, None, "level_unfinished", "level_unfinished"])
        self.assertEqual(a[3]["finish_reason"], "length")

    def test_reselect_across_sessions(self):
        # session 1 holds game b's 5-action pass only (kept on its own), session 2 the 3-action pass;
        # reselecting over both drops the 5-action records
        s1, s2 = os.path.join(self.tmp, "s1"), os.path.join(self.tmp, "s2")
        write_run(s1, "bbbb-00000002_p0", game_b(5))
        write_run(s2, "bbbb-00000002_p0", game_b(3))
        t.main([s1, "--out", s1 + "-sft", "--no-images", "--all"])
        t.main([s2, "--out", s2 + "-sft", "--no-images", "--all"])
        self.assertEqual([r["keep"] for r in read(s1 + "-sft", "bbbb-00000002_p0.jsonl.gz", "s1")], [True])
        merged = os.path.join(self.tmp, "merged")
        self.assertEqual(t.main([s1 + "-sft", s2 + "-sft", "--out", merged, "--reselect"]), 0)
        with open(os.path.join(merged, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        self.assertEqual(index["records"], 1)
        self.assertEqual(index["drops"], {"actions_over_slack": 1})
        self.assertEqual(index["best_actions"], {"bbbb-00000002/1": 3})
        kept = read(merged, "s2__bbbb-00000002_p0.jsonl.gz", "s2-sft")
        self.assertEqual(kept[0]["level_actions"], 3)
        self.assertNotIn("keep", kept[0])

    def test_history_budget(self):
        t.main([self.run, "--out", self.out, "--no-images", "--context-chars", str(len(SYSTEM) + 200)])
        a = read(self.out, "aaaa-00000001_p0.jsonl.gz")
        # the budget holds the system prompt and the current turn only: turn 1 is dropped from turn 2's context
        self.assertEqual([m["role"] for m in a[2]["messages"]], ["system", "user"])

    def test_images(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            print("skip: no PIL")
            return
        t.main([self.run, "--out", self.out, "--upscale", "2"])
        a = read(self.out, "aaaa-00000001_p0.jsonl.gz")
        user = a[0]["messages"][1]
        self.assertEqual([p["type"] for p in user["content"]], ["text", "image_url"])
        self.assertTrue(user["content"][0]["text"].endswith("Current grid image:"))
        self.assertTrue(user["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(a[0]["images"], 1)
        self.assertGreater(a[0]["approx_tokens"], t.IMAGE_TOKENS)

    def test_requests_override(self):
        req = os.path.join(self.run, "prompts")
        os.makedirs(req, exist_ok=True)
        exact = [{"role": "system", "content": "exact system"}, {"role": "user", "content": "exact user"}]
        with open(os.path.join(req, "aaaa-00000001_p0_requests.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"messages": exact, "tools": [], "event": "request", "analysis_step": 2, "request_index_within_turn": 0}) + "\n")
            f.write(json.dumps({"messages": exact, "tools": [], "event": "response", "finish_reason": "tool_calls", "analysis_step": 2, "request_index_within_turn": 0}) + "\n")
        t.main([self.run, "--out", self.out, "--no-images"])
        a = read(self.out, "aaaa-00000001_p0.jsonl.gz")
        self.assertEqual(a[2]["context_source"], "requests")
        self.assertEqual(a[2]["messages"], exact)
        self.assertEqual(a[0]["context_source"], "events")


if __name__ == "__main__":
    unittest.main()
