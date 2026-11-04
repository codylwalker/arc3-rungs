"""twin_keepalive with a fake game and no real waiting.

Run: python -I tests/test_keepalive.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import twin_keepalive as ka  # noqa: E402


class FakeGame:
    class _Run:
        game_id = "ls20-test"

    class _G:
        game_run = None

    def __init__(self, fail=False):
        self.action_count = 1  # the warmup RESET already happened
        self.resets = 0
        self.stop = False
        self.fail = fail
        self.game = FakeGame._G()
        self.game.game_run = FakeGame._Run()

    def should_stop(self):
        return self.stop

    def _execute_auto_reset(self):
        if self.fail:
            raise RuntimeError("gateway closed")
        self.resets += 1
        self.action_count += 1


class Log:
    def __init__(self):
        self.lines = []

    def warning(self, fmt, *args):
        self.lines.append(fmt % args)


def run(game, ups, period=1.0, max_resets=12, on_sleep=None):
    seq = list(ups)
    log = Log()

    def up():
        return seq.pop(0) if seq else False

    def sleep(_):
        if on_sleep:
            on_sleep(game)

    t = ka.start_keepalive(game, log=log, period=period, max_resets=max_resets, endpoint_up=up, sleep=sleep)
    if t is not None:
        t.join(5)
    return t, log


class KeepAlive(unittest.TestCase):
    def test_resets_until_the_cap_while_the_server_stays_down(self):
        g = FakeGame()
        t, log = run(g, [], max_resets=3)
        self.assertEqual(g.resets, 3)
        self.assertIn("keep-alive stops after 3 RESETs", log.lines[-1])

    def test_stops_when_the_endpoint_answers(self):
        g = FakeGame()
        t, log = run(g, [False, False, True])
        self.assertEqual(g.resets, 2)
        self.assertIn("the analyzer endpoint answers (2 keep-alive RESETs)", log.lines[-1])

    def test_stops_when_the_game_acts(self):
        g = FakeGame()
        calls = {"n": 0}

        def on_sleep(game):
            calls["n"] += 1
            if calls["n"] == 2:
                game.action_count += 1  # the game's own first action, between two keep-alives

        t, log = run(g, [], on_sleep=on_sleep)
        self.assertEqual(g.resets, 1)
        self.assertIn("the game has acted", log.lines[-1])

    def test_stops_when_the_game_stops(self):
        g = FakeGame()
        g.stop = True
        run(g, [])
        self.assertEqual(g.resets, 0)

    def test_a_failing_reset_never_raises(self):
        g = FakeGame(fail=True)
        t, log = run(g, [])
        self.assertIn("keep-alive RESET failed", log.lines[-1])

    def test_disabled(self):
        self.assertIsNone(ka.start_keepalive(FakeGame(), period=0))
        os.environ.pop("ARC3_KEEPALIVE_SECONDS", None)
        self.assertIsNone(ka.start_keepalive(FakeGame()))

    def test_health_url(self):
        os.environ["LOCAL_ANALYZER_BASE_URL"] = "http://127.0.0.1:8001/v1"
        self.assertEqual(ka._health_url(), "http://127.0.0.1:8001/health")


if __name__ == "__main__":
    unittest.main(verbosity=1)
