"""Generated games through arc_agi's competition-mode REST serving, the layer the harness drives: an
offline Arcade over a games dir, arc_agi.server.create_app(competition_mode=True) on a localhost port,
and a COMPETITION-mode arc_agi client (what taaf's CompetitionArcadeServer and game_api do; only the
arc_agi toolkit runs here, not the harness checkout). Checks that the server lists the games, that
oracle paths win over HTTP (keyboard and click), that our baselines reach the server's scorer (an oracle
playthrough scores exactly 100: every level caps at 115 since baseline = 1.5 x optimal), and that a
RESET under ONLY_RESET_LEVELS=true (set process-wide by the harness after its first make) gives back
the level's first frame. Run from tools/arc3 with the .venv-arc3 interpreter.
"""
import contextlib
import logging
import os
import socketserver
import sys
import tempfile
import threading
import time
from urllib.request import urlopen
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.dirname(HERE)
sys.path.insert(0, GEN)
sys.path.insert(0, os.path.dirname(GEN))

import corpus  # noqa: E402
import arc_agi  # noqa: E402
from arc_agi import server as arc_server  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

LOG = logging.getLogger("gen.test_competition")
LOG.setLevel(logging.WARNING)
logging.getLogger("arc_agi.scorecard").setLevel(logging.WARNING)
_SET = {}


class _Server(socketserver.ThreadingMixIn, WSGIServer):
    daemon_threads = True


class _Quiet(WSGIRequestHandler):
    def log_message(self, *args):
        pass


def built():
    if not _SET:
        out = tempfile.mkdtemp(prefix="arc3comp_")
        move, fail = corpus.make_one(3, out, max_nodes=20000, max_secs=60)
        assert move, fail
        click, fail = corpus.make_one(6, out, max_nodes=20000, max_secs=60, family="click", kind="toggle")
        assert click, fail
        _SET.update(out=out, move=move, click=click)
    return _SET


@contextlib.contextmanager
def serving(games_dir):
    """A competition-mode server over games_dir on 127.0.0.1; yields a COMPETITION-mode client."""
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=games_dir, logger=LOG)
    app, _api = arc_server.create_app(arcade, competition_mode=True, include_frame_data=True)
    srv = make_server("127.0.0.1", 0, app, server_class=_Server, handler_class=_Quiet)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = "http://127.0.0.1:%d" % srv.server_port
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with urlopen(base + "/api/healthcheck", timeout=1) as r:
                    if r.status == 200:
                        break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        client = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.COMPETITION, arc_base_url=base,
                                arc_api_key="gen-test", environments_dir=tempfile.mkdtemp(prefix="arc3none_"),
                                logger=LOG)
        yield client
    finally:
        srv.shutdown()
        th.join(timeout=5)


def _step(env, rec):
    if rec[0] == 6:
        return env.step(GameAction.ACTION6, data={"x": rec[1], "y": rec[2]})
    return env.step(GameAction.from_id(rec[0]), data={})


def test_server_lists_plays_and_scores_generated_games():
    s = built()
    with serving(s["out"]) as arc:
        listed = {e.game_id for e in arc.available_environments}
        assert {s["move"]["game_id"], s["click"]["game_id"]} <= listed, listed
        card = arc.create_scorecard()
        for cert in (s["move"], s["click"]):
            env = arc.make(cert["game_id"], scorecard_id=card)
            assert env is not None, cert["game_id"]
            for i, lv in enumerate(cert["levels"]):
                for rec in lv["path"]:
                    obs = _step(env, rec)
                    assert obs.state != GameState.GAME_OVER
                assert obs.levels_completed == i + 1 or obs.state == GameState.WIN
            assert obs.state == GameState.WIN, cert["game_id"]
        sc = arc.close_scorecard(card)
        for cert in (s["move"], s["click"]):
            runs = sc.find_environment(cert["game_id"]).runs
            assert runs and abs(runs[0].score - 100.0) < 1e-6, (cert["game_id"], [r.score for r in runs])


def test_level_reset_over_http():
    s = built()
    cert = s["move"]
    os.environ["ONLY_RESET_LEVELS"] = "true"
    try:
        with serving(s["out"]) as arc:
            card = arc.create_scorecard()
            env = arc.make(cert["game_id"], scorecard_id=card)
            for rec in cert["levels"][0]["path"]:
                obs = _step(env, rec)
            start2 = np.array(obs.frame[-1])
            assert obs.levels_completed == 1
            for rec in cert["levels"][1]["path"][:2]:
                obs = _step(env, rec)
            assert not (np.array(obs.frame[-1]) == start2).all()
            obs = env.reset()
            assert obs.levels_completed == 1 and (np.array(obs.frame[-1]) == start2).all()
            arc.close_scorecard(card)
    finally:
        os.environ.pop("ONLY_RESET_LEVELS", None)


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception:  # noqa: BLE001
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if fails else 0)
