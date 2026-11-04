"""Click-family tests: loader contract through arc_agi, ACTION6 exposure, oracle paths as clicks."""
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.dirname(HERE)
ARC3 = os.path.dirname(GEN)
sys.path.insert(0, GEN)
sys.path.insert(0, ARC3)

import corpus  # noqa: E402
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

_CACHE = {}


def built(kind, seed):
    if (kind, seed) not in _CACHE:
        out = tempfile.mkdtemp(prefix="arc3click_")
        cert, fail = corpus.make_one(seed, out, max_nodes=20000, max_secs=60, family="click", kind=kind)
        assert cert, fail
        _CACHE[(kind, seed)] = (out, cert)
    return _CACHE[(kind, seed)]


def _play(kind, seed):
    out, cert = built(kind, seed)
    gid = cert["game_id"].split("-")[0]
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=out)
    env = arc.make(gid, seed=0)
    obs = env.reset()
    assert env.action_space == [GameAction.ACTION6], env.action_space
    for i, lv in enumerate(cert["levels"]):
        for a in lv["path"]:
            assert a[0] == 6
            obs = env.step(GameAction.ACTION6, data={"x": a[1], "y": a[2]})
            assert obs.state != GameState.GAME_OVER
        assert obs.levels_completed == i + 1 or obs.state == GameState.WIN
    assert obs.state == GameState.WIN


def test_toggle_paths():
    _play("toggle", 6)


def test_move_paths():
    _play("move", 5)


def test_random_clicks_rarely_clear_toggle():
    out, cert = built("toggle", 6)
    gid = cert["game_id"].split("-")[0]
    import t1_probe
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=out)
    rng = random.Random(0)
    pol = [p for p in t1_probe.policies(rng) if p.name == "click_center"][0]
    r = t1_probe.run(arc, gid, pol, 0, 100)
    assert r["levels"] <= 1, r


def test_reset_restores_level_start():
    """RESET two clicks into level 2 gives back level 2's first frame, budget bar included."""
    import numpy as np
    out, cert = built("move", 5)
    gid = cert["game_id"].split("-")[0]
    env = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=out).make(gid, seed=0)
    env.reset()
    for a in cert["levels"][0]["path"]:
        obs = env.step(GameAction.ACTION6, data={"x": a[1], "y": a[2]})
    start2 = np.array(obs.frame[-1])
    assert obs.levels_completed == 1
    for a in cert["levels"][1]["path"][:2]:
        obs = env.step(GameAction.ACTION6, data={"x": a[1], "y": a[2]})
    obs = env.reset()
    assert obs.levels_completed == 1 and (np.array(obs.frame[-1]) == start2).all()


def _xor(masks, x):
    out = 0
    for j, m in enumerate(masks):
        if x >> j & 1:
            out ^= m
    return out


def test_gf2_min_clicks_matches_brute_force():
    """The algebraic toggle optimum equals the fewest clicks over every click subset (boards up to 4x3)."""
    import certify
    rng = random.Random(1)
    for rule in ("self", "plus", "rowcol"):
        for w, h in ((2, 2), (3, 3), (4, 3)):
            masks = certify._toggle_masks(w, h, rule)
            n = w * h
            for _ in range(6):
                b = _xor(masks, rng.randrange(1 << n))  # a scramble's difference, so a solution exists
                x, _ = certify.gf2_min_clicks(masks, b, n)
                best = min(bin(t).count("1") for t in range(1 << n) if _xor(masks, t) == b)
                assert x is not None and _xor(masks, x) == b and bin(x).count("1") == best, (rule, w, h, b)


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
