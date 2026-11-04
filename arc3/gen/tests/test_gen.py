"""Generated-game tests. Run from tools/arc3 with the .venv-arc3 interpreter:
    python -m pytest gen/tests -q      (or: python gen/tests/test_gen.py)
Covers the loader contract (arc_agi offline Arcade), the oracle paths, RESET semantics (the engine
re-runs step() after a RESET; it must not spend the budget), baselines, replay digests, a T1 smoke
(blind policies clear no level) and the composer: push lanes, a glyph demanded only with a cross,
no glyph drawn in the colour of anything the game draws.
"""
import hashlib
import json
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
import emit  # noqa: E402
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

_CACHE = {}


def built(seed=3):
    """One generated game per seed, built once per test session: (dir, cert, full_id)."""
    if seed not in _CACHE:
        out = tempfile.mkdtemp(prefix="arc3gen_")
        cert, fail = corpus.make_one(seed, out, max_nodes=20000, max_secs=60)
        assert cert, fail
        gid = cert["game_id"].split("-")[0]
        d = os.path.join(out, gid, cert["game_id"].split("-")[1])
        _CACHE[seed] = (out, d, cert)
    return _CACHE[seed]


def frame_key(obs):
    import numpy as np
    return hashlib.sha256(np.asarray(obs.frame[-1]).astype("int8").tobytes()).hexdigest()


def test_emit_and_loader_contract():
    out, d, cert = built()
    gid = cert["game_id"].split("-")[0]
    src = open(os.path.join(d, gid + ".py"), encoding="utf-8").read()
    assert "import" in src and "from arcengine import" in src
    for line in src.splitlines():
        if line.startswith("import ") or line.startswith("from "):
            assert line.split()[1].split(".")[0] in ("numpy", "arcengine"), line
    assert emit.class_name(gid) in src
    meta = json.load(open(os.path.join(d, "metadata.json"), encoding="utf-8"))
    assert meta["game_id"] == cert["game_id"]
    assert len(meta["baseline_actions"]) == 6
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=out)
    env = arc.make(gid, seed=0)
    assert env is not None
    obs = env.reset()
    assert len(obs.frame[-1]) == 64 and len(obs.frame[-1][0]) == 64
    assert obs.win_levels == 6
    assert GameAction.ACTION6 not in env.action_space


def test_oracle_paths_complete_every_level():
    out, d, cert = built()
    gid = cert["game_id"].split("-")[0]
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=out)
    env = arc.make(gid, seed=0)
    obs = env.reset()
    for i, lv in enumerate(cert["levels"]):
        for a in lv["path"]:  # path records are [id] (movement) or [6, x, y] (click)
            obs = env.step(GameAction.from_id(a[0]), data={})
            assert obs.state != GameState.GAME_OVER
        assert obs.levels_completed == i + 1 or obs.state == GameState.WIN
    assert obs.state == GameState.WIN


def test_reset_restores_level_start():
    out, d, cert = built()
    gid = cert["game_id"].split("-")[0]
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=out)
    env = arc.make(gid, seed=0)
    obs0 = env.reset()
    k0 = frame_key(obs0)
    # play level 1 to completion, then two actions into level 2, then RESET: level 2's start again
    for a in cert["levels"][0]["path"]:
        obs = env.step(GameAction.from_id(a[0]), data={})
    start2 = frame_key(obs)
    assert obs.levels_completed == 1
    for a in cert["levels"][1]["path"][:2]:
        obs = env.step(GameAction.from_id(a[0]), data={})
    obs = env.reset()
    assert obs.levels_completed == 1
    assert frame_key(obs) == start2
    assert frame_key(obs) != k0


def test_baselines_and_budget():
    out, d, cert = built()
    meta = json.load(open(os.path.join(d, "metadata.json"), encoding="utf-8"))
    import math
    for lv, b in zip(cert["levels"], meta["baseline_actions"]):
        assert b == int(math.ceil(1.5 * lv["optimal"]))
        assert lv["optimal"] >= 3
    assert cert["replay_ok"]


def test_replay_digest_is_stable():
    out, d, cert = built()
    gid = cert["game_id"].split("-")[0]
    import t3_seal
    actions = cert["levels"][0]["path"] + cert["levels"][1]["path"]  # already [id] records
    d1, _ = t3_seal.play(Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=out), gid, 0, actions)
    d2, _ = t3_seal.play(Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=out), gid, 0, actions)
    assert d1 == d2 and len(d1) == len(actions) + 1


def test_blind_policies_clear_nothing():
    out, d, cert = built()
    gid = cert["game_id"].split("-")[0]
    import t1_probe
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=out)
    rng = random.Random(0)
    for pol in t1_probe.policies(rng):
        if not pol.name.startswith("blind") and pol.name != "click_center":
            continue
        r = t1_probe.run(arc, gid, pol, 0, 200)
        assert r["levels"] == 0, (pol.name, r)


def _at(rows, x, y):
    return rows[y][x] if 0 <= y < len(rows) and 0 <= x < len(rows[0]) else "#"


def test_push_pairs_have_a_lane():
    """Every block has its plate 1-3 cells away along free cells, and a free cell behind it to push from."""
    import compose
    seen = 0
    for seed in range(40):
        plan = compose.compose_game(seed, k=2, must=["pushable"])
        for spec in plan["levels"]:
            rows = spec["rows"]
            blocks = [(x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch == "B"]
            plates = [(x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch == "P"]
            assert len(blocks) == len(plates) <= 2, (seed, rows)
            for bx, by in blocks:
                ok = False
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    if _at(rows, bx - dx, by - dy) != ".":
                        continue
                    for n in (1, 2, 3):
                        lane = [_at(rows, bx + dx * i, by + dy * i) for i in range(1, n + 1)]
                        ok = ok or (lane[-1] == "P" and all(c == "." for c in lane[:-1]))
                assert ok, ("a block without a push lane", seed, rows)
                seen += 1
    assert seen > 40, seen


def test_move_palette_has_no_clashes():
    """No glyph in use shares a colour with anything the game draws, including the fullest 4-mechanic mix."""
    import compose
    import palette_check
    for seed in range(200):
        plan = compose.compose_game(seed)
        assert not palette_check.clashes(plan["config"], plan["levels"]), (seed, plan["config"]["mechanics"])
    for seed in range(20):
        plan = compose.compose_game(seed, k=4, must=["keydoor", "pushable", "state_flip", "teleporter"])
        assert not palette_check.clashes(plan["config"], plan["levels"]), (seed, plan["config"]["palette"])


def test_glyph_needed_only_with_a_cross():
    """A level demands a glyph only if it has a cross to change it (the cross-less tutorial bug)."""
    import compose
    for seed in range(30):
        plan = compose.compose_game(seed, k=2, must=["state_flip"])
        for spec in plan["levels"]:
            assert ("need_glyph" in spec) == any("C" in r for r in spec["rows"]), (seed, spec)


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
