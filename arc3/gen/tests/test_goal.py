"""Goal-kind tests (the goal-rung set, 2026-10-30). Run from tools/arc3 with the .venv-arc3 interpreter:
    python gen/tests/test_goal.py
Covers the composer's goal kinds (the goal's mechanic leads, no G tile, switch excluded, one key more than doors);
the runtime's goal semantics through the oracle (a keys level needs every key, a plates level every plate); the HUD
signal (pips on the band's last row only in the signalled variant, filled as items are done, the variants' frames
differing nowhere else); the snapshot the oracle merges states by, with a planted incomplete snapshot that must
change the oracle's answer; and the pair: one certified plan emitted twice, same optima, both replaying, distinct
ids, labelled metadata, both loading in arc_agi's offline Arcade.
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.dirname(HERE)
ARC3 = os.path.dirname(GEN)
sys.path.insert(0, GEN)
sys.path.insert(0, ARC3)

import certify  # noqa: E402
import compose  # noqa: E402
import corpus  # noqa: E402
import runtime_move  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402

KEYS_ROWS = ["#######", "#A.K.K#", "#######"]   # both keys to the right: optimal 4
PLATE_ROWS = ["#######", "#A.B.P#", "#######"]  # push the block onto the plate: optimal 3
BACK_ROWS = ["#######", "#K.A.K#", "#######"]   # a key each way: optimal 6, and only by coming back with a key


def _cfg(goal, **extra):
    """A real palette and direction map from the composer, with the goal kind set."""
    plan = compose.compose_game(11, k=2, goal=goal)
    cfg = dict(plan["config"])
    cfg.update(extra)
    return cfg


def _game(goal, rows, **extra):
    return runtime_move.GenGame("gtst", _cfg(goal, **extra), [{"rows": rows, "budget": 0}])


def _act(game, d):
    """Move in direction d (0 up, 1 right, 2 down, 3 left) through the game's own action map."""
    a = next(int(k) for k, v in game._cfg["dir_map"].items() if v == d)
    return game.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)


def test_compose_goal_kinds():
    for goal, mech in sorted(compose.GOAL_MECH.items()):
        for seed in range(6):
            plan = compose.compose_game(seed, k=2, goal=goal)
            ms = plan["config"]["mechanics"]
            assert ms[0] == mech, (goal, seed, ms)
            assert "switch" not in ms, (goal, seed, ms)
            assert plan["config"]["goal"] == goal
            for spec in plan["levels"]:
                rows = spec["rows"]
                assert not any("G" in r for r in rows), (goal, seed, rows)
                text = "".join(rows)
                if goal == "keys":
                    assert text.count("K") >= 1 and text.count("K") > text.count("D"), (seed, rows)
                else:
                    assert text.count("P") >= 1 and text.count("B") >= 1, (seed, rows)


def test_keys_goal_needs_every_key():
    path, _, _ = certify.oracle_solve(_game("keys", KEYS_ROWS), 0, 5000, 10)
    assert path is not None and len(path) == 4, path
    g = _game("keys", KEYS_ROWS)
    g.full_reset()
    _act(g, 1)
    fd = _act(g, 1)  # the first key
    assert fd.levels_completed == 0 and fd.state != GameState.WIN, "one key of two must not complete the level"
    _act(g, 1)
    fd = _act(g, 1)  # the second
    assert fd.state == GameState.WIN, fd.state


def test_plates_goal_needs_every_plate():
    path, _, _ = certify.oracle_solve(_game("plates", PLATE_ROWS), 0, 5000, 10)
    assert path is not None and len(path) == 3, path


def test_signal_only_in_the_signalled_variant():
    plain = _game("keys", KEYS_ROWS, show_goal=False)
    shown = _game("keys", KEYS_ROWS, show_goal=True)
    plain.full_reset()
    shown.full_reset()
    assert plain._by_name("hud_goal") is None, "the consequence variant draws no goal"
    hg = shown._by_name("hud_goal")
    pal = shown._cfg["palette"]
    assert hg is not None and hg.y == runtime_move.HUD - 1, "the goal row is the band's last row"
    assert hg.pixels.tolist() == [[pal["goal"]] * 2 + [-1] + [pal["goal"]] * 2], hg.pixels.tolist()
    f_plain = f_shown = None
    for _ in range(2):  # walk to the first key in both
        f_plain, f_shown = _act(plain, 1), _act(shown, 1)
    assert shown._by_name("hud_goal").pixels.tolist() == [[pal["key"]] * 2 + [-1] + [pal["goal"]] * 2], \
        "a collected key fills its pip"
    a, b = f_plain.frame[-1], f_shown.frame[-1]
    diff = [r for r in range(len(a)) if list(a[r]) != list(b[r])]
    assert diff, "the signalled variant must show something"
    assert max(diff) - min(diff) < len(a) // 4, "the variants differ only in the HUD band: rows %s" % diff


class _Forgetful(runtime_move.GenGame):
    """A planted fault: a snapshot without the keys held or collected, as a runtime that forgot to export them."""

    def export_state(self):
        s = list(super().export_state())
        s[6], s[7] = 0, ()
        return tuple(s)


def test_snapshot_carries_the_goal():
    good, _, _ = certify.oracle_solve(_game("keys", BACK_ROWS), 0, 5000, 10)
    assert good is not None and len(good) == 6, good
    forgetful = _Forgetful("gtst", _cfg("keys"), [{"rows": BACK_ROWS, "budget": 0}])
    bad, _, _ = certify.oracle_solve(forgetful, 0, 5000, 10)
    assert bad is None or len(bad) != 6, \
        "the planted fault must change the oracle's answer, or this test cannot see a missing goal variable"


def test_pair_emits_two_matched_variants():
    out = tempfile.mkdtemp(prefix="arc3goal_")
    certs = None
    for seed in range(40, 60):
        certs, fail = corpus.make_one(seed, out, k=2, max_nodes=20000, max_secs=60, goal="keys", pair=True)
        if certs:
            break
    assert certs, "no seed in 40..59 made a keys pair"
    c, s = certs
    assert (c["variant"], s["variant"]) == ("consequence", "signalled")
    assert c["replay_ok"] and s["replay_ok"]
    assert [r["optimal"] for r in c["levels"]] == [r["optimal"] for r in s["levels"]]
    assert c["game_id"].split("-")[0] != s["game_id"].split("-")[0]
    assert c["pair"] == s["game_id"] and s["pair"] == c["game_id"] and c["pair_seed"] == s["pair_seed"]
    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=out)
    for cert, show in ((c, False), (s, True)):
        gid, h = cert["game_id"].split("-")
        d = os.path.join(out, gid, h)
        meta = json.load(open(os.path.join(d, "metadata.json"), encoding="utf-8"))
        assert meta["goal"] == "keys" and meta["variant"] == cert["variant"], meta
        src = open(os.path.join(d, gid + ".py"), encoding="utf-8").read()
        assert ("'show_goal': %s" % show) in src, "the emitted CONFIG carries show_goal=%s" % show
        assert arc.make(gid, seed=0) is not None, "loads in the offline Arcade: %s" % gid


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
