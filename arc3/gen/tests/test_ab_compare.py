"""ab_compare on synthetic harness outputs: two arms, generated and public games, a game split across two
runs of one arm, a re-emitted game whose hash differs between arms, and the per-mechanic table."""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.dirname(HERE)
sys.path.insert(0, GEN)

import ab_compare  # noqa: E402


def _run(root, name, passes, games):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "artifacts"))
    lines = ["benchmark: test", "passes:    %d" % passes, "", "per-game (mean across passes):"]
    for gid, score, lv, tot, acts, toks in games:
        lines.append("  %s: score=%.2f, levels=%.1f/%d, actions=%d, tokens=%d" % (gid, score, lv, tot, acts, toks))
    with open(os.path.join(d, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return d


def _set(root, games):
    for gid, mech in games:
        d = os.path.join(root, gid.split("-")[0], gid.split("-")[1])
        os.makedirs(d)
        with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"game_id": gid, "mechanics": mech}, f)
    return root


def test_compare_on_synthetic_runs():
    root = tempfile.mkdtemp(prefix="arc3ab_")
    s = _set(os.path.join(root, "set"), [("gaaa-11111111", ["hazard", "ice"]), ("gbbb-22222222", ["keydoor"]),
                                         ("gccc-33333333", ["ice"])])
    a1 = _run(root, "a1", 2, [("gaaa-11111111", 50, 3, 6, 40, 1000), ("gbbb-22222222", 20, 1, 6, 30, 900),
                              ("ls20-aaaaaaaa", 10, 1, 7, 50, 2000)])
    a2 = _run(root, "a2", 2, [("gccc-33333333", 0, 0, 6, 10, 500), ("gaaa-11111111", 70, 5, 6, 40, 3000)])
    b1 = _run(root, "b1", 4, [("gaaa-11111111", 80, 5, 6, 40, 1500), ("gbbb-22222222", 40, 2, 6, 30, 800),
                              ("gccc-99999999", 30, 2, 6, 10, 700), ("ls20-aaaaaaaa", 20, 2, 7, 50, 2100),
                              ("gzzz-00000000", 1, 1, 6, 1, 1)])
    a, b = ab_compare.load_arm([a1, a2]), ab_compare.load_arm([b1])
    assert a["gaaa-11111111"]["levels"] == 4.0 and a["gaaa-11111111"]["passes"] == 4  # merged, weighted by passes
    pairs = ab_compare.pair(a, b)
    assert ("gccc-33333333", "gccc-99999999") in pairs and len(pairs) == 4  # base-id fallback; gzzz unpaired
    r = ab_compare.compare(a, b, pairs)
    assert r["arm_a"]["levels_solved"] == 6.0 and r["arm_b"]["levels_solved"] == 11.0
    assert r["delta_levels"]["b_better"] == 4 and r["delta_levels"]["a_better"] == 0
    out = os.path.join(root, "r.json")
    p = subprocess.run([sys.executable, "-I", os.path.join(GEN, "ab_compare.py"), "--a", a1, a2, "--b", b1,
                        "--set", s, "--json", out], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    j = json.load(open(out, encoding="utf-8"))
    assert j["generated"]["games"] == 3 and j["public"]["games"] == 1
    assert j["by_mechanic"]["ice"]["games"] == 2 and j["direction_agreement"] is True
    assert j["unpaired"] == ["gzzz-00000000"]


if __name__ == "__main__":
    test_compare_on_synthetic_runs()
    print("ok   test_compare_on_synthetic_runs")
