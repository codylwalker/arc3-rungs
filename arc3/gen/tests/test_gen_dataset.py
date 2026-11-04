"""entry/kaggle/build_gen_dataset.py on a synthetic bundle: only the .py and metadata.json per game,
no cert.json, games.json with mechanics, baselines and optima, the marker and a private metadata file.

Run (from tools/arc3): .venv-arc3/Scripts/python.exe -I gen/tests/test_gen_dataset.py
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ENTRY = os.path.normpath(os.path.join(HERE, "..", "..", "..", "entry"))
SCRIPT = os.path.join(ENTRY, "kaggle", "build_gen_dataset.py")


def load_script():
    spec = importlib.util.spec_from_file_location("build_gen_dataset", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_set(d):
    games = []
    for gid, mech, opt in (("gaaa-00000001", ["hazard"], [3, 5]), ("gbbb-00000002", ["ice", "keydoor"], [4, 6, 8])):
        id4, h = gid.split("-")
        p = os.path.join(d, id4, h)
        os.makedirs(p)
        with open(os.path.join(p, id4 + ".py"), "w") as f:
            f.write("# game %s\n" % gid)
        base = [-(-o * 3 // 2) for o in opt]
        with open(os.path.join(p, "metadata.json"), "w") as f:
            json.dump({"game_id": gid, "baseline_actions": base, "mechanics": mech}, f)
        with open(os.path.join(p, "cert.json"), "w") as f:
            json.dump({"game_id": gid, "levels": [{"level": i + 1, "optimal": o, "path": [[1]] * o} for i, o in enumerate(opt)]}, f)
        games.append({"game_id": gid, "mechanics": mech, "baseline_actions": base, "source_set": "set-x",
                      "levels": [{"level": i + 1, "optimal": o, "path": [[1]] * o} for i, o in enumerate(opt)]})
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump({"games": games, "sources": ["set-x"]}, f)


class Test(unittest.TestCase):
    def test_build(self):
        if not os.path.isfile(SCRIPT):
            print("skip: no arc3-entry checkout at", ENTRY)
            return
        mod = load_script()
        tmp = tempfile.mkdtemp(prefix="gends_")
        src, out = os.path.join(tmp, "set"), os.path.join(tmp, "out")
        make_set(src)
        games = mod.build(src, out, "someone")
        self.assertEqual([g["game_id"] for g in games], ["gaaa-00000001", "gbbb-00000002"])
        files = sorted(os.path.relpath(os.path.join(r, n), out).replace(os.sep, "/")
                       for r, _, ns in os.walk(out) for n in ns)
        self.assertEqual(files, ["GEN_GAMES_ROOT", "dataset-metadata.json", "gaaa/00000001/gaaa.py", "gaaa/00000001/metadata.json",
                                 "games.json", "gbbb/00000002/gbbb.py", "gbbb/00000002/metadata.json"])
        with open(os.path.join(out, "games.json")) as f:
            gj = json.load(f)
        self.assertEqual(gj["games"][1], {"game_id": "gbbb-00000002", "mechanics": ["ice", "keydoor"], "baseline_actions": [6, 9, 12],
                                          "optimal": [4, 6, 8], "source_set": "set-x"})
        with open(os.path.join(out, "dataset-metadata.json")) as f:
            meta = json.load(f)
        self.assertEqual(meta["id"], "someone/arc3-gen-games")
        # a rebuild replaces the folder (no stale games)
        os.remove(os.path.join(src, "gbbb", "00000002", "cert.json"))
        games = mod.build(src, out, "someone")
        self.assertEqual(len(games), 2)


if __name__ == "__main__":
    unittest.main()
