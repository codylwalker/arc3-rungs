"""tools/file_access.py on synthetic runs: a clean game, a planted read of the dataset, and the decoys.

Run: python -I tests/test_file_access.py
"""
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import file_access  # noqa: E402


def write_events(d, name, transcripts):
    with open(os.path.join(d, name + "_events.jsonl"), "w", encoding="utf-8") as f:
        for t in transcripts:
            f.write(json.dumps({"type": "analysis", "level": 1, "transcript": t}) + "\n")
        f.write(json.dumps({"type": "action", "game_over": False}) + "\n")


CLEAN = """[SYSTEM PROMPT]
You can open files? No: use the python tool over the frame. Path('x') is documentation, /kaggle too.
[TOOL CALL: python]
from collections import deque
def plan(fr):
    q = deque([fr])
    return q
action(1)
[TOOL RESULT: python]
/kaggle/working printed by the harness, open( in a result
"""

PLANTED = """[TOOL CALL: python]
import json
# open('/kaggle/input') in a comment does not count
g = json.load(open('/kaggle/input/arc3-gen-goal/games.json'))
print(g['games'][0]['variant'])
"""

DECOYS = """[TOOL CALL: python]
reopen(3)
board.open_cells = 4
my_path(1, 2)
x = obj.open(2)
"""


class FileAccess(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="fileaccess_")

    def test_clean_game_has_no_hit(self):
        write_events(self.d, "gaaa-00000001_p0", [CLEAN])
        games = file_access.scan_run(self.d)
        (g,) = games.values()
        self.assertEqual(g["code_blocks"], 1)
        self.assertEqual(g["hits"], [])

    def test_planted_read_is_found(self):
        write_events(self.d, "gbbb-00000002_p0", [CLEAN, PLANTED])
        (g,) = file_access.scan_run(self.d).values()
        self.assertEqual(g["code_blocks"], 2)
        self.assertEqual(len(g["hits"]), 1, g["hits"])
        kind, line = g["hits"][0]
        self.assertEqual(kind, "open")
        self.assertIn("games.json", line)
        self.assertTrue(file_access.names_inputs(line))
        self.assertFalse(file_access.names_inputs("open('/tmp/x','w') if False else None"))

    def test_decoys_do_not_hit(self):
        write_events(self.d, "gccc-00000003_p0", [DECOYS])
        (g,) = file_access.scan_run(self.d).values()
        self.assertEqual(g["hits"], [])

    def test_each_kind(self):
        for line, kind in (("os.listdir('.')", "listing"), ("import glob", "listing"),
                           ("p = Path('/tmp')", "pathlib"), ("s = 'gpk7.py'", "paths"),
                           ("import subprocess", "process"), ("import inspect; inspect.getsource(f)", "introspection"),
                           ("m = sys.modules['x']", "introspection"), ("print(GEN_GAMES_DIR)", "paths")):
            hits = file_access.scan_code(line)
            self.assertEqual([k for k, _ in hits], [kind], line)

    def test_passes_are_separate_games(self):
        write_events(self.d, "gddd-00000004_p0", [CLEAN])
        write_events(self.d, "gddd-00000004_p1", [PLANTED])
        games = file_access.scan_run(self.d)
        self.assertEqual(len(games), 2)
        self.assertEqual(sorted(len(g["hits"]) for g in games.values()), [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=1)
