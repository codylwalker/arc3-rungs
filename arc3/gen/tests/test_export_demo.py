"""gen/export_demo.py: the exported frames are the game's own (pixel-identical with render.py --cert's PNG sheet),
the final sha checks, a pair shares its certificate and differs only in its goal band, the export is deterministic,
and only allowed keys are written.

Run from tools/arc3: .venv-arc3/Scripts/python.exe -I gen/tests/test_export_demo.py
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.dirname(HERE)
sys.path.insert(0, GEN)
import export_demo  # noqa: E402
import render  # noqa: E402

SET = os.path.join(GEN, "out", "goal-set-a")
GAME = "gpk7-89058af4"   # consequence, keys, seed 1000
PAIR = "g718-b2eff13d"   # its signalled twin


def game_dir(gid):
    a, b = gid.split("-")
    return os.path.join(SET, a, b)


def export_one(gid):
    with open(os.path.join(game_dir(gid), "metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(game_dir(gid), "cert.json"), encoding="utf-8") as f:
        cert = json.load(f)
    return export_demo.export_game(game_dir(gid), meta, cert)


def rle_to_frame(hexrle):
    out, at = np.zeros(64 * 64, dtype=np.int64), 0
    for i in range(0, len(hexrle), 5):
        n, c = int(hexrle[i:i + 4], 16), int(hexrle[i + 4], 16)
        out[at:at + n] = c
        at += n
    assert at == 64 * 64
    return out


def decode(level):
    frames = [rle_to_frame(level["f0"])]
    for d in level["d"]:
        g = frames[-1].copy()
        for i in range(0, len(d), 4):
            g[int(d[i:i + 3], 16)] = int(d[i + 3], 16)
        frames.append(g)
    return frames


class ExportDemo(unittest.TestCase):
    def test_frames_match_render_cert_sheet(self):
        """Level 2's decoded frames equal the panels render.py --cert --level 1 draws (scale 6, 6-px gaps)."""
        doc = export_one(GAME)
        level = doc["levels"][1]
        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, "sheet.png")
            subprocess.run([sys.executable, "-I", os.path.join(GEN, "render.py"), game_dir(GAME), "--cert",
                            "--level", "1", "--out", png], check=True, capture_output=True)
            sheet = np.array(Image.open(png).convert("RGB"))
        frames = decode(level)
        self.assertEqual(len(frames), level["optimal"] + 1)
        k = 6
        for i, fr in enumerate(frames):
            x0 = i * (64 * k + 6)
            panel = sheet[:64 * k, x0:x0 + 64 * k]
            self.assertTrue(np.array_equal(panel, render.to_rgb(fr.reshape(64, 64), k)), "panel %d" % i)

    def test_final_sha_and_pair(self):
        a, b = export_one(GAME), export_one(PAIR)
        for doc in (a, b):
            for lv in doc["levels"]:
                hexframe = "".join("%x" % v for v in decode(lv)[-1].tolist())
                self.assertEqual(hashlib.sha256(hexframe.encode("ascii")).hexdigest(), lv["final_sha256"])
        self.assertEqual((a["pair"], b["pair"]), (PAIR, GAME))
        self.assertEqual([lv["path"] for lv in a["levels"]], [lv["path"] for lv in b["levels"]])
        self.assertEqual([lv["optimal"] for lv in a["levels"]], [lv["optimal"] for lv in b["levels"]])
        bands = export_demo.pair_bands(b, a)
        self.assertEqual(len(bands), len(a["levels"]))
        for per_step, la, lb in zip(bands, a["levels"], b["levels"]):
            frames_a, frames_b = decode(la), decode(lb)
            self.assertEqual(len(per_step), len(frames_a))
            for (r0, r1), fa, fb in zip(per_step, frames_a, frames_b):
                self.assertLessEqual(r1 - r0 + 1, export_demo.MAX_BAND_ROWS)
                self.assertLessEqual(r0, export_demo.MAX_BAND_START)
                rows = {int(i) // 64 for i in np.nonzero(fa != fb)[0]}
                self.assertTrue(rows and rows <= set(range(r0, r1 + 1)), "outside the band: %s" % sorted(rows))

    def test_deterministic_and_allowed_keys_only(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            for t in (t1, t2):
                export_demo.dump(export_demo.strip_transient(export_one(GAME)), os.path.join(t, GAME + ".json"))
            with open(os.path.join(t1, GAME + ".json"), "rb") as f1, open(os.path.join(t2, GAME + ".json"), "rb") as f2:
                self.assertEqual(f1.read(), f2.read())

    def test_forbidden_key_is_refused(self):
        with self.assertRaises(SystemExit):
            export_demo.check_keys({"game_id": "x", "harness": {"passes": [{"lastEvent": 1}]}})
        with self.assertRaises(SystemExit):
            export_demo.check_keys({"replay_url": "x"})
        with self.assertRaises(SystemExit):
            export_demo.check_keys({"levels": [{"_hexes": []}]})

    def test_harness_rows_keep_only_allowed_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            art = os.path.join(tmp, "run", "output", "artifacts")
            os.makedirs(art)
            with open(os.path.join(art, "%s_p0_viewer_data.json" % GAME), "w", encoding="utf-8") as f:
                json.dump({"game_id": GAME, "pass_index": 0, "status": "won", "levels_completed": 6,
                           "total_levels": 6, "actions_per_level": [3, 8, 16, 12, 23, 25], "final_score": 100.0,
                           "lastEvent": {"transcript": "SECRET"}, "replay_url": "x", "agent_name": "y",
                           "viewer_steps": [1]}, f)
            rows = export_demo.harness_rows(os.path.join(tmp, "run"))
        self.assertEqual(list(rows), [GAME])
        self.assertEqual(set(rows[GAME][0]), set(export_demo.HARNESS_KEYS))
        self.assertNotIn("SECRET", json.dumps(rows))


if __name__ == "__main__":
    unittest.main(verbosity=1)
