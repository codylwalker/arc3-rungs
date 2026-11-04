"""Render frames of an emitted game to PNG for a visual check.

Usage: python render.py <game dir with metadata.json> [--level N] [--actions 1,2,6@20x26] --out shot.png
       python render.py <game dir> --cert [--level N] --out shot.png   (replays the oracle path of that level)
Draws the 64x64 frame at 6x with an approximate ARC palette, one panel per action step.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arcengine import ActionInput, GameAction  # noqa: E402

PALETTE = [(255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102), (51, 51, 51), (0, 0, 0),
           (229, 58, 163), (255, 123, 204), (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 133, 27),
           (255, 220, 0), (46, 204, 64), (160, 112, 0), (146, 18, 49)]


def load_game(d):
    meta = json.load(open(os.path.join(d, "metadata.json"), encoding="utf-8"))
    gid = meta["game_id"].split("-")[0]
    cls = gid[0].upper() + gid[1:4]
    ns = {"__name__": "arc_agi_3.%s" % gid}
    path = os.path.join(d, "%s.py" % gid)
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
    return ns[cls](), meta


def to_rgb(frame, k=6):
    arr = np.array(frame, dtype=np.int64)
    rgb = np.zeros(arr.shape + (3,), dtype=np.uint8)
    for i, c in enumerate(PALETTE):
        rgb[arr == i] = c
    return np.repeat(np.repeat(rgb, k, axis=0), k, axis=1)


def parse_action(tok):
    if isinstance(tok, list):
        if tok[0] == 6:
            return ActionInput(id=GameAction.ACTION6, data={"x": tok[1], "y": tok[2]})
        return ActionInput(id=GameAction.from_id(tok[0]))
    if "@" in tok:
        x, y = tok.split("@")[1].split("x")
        return ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)})
    return ActionInput(id=GameAction.from_id(int(tok)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game_dir")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--actions", default="")
    ap.add_argument("--cert", action="store_true")
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--cols", type=int, default=0, help="wrap the panels into rows of this many (0: one row)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    game, meta = load_game(a.game_dir)
    fd = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if a.level:
        game.set_level(a.level)
        game._score = a.level
        fd = game.perform_action(ActionInput(id=GameAction.ACTION7), raw=True)  # a no-op action renders the level
    if a.cert:
        cert = json.load(open(os.path.join(a.game_dir, "cert.json"), encoding="utf-8"))
        actions = [parse_action(t) for t in cert["levels"][a.level]["path"]]
    else:
        actions = [parse_action(t) for t in a.actions.split(",") if t]
    panels = [to_rgb(fd.frame[-1], a.scale)]
    for act in actions:
        fd = game.perform_action(act, raw=True)
        if fd.frame:
            panels.append(to_rgb(fd.frame[-1], a.scale))
    cols = a.cols or len(panels)
    blank = np.full_like(panels[0], 255)
    gap = np.full((panels[0].shape[0], 6, 3), 255, dtype=np.uint8)
    rows = []
    for i in range(0, len(panels), cols):
        chunk = panels[i:i + cols] + [blank] * (cols - len(panels[i:i + cols]) if len(panels) > cols else 0)
        row = chunk[0]
        for p in chunk[1:]:
            row = np.concatenate([row, gap, p], axis=1)
        rows.append(row)
    vgap = np.full((6, rows[0].shape[1], 3), 255, dtype=np.uint8)
    sheet = rows[0]
    for r in rows[1:]:
        sheet = np.concatenate([sheet, vgap, r], axis=0)
    Image.fromarray(sheet).save(a.out)
    print("wrote", a.out, "state", fd.state, "levels", fd.levels_completed, "/", fd.win_levels)


if __name__ == "__main__":
    main()
