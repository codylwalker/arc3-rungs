"""Where do the 25 public games draw their budget bars, and would the twin tools' mask catch them?

Plays each public game with a seeded random policy through arc_agi's offline Arcade (the T1 probe's
path; clicks aimed at non-background cells four times in five, so they hit something), keeps level 1's
(before, after) frames, and uses the EFFECTIVE ones (the frame changed). Per game:
  - bar-like colours found ANYWHERE: the frame-wide cell count changes on >= 10% of the effective
    actions and >= 90% of those changes go one way (a sprite keeps its count; a bar shrinks or grows),
    and the cells it entered or left are a thin strip (<= 3 rows or <= 3 columns) or lie >= 90% in
    the edge band; reported as a bounding box, the cells in the four-cell edge band
    (twin_tools.TWIN_BORDER), and the cells twin_tools._tw_auto_mask masks;
  - the mask computed on all level-1 transitions and on the effective ones.
Usage (from arc3/, where the public games are): python -I ../analysis/public_bars.py <entry dir> [--actions N] [--seeds K] [--json out]
"""
import argparse
import json
import logging
import random
import sys
from types import SimpleNamespace


def frame_of(obs):
    f = obs.frame[-1] if obs.frame else None
    return [[int(v) for v in r] for r in f] if f is not None else None


def counts(frame):
    out = {}
    for row in frame:
        for v in row:
            out[v] = out.get(v, 0) + 1
    return out


def click_target(rng, frame):
    if frame is not None and rng.random() < 0.8:
        cb = counts(frame)
        bg = max(cb, key=cb.get)
        cells = [(r, c) for r, row in enumerate(frame) for c, v in enumerate(row) if v != bg]
        if cells:
            r, c = rng.choice(cells)
            return {"x": c, "y": r}
    return {"x": rng.randint(0, 63), "y": rng.randint(0, 63)}


def collect(arc, game, seed, n, GameAction, GameState):
    rng = random.Random(seed)
    env = arc.make(game, seed=seed)
    obs = env.reset()
    trans = []
    for _ in range(n):
        acts = [a for a in env.action_space if a != GameAction.RESET]
        if not acts:
            break
        a = rng.choice(acts)
        before, lv = frame_of(obs), obs.levels_completed
        data = click_target(rng, before) if a == GameAction.ACTION6 else {}
        nobs = env.step(a, data=data)
        if nobs is None:
            break
        after = frame_of(nobs)
        terminal = nobs.state in (GameState.GAME_OVER, GameState.WIN) or nobs.levels_completed != lv
        if before is not None and after is not None and lv == 0 and not terminal:
            trans.append((before, after, a.name))
        obs = nobs
        if nobs.state == GameState.GAME_OVER:
            obs = env.reset()
        if nobs.levels_completed > 0 or nobs.state == GameState.WIN:
            break  # level 1 only
    return trans


def changed_cells(trans, colour):
    cells = set()
    for b, a, _ in trans:
        for r in range(len(b)):
            br, ar = b[r], a[r]
            if br != ar:
                for c in range(len(br)):
                    if br[c] != ar[c] and colour in (br[c], ar[c]):
                        cells.add((r, c))
    return cells


def bars(eff, border):
    if not eff:
        return []
    h, w = len(eff[0][0]), len(eff[0][0][0])
    ch = {}
    for b, a, _ in eff:
        cb, ca = counts(b), counts(a)
        for v in set(cb) | set(ca):
            d = ca.get(v, 0) - cb.get(v, 0)
            rec = ch.setdefault(v, {"changes": 0, "down": 0, "up": 0})
            if d:
                rec["changes"] += 1
                rec["down" if d < 0 else "up"] += 1
    out = []
    for v, rec in ch.items():
        if not rec["changes"] or rec["changes"] * 10 < len(eff):
            continue
        one_way = max(rec["down"], rec["up"]) / rec["changes"]
        if one_way < 0.9:
            continue
        cells = changed_cells(eff, v)
        if not cells:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        in_band = sum(1 for r, c in cells if r < border or r >= h - border or c < border or c >= w - border)
        thin = (max(rows) - min(rows) <= 2) or (max(cols) - min(cols) <= 2)
        if not (thin or in_band * 10 >= len(cells) * 9):
            continue
        out.append({"colour": v, "changes": rec["changes"], "of": len(eff), "one_way": round(one_way, 2),
                    "dir": "down" if rec["down"] >= rec["up"] else "up", "cells": len(cells),
                    "rows": [min(rows), max(rows)], "cols": [min(cols), max(cols)], "in_band": in_band,
                    "thin": thin, "_cells": cells})
    return sorted(out, key=lambda x: -x["changes"])


def mask_of(twin_tools, trans):
    if not trans:
        return set()
    tw = [SimpleNamespace(before_frame=b, after_frame=a, action=name, result={}) for b, a, name in trans]
    m = twin_tools._tw_auto_mask(tw, None, (len(trans[0][0]), len(trans[0][0][0])))
    return set(m or ())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("entry_dir")
    ap.add_argument("--actions", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, a.entry_dir)
    import twin_tools  # noqa: E402  (our own module, entry/)
    from arc_agi import Arcade, OperationMode  # noqa: E402
    from arcengine import GameAction, GameState  # noqa: E402
    logging.disable(logging.INFO)
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    ids = sorted(e.game_id.split("-")[0] for e in arc.get_environments())
    border = twin_tools.TWIN_BORDER
    report = {}
    for g in ids:
        trans = []
        for s in range(a.seeds):
            trans += collect(arc, g, s, a.actions, GameAction, GameState)
        eff = [t for t in trans if t[0] != t[1]]
        found = bars(eff, border)
        m_all, m_eff = mask_of(twin_tools, trans), mask_of(twin_tools, eff)
        for f in found:
            cells = f.pop("_cells")
            f["masked_all"] = len(cells & m_all)
            f["masked_eff"] = len(cells & m_eff)
        report[g] = {"transitions": len(trans), "effective": len(eff), "bars": found,
                     "mask_all": len(m_all), "mask_eff": len(m_eff)}
        line = "%s: %3d transitions, %3d effective, mask %d/%d | " % (g, len(trans), len(eff), len(m_all), len(m_eff))
        line += " ; ".join(
            "c%d %s %d/%d cells %d rows %d-%d cols %d-%d band %d masked %d/%d" % (
                f["colour"], f["dir"], f["changes"], f["of"], f["cells"], f["rows"][0], f["rows"][1],
                f["cols"][0], f["cols"][1], f["in_band"], f["masked_all"], f["masked_eff"]) for f in found) \
            or "no bar-like colour"
        print(line, flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
