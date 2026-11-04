"""Export a set's certified paths as frames for the demo page (arc3-web/static/demo).

    python -I gen/export_demo.py --set gen/out/goal-set-a --out <arc3-web>/static/demo
                                 [--runs <arc3-entry>/out/kaggle/gen/goal-v2] [--claims gen/demo_claims.json] [--check]

Per game, `<game_id>.json`: the metadata, the palette, and for each level its certificate (optimal, baseline, budget,
nodes, secs, path) and the frames of the certified path, replayed through the real game exactly as render.py --cert
does (RESET for level 0; set_level + ACTION7 for a later level; then each action), so the exporter's test can compare
pixels with render.py's PNG. Frames are 64x64 colour indices 0-15: the level's first frame as run-length hex
(`<count hex4><colour hex1>` runs), each step as a sparse diff (`<index hex3><colour hex1>` cells), and a sha256 of
the final frame's 4096-nibble hex so a decoder can check itself. For the two games of a pair, each level carries
`band`, per step, the frame rows [first, last] where the variants' frames differ (the HUD's goal row as the camera
letterboxed it at that step; found from the frames, and refused if wider than MAX_BAND_ROWS or below the top third). `index.json`: the games, the pairs, the verdicts
of tools/goal_pairs.py (ratio and interval), and the claims file copied verbatim.

With --runs, each game gets its harness rows, joined by game id from `*_viewer_data.json`: ONLY pass_index, status,
levels_completed, total_levels, actions_per_level and final_score. Nothing from transcripts, tokens, notebook paths
or model names is read, and every key written is on ALLOWED; a key off the list stops the export.

Output is deterministic (sorted keys, no indentation, LF). --check re-exports to a temporary directory and compares
bytes with --out, so a stale committed export fails a build.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certify  # noqa: E402
import render  # noqa: E402
from arcengine import ActionInput, GameAction  # noqa: E402

SCHEMA = "arc3-demo/1"
SIDE = 64
HARNESS_KEYS = ("pass_index", "status", "levels_completed", "total_levels", "actions_per_level", "final_score")
ALLOWED = {
    "schema", "set", "sets", "game_id", "title", "seed", "mechanics", "goal", "variant", "pair", "pair_seed",
    "hud_rows", "cell", "grid", "palette", "licence", "game", "cert", "frames", "levels", "level", "optimal",
    "baseline", "budget", "nodes", "secs", "tries", "path", "f0", "d", "final_sha256", "replay_ok", "harness",
    "band", "source", "passes", "games", "pairs", "verdicts", "claims", "note", "id", "text", "value", "ci", "certifies",
    "does", "does_not", "label", "file", "generated_from", "manifest_sha256", "ratio", "game_passes", "signalled",
    "consequence", "name", *HARNESS_KEYS,
}


def hud_cell(game_py):
    with open(game_py, encoding="utf-8") as f:
        src = f.read()
    hud = re.search(r"^HUD = (\d+)", src, re.M)
    cell = re.search(r"^CELL = (\d+)", src, re.M)
    return (int(hud.group(1)) if hud else 4), (int(cell.group(1)) if cell else 3)


def frame_hex(frame):
    arr = np.asarray(frame, dtype=np.int64).reshape(-1)
    assert arr.size == SIDE * SIDE and arr.min() >= 0 and arr.max() <= 15, "a 64x64 frame of colours 0-15"
    return "".join("%x" % v for v in arr.tolist())


def rle(hexframe):
    out, i, n = [], 0, len(hexframe)
    while i < n:
        c, j = hexframe[i], i + 1
        while j < n and hexframe[j] == c:
            j += 1
        out.append("%04x%s" % (j - i, c))
        i = j
    return "".join(out)


def diff(prev_hex, cur_hex):
    return "".join("%03x%s" % (i, cur_hex[i]) for i in range(len(cur_hex)) if cur_hex[i] != prev_hex[i])


def replay_frames(game, level_index, path):
    """The level's first frame and the frame after each certified action, as render.py --cert sees them."""
    fd = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if level_index:
        game.set_level(level_index)
        game._score = level_index
        fd = game.perform_action(ActionInput(id=GameAction.ACTION7), raw=True)  # a no-op renders the level
    frames = [fd.frame[-1]]
    done = []
    for tok in path:
        fd = game.perform_action(render.parse_action(tok), raw=True)
        frames.append(fd.frame[-1] if fd.frame else frames[-1])
        done.append(int(fd.levels_completed))
    return frames, done


def export_game(game_dir, meta, cert):
    game, _ = render.load_game(game_dir)
    gid = meta["game_id"]
    hud, cell = hud_cell(os.path.join(game_dir, gid.split("-")[0] + ".py"))
    baselines = certify.baselines(cert["levels"])
    budgets = certify.budgets(cert["levels"])
    levels = []
    for i, lv in enumerate(cert["levels"]):
        frames, _ = replay_frames(game, i, lv["path"])
        hexes = [frame_hex(fr) for fr in frames]
        levels.append({
            "level": int(lv["level"]), "optimal": int(lv["optimal"]), "baseline": int(baselines[i]),
            "budget": int(budgets[i]), "nodes": int(lv.get("nodes", 0)), "secs": float(lv.get("secs", 0.0)),
            "tries": int(lv.get("tries", 0)), "path": lv["path"],
            "f0": rle(hexes[0]), "d": [diff(hexes[k - 1], hexes[k]) for k in range(1, len(hexes))],
            "final_sha256": hashlib.sha256(hexes[-1].encode("ascii")).hexdigest(),
            "_hexes": hexes,  # transient: the pair pass reads them; dump() never sees them
        })
        game, _ = render.load_game(game_dir)  # a fresh game per level: no state carries over
    return {
        "schema": SCHEMA, "game_id": gid, "title": meta.get("title", gid[:4]), "seed": meta.get("seed"),
        "mechanics": list(meta.get("mechanics", [])), "goal": meta.get("goal"), "variant": meta.get("variant"),
        "pair": cert.get("pair"), "pair_seed": meta.get("pair_seed"), "hud_rows": hud, "cell": cell,
        "grid": [SIDE, SIDE], "palette": [list(c) for c in render.PALETTE],
        "licence": {"game": "MIT-0", "cert": "CC0-1.0", "frames": "CC0-1.0"},
        "replay_ok": bool(cert.get("replay_ok", False)), "levels": levels,
    }


# The camera letterboxes and scales each level into the 64x64 frame (a 7x7 board fills it), so the HUD's one-pixel
# goal row lands on different frame rows per level, and a level's last frame is already the next level's first.
# So the band is per STEP: the rows where the two variants' frames differ at that step must be one thin strip in
# the frame's top third, where the HUD is drawn.
MAX_BAND_ROWS = 6
MAX_BAND_START = 22


def pair_bands(a, b):
    """Per level, per step, the band of frame rows [first, last] where the two variants' frames differ: the HUD's
    goal row as the camera drew it at that step, found from the frames and never assumed from the HUD's height.
    Refused if a step's difference is not one thin strip in the top third."""
    bands = []
    for la, lb in zip(a["levels"], b["levels"]):
        per_step = []
        for t, (ha, hb) in enumerate(zip(la["_hexes"], lb["_hexes"])):
            rows = {i // SIDE for i in range(SIDE * SIDE) if ha[i] != hb[i]}
            if not rows:
                raise SystemExit("%s / %s level %d step %d: the variants do not differ" % (
                    a["game_id"], b["game_id"], la["level"], t))
            r0, r1 = min(rows), max(rows)
            if r1 - r0 + 1 > MAX_BAND_ROWS or r0 > MAX_BAND_START:
                raise SystemExit("%s / %s level %d step %d: the variants differ over rows %d-%d, not the HUD's goal row"
                                 % (a["game_id"], b["game_id"], la["level"], t, r0, r1))
            per_step.append([r0, r1])
        bands.append(per_step)
    return bands


def strip_transient(doc):
    for lv in doc["levels"]:
        lv.pop("_hexes", None)
    return doc


def harness_rows(runs_dir):
    """{game_id: [row, ...]} from every *_viewer_data.json under runs_dir, the allowed keys only."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "**", "*_viewer_data.json"), recursive=True)):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        gid = d.get("game_id")
        if not gid:
            continue
        row = {k: d[k] for k in HARNESS_KEYS if k in d}
        rows.setdefault(gid, []).append(row)
    for gid in rows:
        rows[gid].sort(key=lambda r: r.get("pass_index", 0))
    return rows


def verdicts(runs_dir):
    for path in sorted(glob.glob(os.path.join(runs_dir, "**", "goal-pairs.json"), recursive=True)):
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        g2 = v.get("P-G2", {})
        return {"pairs": v.get("pairs"), "game_passes": v.get("game_passes"),
                "ratio": g2.get("ratio_consequence_over_signalled"), "ci": g2.get("ci")}
    return None


def check_keys(obj, where="root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ALLOWED:
                raise SystemExit("key %r at %s is not on the allow-list" % (k, where))
            check_keys(v, "%s.%s" % (where, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_keys(v, "%s[%d]" % (where, i))


def dump(obj, path):
    check_keys(obj, os.path.basename(path))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        f.write("\n")


def export(sets, out, runs_dir=None, claims_path=None):
    os.makedirs(out, exist_ok=True)
    rows = harness_rows(runs_dir) if runs_dir else {}
    index_games, pairs, mechanics, set_names, manifests, docs = [], {}, set(), [], [], {}
    for set_dir in sets:
        with open(os.path.join(set_dir, "manifest.json"), "rb") as f:
            raw = f.read()
        manifests.append({"name": os.path.basename(os.path.normpath(set_dir)),
                          "manifest_sha256": hashlib.sha256(raw).hexdigest()})
        set_names.append(manifests[-1]["name"])
        for g in json.loads(raw)["games"]:
            gid = g["game_id"]
            id4, sha8 = gid.split("-")[0], gid.split("-")[1]
            game_dir = os.path.join(set_dir, id4, sha8)
            with open(os.path.join(game_dir, "metadata.json"), encoding="utf-8") as f:
                meta = json.load(f)
            with open(os.path.join(game_dir, "cert.json"), encoding="utf-8") as f:
                cert = json.load(f)
            doc = export_game(game_dir, meta, cert)
            if gid in rows:
                doc["harness"] = {"source": os.path.basename(os.path.normpath(runs_dir)), "passes": rows[gid]}
            docs[gid] = doc
            mechanics.update(doc["mechanics"])
            index_games.append({
                "game_id": gid, "title": doc["title"], "set": set_names[-1], "seed": doc["seed"],
                "mechanics": doc["mechanics"], "goal": doc["goal"], "variant": doc["variant"], "pair": doc["pair"],
                "pair_seed": doc["pair_seed"], "levels": len(doc["levels"]),
                "optimal": [lv["optimal"] for lv in doc["levels"]],
                "baseline": [lv["baseline"] for lv in doc["levels"]],
                "budget": [lv["budget"] for lv in doc["levels"]],
                "replay_ok": doc["replay_ok"], "file": gid + ".json",
                "passes": len(rows.get(gid, [])),
            })
            if doc["variant"] and doc["pair_seed"] is not None:
                pairs.setdefault(doc["pair_seed"], {})[doc["variant"]] = gid
    # the pair pass: the goal row's band, found from the frames, written on both games of a pair
    for _, p in sorted(pairs.items()):
        if "signalled" in p and "consequence" in p and p["signalled"] in docs and p["consequence"] in docs:
            a, b = docs[p["signalled"]], docs[p["consequence"]]
            for la, lb, band in zip(a["levels"], b["levels"], pair_bands(a, b)):
                la["band"] = band
                lb["band"] = band
    for gid, doc in docs.items():
        dump(strip_transient(doc), os.path.join(out, gid + ".json"))
    index = {
        "schema": SCHEMA, "sets": manifests, "mechanics": sorted(mechanics),
        "games": sorted(index_games, key=lambda g: g["game_id"]),
        "pairs": [{"signalled": p.get("signalled"), "consequence": p.get("consequence")}
                  for _, p in sorted(pairs.items()) if "signalled" in p and "consequence" in p],
        "verdicts": verdicts(runs_dir) if runs_dir else None,
        "claims": None,
        "licence": {"game": "MIT-0", "cert": "CC0-1.0", "frames": "CC0-1.0"},
    }
    if claims_path:
        with open(claims_path, encoding="utf-8") as f:
            index["claims"] = json.load(f)
    dump(index, os.path.join(out, "index.json"))
    return index


def same_tree(a, b):
    """The files of a and b, byte for byte; the names that differ."""
    names = sorted(set(os.listdir(a)) | set(os.listdir(b)))
    bad = []
    for n in names:
        pa, pb = os.path.join(a, n), os.path.join(b, n)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            bad.append(n)
            continue
        with open(pa, "rb") as fa, open(pb, "rb") as fb:
            if fa.read() != fb.read():
                bad.append(n)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="a set dir with manifest.json (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=None, help="a run's out dir: *_viewer_data.json and goal-pairs.json under it")
    ap.add_argument("--claims", default=None, help="demo_claims.json, copied verbatim into index.json")
    ap.add_argument("--check", action="store_true", help="re-export to a temp dir and compare bytes with --out")
    a = ap.parse_args()
    if a.check:
        tmp = tempfile.mkdtemp(prefix="arc3-demo-")
        try:
            export(a.set, tmp, a.runs, a.claims)
            bad = same_tree(tmp, a.out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if bad:
            print("STALE: %d file(s) differ from a fresh export: %s" % (len(bad), ", ".join(bad[:8])))
            return 1
        print("check ok: %s matches a fresh export" % a.out)
        return 0
    index = export(a.set, a.out, a.runs, a.claims)
    total = sum(os.path.getsize(os.path.join(a.out, n)) for n in os.listdir(a.out))
    print("wrote %d games, %d pairs, %d mechanics -> %s (%.0f KB)" % (
        len(index["games"]), len(index["pairs"]), len(index["mechanics"]), a.out, total / 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
