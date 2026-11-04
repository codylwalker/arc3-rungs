"""Re-emit a set's games with the current runtime, keeping each game's levels, oracle paths and baselines.
A movement game whose palette clashes (palette_check) is repainted with move_palette, seeded by its seed.
Every oracle path is replayed on the new file before the old <id>/<hash>/ dir is removed; a game whose
replay fails keeps its old dir. The manifest's entries follow the new ids.

Usage (from tools/arc3): python gen/reemit.py gen/out/set-a [--dry-run]
"""
import argparse
import json
import os
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import certify  # noqa: E402
import compose  # noqa: E402
import corpus  # noqa: E402
import emit  # noqa: E402
import palette_check  # noqa: E402


def game_dirs(set_dir):
    out = []
    for gid in sorted(os.listdir(set_dir)):
        gdir = os.path.join(set_dir, gid)
        if not os.path.isdir(gdir):
            continue
        for h in sorted(os.listdir(gdir)):
            d = os.path.join(gdir, h)
            if os.path.isfile(os.path.join(d, "%s.py" % gid)) and os.path.isfile(os.path.join(d, "cert.json")):
                out.append((gid, h, d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    set_dir = os.path.abspath(a.set_dir)
    renames, kept, failed = {}, 0, []
    for gid, h, d in game_dirs(set_dir):
        with open(os.path.join(d, "%s.py" % gid), encoding="utf-8") as f:
            src = f.read()
        cfg, levels = palette_check.literal(src, "CONFIG"), palette_check.literal(src, "LEVELS")
        with open(os.path.join(d, "metadata.json"), encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(d, "cert.json"), encoding="utf-8") as f:
            cert = json.load(f)
        family = "click" if "kind" in cfg else "move"
        why = []
        for lv in cert["levels"]:  # one path format across a set: [id] or [6, x, y]
            lv["path"] = [[r] if isinstance(r, int) else r for r in lv["path"]]
        if family == "move":
            found = palette_check.clashes(cfg, levels)
            if found:
                cfg = dict(cfg)
                cfg["palette"] = compose.move_palette(random.Random(cfg["seed"] * 7919 + 101), cfg["mechanics"])
                assert not palette_check.clashes(cfg, levels), (gid, palette_check.clashes(cfg, levels))
                why.append("repaint (%s)" % " ".join(found))
        plan = {"game_id": gid, "config": cfg, "levels": levels}
        cls_name = emit.class_name(gid)
        _src, new_h = emit.game_source(plan, family)
        if new_h == h:
            kept += 1
            continue
        why.append("runtime")
        print("%s-%s -> %s-%s: %s" % (gid, h, gid, new_h, ", ".join(why)))
        if a.dry_run:
            continue
        nd, full_id = emit.emit_game(plan, set_dir, baselines=meta["baseline_actions"], family=family)
        assert os.path.basename(nd) == new_h, (nd, new_h)
        cls = corpus.load_emitted(os.path.join(nd, "%s.py" % gid), cls_name)
        ok = certify.replay(cls(), [lv["path"] for lv in cert["levels"]])
        if not ok:
            shutil.rmtree(nd)
            failed.append(gid)
            print("  replay FAILED on the new file; kept %s" % h)
            continue
        old_id = cert["game_id"]
        cert["game_id"], cert["replay_ok"] = full_id, True
        cert.setdefault("reemitted", []).append({"from": old_id, "why": why})
        with open(os.path.join(nd, "cert.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(cert, f, indent=1)
        assert os.path.dirname(d) == os.path.join(set_dir, gid) and d != nd
        shutil.rmtree(d)
        renames[old_id] = cert
    man_path = os.path.join(set_dir, "manifest.json")
    if renames and os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            man = json.load(f)
        man["games"] = [renames.get(g["game_id"], g) for g in man["games"]]
        with open(man_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(man, f, indent=1)
    print("re-emitted %d, unchanged %d, replay failures %d%s" % (len(renames), kept, len(failed),
                                                                (" " + ",".join(failed)) if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
