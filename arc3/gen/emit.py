"""Emit a game plan as an ARCEngine game: <out>/<id>/<hash>/<id>.py + metadata.json, loadable by
arc_agi's offline Arcade (environments_dir=<out>). The runtime source is inlined so the file imports
only arcengine and numpy.
"""
import hashlib
import json
import os
import pprint

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = '''# Generated ARC-AGI-3-style game (house-side, uncertified unless cert.json says otherwise).
# MIT-0 for the emitted file; mechanics are ideas, not code, from the public set.
'''


def runtime_source(family="move"):
    with open(os.path.join(HERE, "runtime_%s.py" % family), encoding="utf-8") as f:
        src = f.read()
    # drop the header comment block; keep the imports and everything after
    lines = src.splitlines()
    i = 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:]) + "\n"


def class_name(gid):
    return gid[0].upper() + gid[1:4]


def game_source(plan, family="move"):
    """(source, hash): the emitted file's text and the 8-hex hash that names its directory."""
    gid = plan["game_id"]
    body = "CONFIG = %s\n\nLEVELS = %s\n\n" % (pprint.pformat(plan["config"], width=100), pprint.pformat(plan["levels"], width=100))
    body += runtime_source(family)
    cls = class_name(gid)
    base = "GenGame" if family == "move" else "GenClickGame"
    body += "\n\nclass %s(%s):\n    def __init__(self):\n        super().__init__(%r, CONFIG, LEVELS)\n" % (cls, base, gid)
    src = HEADER + body
    return src, hashlib.sha256(src.encode("utf-8")).hexdigest()[:8]


def emit_game(plan, out_dir, baselines=None, family="move", extra=None):
    """`extra`: fields added to metadata.json (the goal-rung pairs: goal, variant, pair_seed)."""
    gid = plan["game_id"]
    src, h = game_source(plan, family)
    d = os.path.join(out_dir, gid, h)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "%s.py" % gid), "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    meta = {"game_id": "%s-%s" % (gid, h), "title": gid, "default_fps": 30,
            "tags": ["keyboard"] if family == "move" else ["click"],
            "baseline_actions": list(baselines or [0] * len(plan["levels"])),
            "mechanics": plan["config"]["mechanics"], "seed": plan["config"]["seed"]}
    meta.update(extra or {})
    with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=1)
    return d, meta["game_id"]
