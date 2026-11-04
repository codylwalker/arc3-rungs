"""Did the model's code touch the file system? The leak check for runs on generated games.

Usage: python -I tools/file_access.py <run output dir> [--json out.json] [--lines 5]

On Kaggle the generated games' files sit under /kaggle/input beside the harness: each game's source
(`<id4>.py`), its `metadata.json` (the goal-rung pairs' labels `goal`, `variant` and `pair_seed` among them)
and the dataset's `games.json`. The harness never shows them to the model: it reads only `game_id` and
`baseline_actions` from an environment's info (`taaf/game_api.py`), and the model sees frames. The one
channel left is the model's own code in its Python sandbox. This scans every tool-call block of every
`*_events.jsonl` / `*_events.json.gz` for file-system access and introspection, and prints each hit with its
game and line. The patterns extend the ones the 2026-10-28 v2 check used by hand (the run of
2026-10-28). Only the model's code is read: the system prompt documents the tools and the tool
results echo paths, so neither counts.

Exit 0 when no line hits, 1 otherwise. A hit is a line to read, not a verdict: a no-op
`open('/tmp/x', 'w') if False else None` counts.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_run  # noqa: E402

PATTERNS = [
    ("open", r"(?<![\w.])open\s*\(|\b(?:io|os|codecs|gzip)\.open\s*\("),
    ("listing", r"\blistdir\b|\bscandir\b|\bos\.walk\b|\bglob\b|\biterdir\b|\brglob\b"),
    ("pathlib", r"\bpathlib\b|(?<![\w.])Path\s*\(|\.read_text\s*\(|\.read_bytes\s*\("),
    ("paths", r"/kaggle\b|\bmetadata\.json\b|\bgames\.json\b|\bGEN_GAMES\w*|\benvironment_files\b|\bcert\.json\b"
              r"|[\w/\\-]+\.py['\"]"),
    ("process", r"\bsubprocess\b|\bos\.system\b|\bos\.popen\b|\bpty\b"),
    ("introspection", r"\binspect\.|\bsys\.modules\b|\bgc\.get_objects\b|\bimportlib\b|__import__\s*\("),
]
COMPILED = [(name, re.compile(rx)) for name, rx in PATTERNS]
# the hits that matter: a line naming the inputs, whatever kind it was filed under
INPUTS = re.compile(dict(PATTERNS)["paths"])


def names_inputs(line):
    return bool(INPUTS.search(line))


def scan_code(code):
    """[(kind, line)] for each line of a code block that matches a pattern (first kind wins per line)."""
    hits = []
    for line in code.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        for name, rx in COMPILED:
            if rx.search(s):
                hits.append((name, s))
                break
    return hits


def scan_run(root):
    """{game key: {"hits": [(kind, line)], "code_blocks": n}} over a run's event files."""
    games = {}
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if not name.endswith(("_events.jsonl", "_events.json.gz")):
                continue
            path = os.path.join(dirpath, name)
            g = games.setdefault(analyze_run.game_key(path), {"hits": [], "code_blocks": 0, "seen": set()})
            for ev in analyze_run.load_events(path):
                if ev.get("type") != "analysis":
                    continue
                for label, arg, body in analyze_run.blocks(str(ev.get("transcript") or "")):
                    if not label.startswith(("ASSISTANT TOOL CALL", "TOOL CALL")):
                        continue
                    ident = analyze_run.call_id(label, arg, body)
                    if ident is not None:
                        if ident in g["seen"]:
                            continue
                        g["seen"].add(ident)
                    g["code_blocks"] += 1
                    g["hits"].extend(scan_code(body))
    for g in games.values():
        del g["seen"]
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--json", default=None)
    ap.add_argument("--lines", type=int, default=5, help="distinct hit lines printed per game")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    games = scan_run(a.run)
    blocks_total = sum(g["code_blocks"] for g in games.values())
    hit_games = {k: g for k, g in games.items() if g["hits"]}
    input_games = sorted(os.path.basename(k) for k, g in games.items() if any(names_inputs(l) for _, l in g["hits"]))
    print("game-passes %d; code blocks %d; game-passes with a hit %d; naming an input path %d %s" % (
        len(games), blocks_total, len(hit_games), len(input_games), input_games))
    for key, g in sorted(hit_games.items()):
        kinds = {}
        for kind, _ in g["hits"]:
            kinds[kind] = kinds.get(kind, 0) + 1
        print("%s: %d hit lines %s" % (os.path.basename(key), len(g["hits"]), kinds))
        shown = []
        for _, line in g["hits"]:
            if line not in shown:
                shown.append(line)
        for line in shown[: a.lines]:
            print("    " + line[:200])
    if a.json:
        out = {os.path.basename(k): {"code_blocks": g["code_blocks"], "hits": g["hits"]} for k, g in games.items()}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 1 if hit_games else 0


if __name__ == "__main__":
    sys.exit(main())
