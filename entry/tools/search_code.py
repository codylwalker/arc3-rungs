"""Which games' code holds a search the model wrote itself, and which call the twin tools' planner?

Usage: python -I tools/search_code.py RUN_DIR [RUN_DIR ...]

Reads the same `*_events.jsonl` / `*_events.json.gz` files as analyze_run.py and searches only the
tool-call blocks (the model's python code; the system prompt documents the tools and is never
searched). A game counts as writing its own search if its code contains a breadth-first queue
(`deque`), a priority queue (`heapq`), the word `bfs`, or a function whose name contains solve, search
or bfs. The twin tools' `plan` / `run_plan` are counted by analyze_run's call rule, the model's own
functions of those names apart. Per run: the games of each kind, the kinds of search found, and the
games with neither.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_run as ar  # noqa: E402

SEARCH = re.compile(r"\bdeque\b|\bheapq\b|\bbfs\b|\bdef\s+\w*(?:solve|search|bfs)\w*\s*\(", re.I)
PLANNER = ("plan", "run_plan")


def kind(match):
    text = match.lower()
    return "def solve/search" if text.startswith("def") else text


def scan(root):
    games = {}
    for dirpath, _, files in os.walk(root):
        for name in sorted(files):
            if not name.endswith(("_events.jsonl", "_events.json.gz")):
                continue
            path = os.path.join(dirpath, name)
            key = os.path.basename(ar.game_key(path))
            g = games.setdefault(key, {"search": set(), "planner_calls": 0, "defined": []})
            for ev in ar.load_events(path):
                if ev.get("type") != "analysis":
                    continue
                for label, _, body in ar.blocks(str(ev.get("transcript") or "")):
                    if not label.startswith(("ASSISTANT TOOL CALL", "TOOL CALL")):
                        continue
                    g["search"].update(kind(m.group(0)) for m in SEARCH.finditer(body))
                    for line in body.splitlines():
                        d = ar.DEFINE.match(line)
                        if d:
                            if d.group(1) not in g["defined"]:
                                g["defined"].append(d.group(1))
                            continue
                        g["planner_calls"] += sum(1 for t in ar.CALL.findall(line)
                                                  if t in PLANNER and t not in g["defined"])
    return games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    for root in args.runs:
        games = scan(root)
        own = sorted(k for k, g in games.items() if g["search"])
        tool = sorted(k for k, g in games.items() if g["planner_calls"])
        kinds = {}
        for g in games.values():
            for k in g["search"]:
                kinds[k] = kinds.get(k, 0) + 1
        print("%s | games %d | own search %d | twin planner called %d | both %d | neither %s | kinds %s" % (
            root, len(games), len(own), len(tool), len(set(own) & set(tool)),
            sorted(k for k, g in games.items() if not g["search"] and not g["planner_calls"]) or "none",
            dict(sorted(kinds.items()))))
        if tool:
            print("  planner called in: %s" % ", ".join("%s (%d%s)" % (
                k, games[k]["planner_calls"], ", own search too" if games[k]["search"] else "") for k in tool))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
