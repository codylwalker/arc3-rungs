"""Did the model use twin_tools, and what did they report? Reads a harness run's output directory.

Usage: python -I tools/analyze_run.py <run output dir> [--json out.json]

Reads every `*_events.jsonl` / `*_events.json.gz` (the harness's per-game viewer events: "analysis"
events carry the step's transcript, "action" events the flags) and `*_viewer_data.json` (levels and
score per game). Per game and in total: calls of each tool in the model's code, check_model match
rates from its summaries, plan statuses, run_plan stops, actions, levels, score.

Transcripts are labelled blocks. In the real runs (2026-10-22 v2) one analysis event is one step:
`[SYSTEM PROMPT]`, `[USER PROMPT]`, `[MODEL RESPONSE META]`, `[THINKING]`, `[ASSISTANT]`,
`[TOOL CALL: python]` (the code), `[TOOL RESULT: python]`, `[ANALYZER STATUS]`. The renderer in the
source can also emit `[ASSISTANT TOOL CALL: name]` with an `id:` line; those are deduplicated by id. The
system prompt documents the tools, so only tool-call blocks are searched for calls and only tool-result
blocks for results. (v1: counting the whole transcript gave 110 "calls" of every tool, all of them the
prompt's documentation; v2 with only the source's label form found no call blocks at all.)

The model also writes functions of its own under the tools' names: in the 2026-10-28 generated-games
run with the tools OFF, `def plan(fr):` (a BFS over its own board model) produced 21 "plan" and 11
"run_plan" calls. A `def <tool>(` line is a definition, not a call, and once a game's code has defined a
name, later calls of it go to the model's function (the namespace persists across steps), so they are
counted apart as `own_calls`, in order: a call before the definition is still the tool's.
"""
import argparse
import gzip
import json
import os
import re
import sys

TOOLS = ("check_model", "check_goal", "plan", "run_plan", "best_probe")
CALL = re.compile(r"(?<![\w.])(%s)\(" % "|".join(TOOLS))
DEFINE = re.compile(r"^\s*def\s+(%s)\s*\(" % "|".join(TOOLS))
REPRODUCES = re.compile(r"predict reproduces (\d+) of (\d+) checked transitions")
PLAN_STATUS = re.compile(r"['\"]status['\"]\s*:\s*['\"](found|exhausted|node_limit|time_limit|predict_error|goal_error)['\"]")
RUN_STOP = re.compile(r"['\"]stopped['\"]\s*:\s*['\"](mismatch|level_completed|game_over|run_complete|done|predict_error)['\"]")
PROBE = re.compile(r"splits the (\d+) hypotheses")
# the unasked certification line (ARC3_AUTO_CERTIFY, twin_auto_certify): "all" or "N of M"
CERTIFY = re.compile(r"\[certify\] (\w+) reproduces (?:all (\d+)|(\d+) of (\d+)) logged transitions")
META_CHARS = re.compile(r"^(?:reasoning|content)_chars:\s*(\d+)\s*$", re.M)  # one [MODEL RESPONSE META] per model reply
HEADER = re.compile(r"^\[([A-Z][A-Z _]*?)(?::\s*([^\]]*))?\]\s*$")


def blocks(text):
    """(label, header argument, body) for each labelled block of a transcript."""
    out = []
    label, arg, body = None, None, []
    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            if label is not None:
                out.append((label, arg, "\n".join(body)))
            label, arg, body = m.group(1).strip(), (m.group(2) or "").strip(), []
        elif label is not None:
            body.append(line)
    if label is not None:
        out.append((label, arg, "\n".join(body)))
    return out


TOOL_NAMES = {"python"}


def call_id(label, arg, body):
    """A real call id if the block carries one; None otherwise (then the block is never deduplicated).

    The 2026-10-22 v2 transcripts label blocks `[TOOL CALL: python]` / `[TOOL RESULT: python]` with no
    id, one step per analysis event, so there is nothing to dedupe; the source's
    `[ASSISTANT TOOL CALL: name]` + `id:` form, which can repeat across steps, is deduped by id.
    """
    if label.startswith("TOOL RESULT"):
        return arg if arg and arg not in TOOL_NAMES else None
    m = re.search(r"^id:\s*(\S+)", body, re.M)
    return m.group(1) if m else None


def load_events(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    events = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    pass
    return events


def game_key(path):
    name = os.path.basename(path)
    for suffix in ("_events.jsonl", "_events.json.gz", "_viewer_data.json"):
        if name.endswith(suffix):
            return os.path.join(os.path.dirname(path), name[: -len(suffix)])
    return path


def new_game():
    return {"calls": {t: 0 for t in TOOLS}, "own": {t: 0 for t in TOOLS}, "defined": [], "call_lines": set(),
            "reproduces": [], "plan": {}, "run_plan": {}, "probes": 0, "analysis_steps": 0,
            "actions": 0, "game_overs": 0, "tool_calls": 0, "gen_chars": {}, "seen": set(),
            "certify": [0, 0]}  # [certify] lines seen, and how many said "all"


def analyse(root):
    games = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            path = os.path.join(dirpath, name)
            key = game_key(path)
            if name.endswith(("_events.jsonl", "_events.json.gz")):
                g = games.setdefault(key, new_game())
                for ev in load_events(path):
                    if ev.get("type") == "analysis":
                        g["analysis_steps"] += 1
                        level = str(ev.get("level"))
                        for label, arg, body in blocks(str(ev.get("transcript") or "")):
                            if label.startswith("MODEL RESPONSE META"):
                                # what the model generated, by the level being played: reasoning + reply here,
                                # the code below (a token proxy; one step per event, as in the v2 format)
                                n = sum(int(c) for c in META_CHARS.findall(body))
                                g["gen_chars"][level] = g["gen_chars"].get(level, 0) + n
                                continue
                            is_call = label.startswith(("ASSISTANT TOOL CALL", "TOOL CALL"))
                            is_result = label.startswith("TOOL RESULT")
                            if not (is_call or is_result):
                                continue
                            ident = call_id(label, arg, body)
                            if ident is not None:
                                mark = ("result" if is_result else "call", ident)
                                if mark in g["seen"]:
                                    continue
                                g["seen"].add(mark)
                            if is_call:
                                g["tool_calls"] += 1
                                g["gen_chars"][level] = g["gen_chars"].get(level, 0) + len(body)
                                for line in body.splitlines():
                                    d = DEFINE.match(line)
                                    if d:
                                        if d.group(1) not in g["defined"]:
                                            g["defined"].append(d.group(1))
                                        continue
                                    hits = CALL.findall(line)
                                    for tool in hits:
                                        g["own" if tool in g["defined"] else "calls"][tool] += 1
                                    if any(tool not in g["defined"] for tool in hits):
                                        g["call_lines"].add(line.strip())
                            else:
                                g["reproduces"] += [(int(a), int(b)) for a, b in REPRODUCES.findall(body)]
                                for s in PLAN_STATUS.findall(body):
                                    g["plan"][s] = g["plan"].get(s, 0) + 1
                                for s in RUN_STOP.findall(body):
                                    g["run_plan"][s] = g["run_plan"].get(s, 0) + 1
                                g["probes"] += len(PROBE.findall(body))
                                for _name, all_n, _got, _of in CERTIFY.findall(body):
                                    g["certify"][0] += 1
                                    g["certify"][1] += 1 if all_n else 0
                    elif ev.get("type") == "action":
                        g["actions"] += 1
                        g["game_overs"] += 1 if ev.get("game_over") else 0
            elif name.endswith("_viewer_data.json"):
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                except ValueError:
                    continue
                g = games.setdefault(key, new_game())
                for field in ("game_id", "levels_completed", "total_levels", "actions_per_level", "final_score"):
                    if field in data:
                        g[field] = data[field]
    return games


def summarise(games):
    total = {t: 0 for t in TOOLS}
    own = {t: 0 for t in TOOLS}
    used = 0
    rows = []
    for key, g in sorted(games.items()):
        for t in TOOLS:
            total[t] += g["calls"][t]
            own[t] += g["own"][t]
        any_use = any(g["calls"].values())
        used += 1 if any_use else 0
        rep = g["reproduces"]
        rate = (sum(a for a, _ in rep) / max(1, sum(b for _, b in rep))) if rep else None
        rows.append({
            "game": g.get("game_id") or os.path.basename(key), "levels": g.get("levels_completed"),
            "of": g.get("total_levels"), "score": g.get("final_score"), "actions": g["actions"],
            "game_overs": g["game_overs"], "steps": g["analysis_steps"], "python_calls": g["tool_calls"],
            "calls": g["calls"], "distinct_call_lines": len(g["call_lines"]),
            "self_defined": g["defined"], "own_calls": g["own"], "gen_chars_by_level": g["gen_chars"],
            "check_model_match_rate": None if rate is None else round(rate, 3),
            "plan": g["plan"], "run_plan": g["run_plan"], "probes": g["probes"],
            "certify_lines": g["certify"][0], "certify_full": g["certify"][1],
        })
    return {"games": len(games), "games_using_tools": used, "calls": total, "own_calls": own,
            "games_defining_own": sum(1 for g in games.values() if g["defined"]), "per_game": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = summarise(analyse(args.root))
    print("games %d, using twin_tools %d; calls %s; the model's own functions of those names in %d games, called %s" % (
        out["games"], out["games_using_tools"], out["calls"], out["games_defining_own"],
        {k: v for k, v in out["own_calls"].items() if v}))
    for r in out["per_game"]:
        print("  %-12s levels %s/%s score %s actions %s | calls %s | own %s | check_model %s | plan %s | run_plan %s | certify %d (%d all)" % (
            r["game"], r["levels"], r["of"], r["score"], r["actions"],
            {k: v for k, v in r["calls"].items() if v}, {k: v for k, v in r["own_calls"].items() if v},
            r["check_model_match_rate"], r["plan"], r["run_plan"], r["certify_lines"], r["certify_full"]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
