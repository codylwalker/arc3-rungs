"""Harness run -> rejection-sampled SFT records (one record per model response, with its context).

Usage: python -I tools/traj_to_sft.py <run output dir> [<more run dirs>] --out <dir>
           [--slack 1.25] [--context-chars 110000] [--upscale 10] [--no-images] [--all]

Reads every `artifacts/*_events.jsonl` (or `.json.gz`) the harness wrote: the "analysis" events carry one
turn's transcript (labelled blocks: `[SYSTEM PROMPT]`, `[USER PROMPT]`, then per model response
`[MODEL RESPONSE META]` with `raw_tool_calls:` JSON, `[THINKING]`, `[ASSISTANT]`, `[TOOL CALL: python]`,
`[TOOL RESULT: python]`), and the "action" events carry `level_completed` and `action_num`. When the run
also has `prompts/<stem>_requests.jsonl` (demo runs set `save_request_logs`), the exact request
`messages` (trimmed history, images as the harness sent them) replace the reconstructed context.

Selection (the rejection step), per game and level across every run dir given: the level was completed,
its action count is within `--slack` of the best count seen for that game and level, the response
finished (`finish_reason` tool_calls or stop) and carries reasoning or a tool call. `--all` writes every
record with a `keep` flag and a `drop` reason instead.

Record: {"game", "pass", "level", "analysis_step", "response_index", "level_actions", "best_actions",
"finish_reason", "context_source": "requests" | "events", "images": n, "approx_tokens",
"messages": [...OpenAI chat messages, the system prompt first...],
"completion": {"reasoning_content", "content", "tool_calls"}}. Training is on the completion only.

Reconstructed context (events): the system prompt, the previous turns' user prompts, assistant
responses and tool results, oldest turns dropped to fit `--context-chars`, the current turn's messages
before this response, and the current grid image rendered from the turn's `board` the way the harness
renders it (PIL; `--no-images` or no PIL leaves the text alone). Diff images are not reconstructed.

Output: <out>/<run label>__<game>_p<pass>.jsonl.gz and <out>/index.json (per game and level: actions,
completed, records kept, drop reasons). Stdlib only; PIL optional. Runs locally (on the downloaded
`output/` dir) and as a notebook cell at the end of a demo run (`/kaggle/working`, with `--all`); then
`--reselect <sft dir> [...] --out <merged>` run locally applies the rule across every session, since a
session's best action counts are only its own. Run with -I on downloaded files.

The data contains competition boards and the competition system prompt: it stays private (Kaggle rules
4b, no redistribution of Competition Data). Only an adapter trained on it may be shared.
"""
import argparse
import base64
import collections
import glob
import gzip
import io
import json
import math
import os
import re
import sys

HEADER = re.compile(r"^\[([A-Z][A-Z _]*?)(?::\s*([^\]]*))?\]\s*$")
TURN_HEADER = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| [^|]* \| [^-]*---\s*$")
STEM = re.compile(r"^(?P<game>.+?)_p(?P<pass>\d+)_events\.jsonl?(?:\.gz)?$")
META_INT = ("tool_call_count", "content_chars", "reasoning_chars")
CHARS_PER_TOKEN = 4.3     # measured by the harness: 25,189 chars -> 5,892 tokens
IMAGE_TOKENS = 402        # a 64-cell board at upscale 10
GRID_LABEL = "\n\nCurrent grid image:"
PALETTE = {0: (255, 255, 255), 1: (204, 204, 204), 2: (153, 153, 153), 3: (102, 102, 102), 4: (51, 51, 51),
           5: (0, 0, 0), 6: (229, 58, 163), 7: (255, 123, 204), 8: (249, 60, 49), 9: (30, 147, 255),
           10: (136, 216, 241), 11: (255, 220, 0), 12: (255, 133, 27), 13: (146, 18, 49), 14: (79, 204, 48),
           15: (163, 86, 214)}


# ---------------------------------------------------------------- inputs

def run_labels(dirs, labels=None):
    """One label per run dir for file names: --label values, else distinct basenames, else r0, r1, ..."""
    if labels:
        if len(labels) != len(dirs):
            raise SystemExit("--label needs one value per run dir")
        return list(labels)
    names = [os.path.basename(os.path.normpath(d)) for d in dirs]
    if len(set(names)) == len(names) and all(names):
        return names
    return ["r%d" % i for i in range(len(dirs))]


def find_runs(dirs, labels=None):
    """(label, game, pass, events path, requests path or None) for every events file under the run dirs."""
    out = []
    for label, d in zip(run_labels(dirs, labels), dirs):
        for path in sorted(glob.glob(os.path.join(d, "**", "*_events.jsonl*"), recursive=True)):
            m = STEM.match(os.path.basename(path))
            if not m:
                continue
            stem = os.path.basename(path).split("_events.")[0]
            req = None
            for cand in glob.glob(os.path.join(d, "**", stem + "_requests.jsonl"), recursive=True):
                req = cand
                break
            out.append((label, m.group("game"), int(m.group("pass")), path, req))
    return out


def load_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if path.endswith(".json") or path.endswith(".json.gz"):
        data = json.loads(text)
        return data if isinstance(data, list) else data.get("events", [])
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def level_stats(events):
    """{level: {"completed": bool, "actions": n}} from the action events; level numbers are 1-based.

    The action count of level L is the action_num at its completion minus that at L-1's completion, the
    same count the scorer divides the baseline by. An unfinished last level gets the actions spent so far.
    """
    stats = {}
    prev_num, level = 0, 1
    for e in events:
        if e.get("type") != "action":
            continue
        num = int(e.get("action_num") or 0)
        if e.get("level_completed"):
            done = int(e.get("score") or level)
            stats[done] = {"completed": True, "actions": num - prev_num}
            prev_num, level = num, done + 1
        else:
            stats.setdefault(level, {"completed": False, "actions": 0})["actions"] = num - prev_num
    return stats


# ---------------------------------------------------------------- transcripts

def blocks(text):
    """(label, header argument, body) per labelled block; the turn header line is skipped."""
    out, label, arg, body = [], None, None, []
    for line in text.splitlines():
        if TURN_HEADER.match(line):
            continue
        m = HEADER.match(line)
        if m:
            if label is not None:
                out.append((label, arg, "\n".join(body).strip("\n")))
            label, arg, body = m.group(1).strip(), (m.group(2) or "").strip(), []
        elif label is not None:
            body.append(line)
    if label is not None:
        out.append((label, arg, "\n".join(body).strip("\n")))
    return out


def parse_meta(body):
    meta, lines = {}, body.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("raw_tool_calls:"):
            rest = "\n".join(lines[i + 1:]).strip()
            try:
                meta["tool_calls"] = json.loads(rest) if rest else []
            except ValueError:
                meta["tool_calls"] = []
            break
        if ":" in line:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            meta[k] = int(v) if k in META_INT and v.isdigit() else v
    return meta


def tool_call_from_markup(name, body, index):
    """A tool call from the rendered `<parameter=code>` markup, for a META block without raw_tool_calls."""
    m = re.search(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", body, re.S)
    args = {m.group(1): m.group(2)} if m else {}
    return {"id": "call_%d" % index, "index": index, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def parse_turn(text):
    """One analysis transcript -> (system prompt, [messages and responses in order]).

    The list holds dicts: {"role": "user", "content": text}, {"role": "tool", "content": text} and
    {"role": "assistant", "response": True, "meta": {...}, "reasoning_content": str, "content": str,
    "tool_calls": [...]}. Each assistant entry is one model response, a training example.
    """
    system, seq, cur = "", [], None
    for label, arg, body in blocks(text):
        if label == "SYSTEM PROMPT":
            system = body
        elif label == "USER PROMPT":
            seq.append({"role": "user", "content": body})
            cur = None
        elif label == "MODEL RESPONSE META":
            cur = {"role": "assistant", "response": True, "meta": parse_meta(body), "reasoning_content": "",
                   "content": "", "tool_calls": [], "_markup_calls": 0}
            cur["tool_calls"] = list(cur["meta"].pop("tool_calls", []) or [])
            seq.append(cur)
        elif label == "THINKING" and cur is not None:
            cur["reasoning_content"] = body
        elif label == "ASSISTANT" and cur is not None:
            cur["content"] = body
        elif label.startswith("TOOL CALL") and cur is not None:
            # raw_tool_calls (from the META block) is exact; markup is the fallback for a META without it
            if len(cur["tool_calls"]) <= cur["_markup_calls"]:
                cur["tool_calls"].append(tool_call_from_markup(arg or "python", body, len(cur["tool_calls"])))
            cur["_markup_calls"] += 1
        elif label.startswith("TOOL RESULT"):
            calls = cur["tool_calls"] if cur is not None else []
            n = sum(1 for s in seq if s["role"] == "tool" and s.get("_of") is cur)
            call_id = calls[n]["id"] if n < len(calls) else ""
            seq.append({"role": "tool", "tool_call_id": call_id, "content": body, "_of": cur})
    for s in seq:
        s.pop("_of", None)
        s.pop("_markup_calls", None)
    return system, seq


def assistant_message(resp):
    msg = {"role": "assistant"}
    if resp["reasoning_content"]:
        msg["reasoning_content"] = resp["reasoning_content"]
    if resp["content"]:
        msg["content"] = resp["content"]
    if resp["tool_calls"]:
        msg["tool_calls"] = resp["tool_calls"]
    return msg


def as_message(entry):
    if entry["role"] == "assistant":
        return assistant_message(entry)
    return {k: v for k, v in entry.items() if k in ("role", "content", "tool_call_id")}


# ---------------------------------------------------------------- images

def grid_png_data_url(board, upscale):
    try:
        from PIL import Image
    except ImportError:
        return None
    rows, cols = len(board), max(len(r) for r in board)
    img = Image.new("RGB", (cols, rows), PALETTE[0])
    px = img.load()
    for y, row in enumerate(board):
        for x in range(cols):
            px[x, y] = PALETTE.get(int(row[x]) if x < len(row) else 0, PALETTE[0])
    if upscale > 1:
        img = img.resize((cols * upscale, rows * upscale), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def with_image(user_msg, data_url):
    if data_url is None:
        return user_msg
    return {"role": "user", "content": [{"type": "text", "text": user_msg["content"] + GRID_LABEL},
                                        {"type": "image_url", "image_url": {"url": data_url}}]}


# ---------------------------------------------------------------- sizes

def msg_chars(msg):
    c = msg.get("content")
    n = 0
    if isinstance(c, str):
        n += len(c)
    elif isinstance(c, list):
        n += sum(len(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
    n += len(msg.get("reasoning_content") or "")
    for tc in msg.get("tool_calls") or []:
        n += len(json.dumps(tc))
    return n


def msg_images(msg):
    c = msg.get("content")
    return sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image_url") if isinstance(c, list) else 0


def approx_tokens(messages):
    return int(sum(msg_chars(m) for m in messages) / CHARS_PER_TOKEN) + IMAGE_TOKENS * sum(msg_images(m) for m in messages)


def fit_history(system, history_turns, current, budget_chars):
    """System + the newest whole prior turns that fit + the current turn's messages."""
    fixed = len(system) + sum(msg_chars(m) for m in current)
    kept, used = [], fixed
    for turn in reversed(history_turns):
        size = sum(msg_chars(m) for m in turn)
        if used + size > budget_chars:
            break
        kept.insert(0, turn)
        used += size
    out = [{"role": "system", "content": system}]
    for turn in kept:
        out.extend(turn)
    out.extend(current)
    return out


# ---------------------------------------------------------------- requests.jsonl

def load_requests(path):
    """{(analysis_step, request index): (messages, finish_reason)} from a per-game requests log."""
    out = {}
    for rec in load_jsonl(path):
        key = (rec.get("analysis_step"), rec.get("request_index_within_turn"))
        if rec.get("event") == "request":
            out.setdefault(key, [None, None])[0] = rec.get("messages")
        elif rec.get("event") == "response":
            out.setdefault(key, [None, None])[1] = rec.get("finish_reason")
    return out


# ---------------------------------------------------------------- the conversion

def select(resp_tool_calls, resp_reasoning, resp_content, completed, actions, best_actions, slack, finish):
    """The rejection rule; None keeps the record, else the drop reason."""
    if not completed:
        return "level_unfinished"
    if best_actions and actions > slack * best_actions:
        return "actions_over_slack"
    if finish not in ("tool_calls", "stop"):
        return "finish_" + (finish or "unknown")
    if not resp_tool_calls and not resp_reasoning and not resp_content:
        return "empty"
    return None


def records_for_run(label, game, pass_no, events, requests, best, slack, context_chars, upscale, images, keep_all):
    stats = level_stats(events)
    history, out, drops = [], [], collections.Counter()
    for e in events:
        if e.get("type") != "analysis":
            continue
        step = e.get("analysis_step")
        level = int(e.get("score") or 0) + 1
        system, seq = parse_turn(e.get("transcript") or "")
        if not system or not seq:
            drops["unparsed"] += 1
            continue
        data_url = grid_png_data_url(e.get("board") or [], upscale) if images and e.get("board") else None
        # the turn's own messages, the current grid image on its first user prompt
        turn_msgs, first_user = [], True
        responses = []
        for entry in seq:
            if entry["role"] == "user":
                msg = with_image(as_message(entry), data_url) if first_user else as_message(entry)
                first_user = False
                turn_msgs.append(msg)
            elif entry["role"] == "assistant":
                responses.append((len(turn_msgs), entry))
                turn_msgs.append(as_message(entry))
            else:
                turn_msgs.append(as_message(entry))
        st = stats.get(level, {"completed": False, "actions": 0})
        best_actions = best.get((game, level))
        for idx, (pos, resp) in enumerate(responses):
            meta = resp["meta"]
            finish = str(meta.get("finish_reason") or "")
            req = requests.get((step, idx)) if requests else None
            if req and req[1]:
                finish = str(req[1])
            drop = select(resp["tool_calls"], resp["reasoning_content"], resp["content"], st["completed"],
                          st["actions"], best_actions, slack, finish)
            if drop and not keep_all:
                drops[drop] += 1
                continue
            if req and req[0]:
                messages, source = req[0], "requests"
            else:
                messages, source = fit_history(system, history, turn_msgs[:pos], context_chars), "events"
            rec = {"run": label, "game": game, "pass": pass_no, "level": level, "analysis_step": step,
                   "response_index": idx, "level_completed": st["completed"],
                   "level_actions": st["actions"], "best_actions": best_actions, "finish_reason": finish,
                   "context_source": source, "images": sum(msg_images(m) for m in messages),
                   "approx_tokens": approx_tokens(messages), "messages": messages,
                   "completion": {"reasoning_content": resp["reasoning_content"], "content": resp["content"],
                                  "tool_calls": resp["tool_calls"]}}
            if keep_all:
                rec["keep"] = drop is None
                rec["drop"] = drop
                drops[drop or "kept"] += 1
            out.append(rec)
        history.append(turn_msgs)
    return out, stats, drops


def best_actions_over(runs):
    """{(game, level): fewest actions of a completed level} across every run given."""
    best = {}
    loaded = {}
    for _, game, pass_no, path, _ in runs:
        events = load_jsonl(path)
        loaded[path] = events
        for level, st in level_stats(events).items():
            if st["completed"]:
                key = (game, level)
                best[key] = min(best.get(key, math.inf), st["actions"])
    return best, loaded


def reselect(sft_dirs, out, slack):
    """Merge `--all` outputs of several sessions and apply the rejection rule across all of them.

    The in-notebook pass sees one session, so its best action counts are that session's; a local
    pass re-selects over every session's index.json (completed levels and their actions) and records.
    """
    best, files = {}, []
    for d in sft_dirs:
        with open(os.path.join(d, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
        for run in index["runs"]:
            for level, st in run["levels"].items():
                if st["completed"]:
                    key = (run["game"], int(level))
                    best[key] = min(best.get(key, math.inf), st["actions"])
            files.append((d, run["file"]))
    os.makedirs(out, exist_ok=True)
    total, tokens, drops, games = 0, 0, collections.Counter(), set()
    for d, name in files:
        kept = []
        with gzip.open(os.path.join(d, name), "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                c = r["completion"]
                drop = select(c["tool_calls"], c["reasoning_content"], c["content"], r["level_completed"],
                              r["level_actions"], best.get((r["game"], r["level"])), slack, r["finish_reason"])
                if drop:
                    drops[drop] += 1
                    continue
                r["best_actions"] = best.get((r["game"], r["level"]))
                r.pop("keep", None)
                r.pop("drop", None)
                kept.append(r)
        label = os.path.basename(os.path.normpath(d))
        with gzip.open(os.path.join(out, "%s__%s" % (label, name)), "wt", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
        total += len(kept)
        tokens += sum(int(r["approx_tokens"]) + len(json.dumps(r["completion"])) // 4 for r in kept)
        games.update(r["game"] for r in kept)
    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"reselected_from": sft_dirs, "slack": slack, "records": total, "kept_tokens": tokens,
                   "games": sorted(games), "drops": dict(drops),
                   "best_actions": {"%s/%d" % k: v for k, v in sorted(best.items())}}, f, indent=1)
    print("reselected: %d records, ~%d tokens, %d games, drops %s -> %s" % (total, tokens, len(games), dict(drops), out))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+", help="run output dirs (each holds artifacts/*_events.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", nargs="*", help="one file-name label per run dir (default: basenames or r0, r1, ...)")
    ap.add_argument("--slack", type=float, default=1.25, help="keep levels with actions <= slack x best")
    ap.add_argument("--context-chars", type=int, default=110000, help="history budget (~25K tokens at 4.3 chars/token)")
    ap.add_argument("--upscale", type=int, default=10, help="grid image upscale, as the harness's MULTIMODAL_UPSCALE")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--all", action="store_true", help="write every record with keep/drop fields")
    ap.add_argument("--reselect", action="store_true", help="dirs are --all outputs: merge them and re-apply the rule")
    args = ap.parse_args(argv)
    if args.reselect:
        return reselect(args.dirs, args.out, args.slack)
    runs = find_runs(args.dirs, args.label)
    if not runs:
        print("no *_events.jsonl under", args.dirs)
        return 1
    best, loaded = best_actions_over(runs)
    os.makedirs(args.out, exist_ok=True)
    index = {"runs": [], "games": {}, "records": 0, "kept_tokens": 0, "drops": collections.Counter()}
    images = not args.no_images
    for label, game, pass_no, path, req_path in runs:
        requests = load_requests(req_path) if req_path else {}
        recs, stats, drops = records_for_run(label, game, pass_no, loaded[path], requests, best, args.slack,
                                             args.context_chars, args.upscale, images, args.all)
        name = "%s__%s_p%d.jsonl.gz" % (label, game, pass_no)
        with gzip.open(os.path.join(args.out, name), "wt", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
        kept = [r for r in recs if r.get("keep", True)]
        index["runs"].append({"run": label, "game": game, "pass": pass_no, "events": path,
                              "requests": bool(req_path), "levels": {str(k): v for k, v in sorted(stats.items())},
                              "records": len(kept), "drops": dict(drops), "file": name})
        index["records"] += len(kept)
        index["kept_tokens"] += sum(int(r["approx_tokens"]) + len(json.dumps(r["completion"])) // 4 for r in kept)
        index["drops"].update(drops)
        g = index["games"].setdefault(game, {"levels_completed": 0, "records": 0, "best_actions": {}})
        g["levels_completed"] = max(g["levels_completed"], sum(1 for v in stats.values() if v["completed"]))
        g["records"] += len(kept)
        g["best_actions"] = {str(l): a for (gg, l), a in sorted(best.items()) if gg == game}
        print("%s %s p%d: levels %s kept %d records%s" % (
            label, game, pass_no, "/".join("%d:%s%d" % (l, "+" if v["completed"] else "-", v["actions"]) for l, v in sorted(stats.items())),
            len(kept), (" | drops " + json.dumps(dict(drops))) if drops else ""))
    index["drops"] = dict(index["drops"])
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    print("total: %d records, ~%d tokens incl. completions, %d games -> %s" % (
        index["records"], index["kept_tokens"], len(index["games"]), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
