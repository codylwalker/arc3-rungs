"""Is there an observable "wall" signal? At the moment a level has absorbed T generated tokens, do the
game's own events separate levels that will still be completed from levels that never are?

Usage: python -I tools/wall_signal.py <run output dir> [...]
Each dir is a Kaggle demo output (artifacts/*_events.jsonl + summary.txt). Read-only.

Spend per analysis turn = reasoning + content characters + tool-call code characters from the
transcript's [MODEL RESPONSE META] blocks, converted to tokens per game with the game's exact total from
summary.txt. A turn's spend goes to the level being played when it started. Board states are hashed on
the interior (a 4-cell border cropped, as the harness's guards define it) with each level's
always-changing cells (budget bars, timers: changed on >= 80% of consecutive actions) masked.
Reports, for T = 30K / 45K / 60K tokens, each feature's AUC for "never completed" (0.5 = no signal).
"""
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

BORDER = 4
THRESHOLDS = (30000, 45000, 60000)
META = re.compile(r"\[MODEL RESPONSE META\]\n(.*?)(?=\n\[(?:THINKING|TOOL CALL|TOOL RESULT|MODEL RESPONSE META|ANALYZER STATUS|USER PROMPT)|\Z)", re.S)
NUM = {k: re.compile(r"%s: (\d+)" % k) for k in ("reasoning_chars", "content_chars")}
ARGS = re.compile(r'"arguments": "((?:[^"\\]|\\.)*)"')
PER_GAME = re.compile(r"^\s+(\S+): score=[\d.]+, levels=[\d.]+/\d+, actions=\d+, tokens=(\d+)", re.M)


def turn_chars(transcript):
    total = 0
    for block in META.findall(transcript):
        for rx in NUM.values():
            m = rx.search(block)
            total += int(m.group(1)) if m else 0
        total += sum(len(a) for a in ARGS.findall(block))
    return total


def interior(board):
    rows = board[BORDER:-BORDER] if len(board) > 2 * BORDER else board
    return [r[BORDER:-BORDER] if len(r) > 2 * BORDER else r for r in rows]


def masked_hash(grid, mask):
    flat = [v if (i, j) not in mask else -1 for i, row in enumerate(grid) for j, v in enumerate(row)]
    return hashlib.md5(json.dumps(flat).encode()).hexdigest()


def load(path):
    """[(level, chars, [actions])] turns in file order; an action = (interior board, changed, game_over, completed)."""
    turns, pending = [], []
    level = 1
    with open(path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            t = e.get("type")
            if t == "action":
                pending.append((interior(e.get("board") or []), bool(e.get("board_changed")),
                                bool(e.get("game_over")), bool(e.get("level_completed"))))
            elif t == "analysis":
                turns.append((level, turn_chars(e.get("transcript") or ""), pending))
                level += sum(1 for a in pending if a[3])
                pending = []
    if pending:
        turns.append((level, 0, pending))
    return turns


def level_masks(turns):
    """Per level: cells that change on >= 80% of consecutive actions (timers and budget bars)."""
    boards = defaultdict(list)
    for level, _, actions in turns:
        cur = level
        for board, _, _, completed in actions:
            boards[cur].append(board)
            cur += 1 if completed else 0
    masks = {}
    for lv, seq in boards.items():
        counts, pairs = defaultdict(int), 0
        for a, b in zip(seq, seq[1:]):
            if not a or not b or len(a) != len(b):
                continue
            pairs += 1
            for i, (ra, rb) in enumerate(zip(a, b)):
                for j, (x, y) in enumerate(zip(ra, rb)):
                    if x != y:
                        counts[(i, j)] += 1
        masks[lv] = {c for c, n in counts.items() if pairs >= 5 and n >= 0.8 * pairs}
    return masks


def instance_levels(path, tokens_per_char):
    """{level: {"completed": bool, "snap": {T: features}}} for one game instance."""
    turns = load(path)
    masks = level_masks(turns)
    out = defaultdict(lambda: {"completed": False, "snap": {}, "spend": 0.0})
    st = defaultdict(lambda: {"actions": 0, "states": set(), "noops": 0, "deaths": 0, "turns": 0,
                              "zero_turns": 0, "since_new": 0, "recent": []})
    for level, chars, actions in turns:
        rec, s = out[level], st[level]
        rec["spend"] += chars * tokens_per_char
        s["turns"] += 1
        s["zero_turns"] += 0 if actions else 1
        cur = level
        for board, changed, over, completed in actions:
            sc = st[cur]
            sc["actions"] += 1
            sc["noops"] += 0 if changed else 1
            sc["deaths"] += 1 if over else 0
            h = masked_hash(board, masks.get(cur, set()))
            new = h not in sc["states"]
            sc["states"].add(h)
            sc["since_new"] = 0 if new else sc["since_new"] + 1
            sc["recent"] = (sc["recent"] + [1 if new else 0])[-20:]
            if completed:
                out[cur]["completed"] = True
                cur += 1
        for T in THRESHOLDS:
            if T not in rec["snap"] and rec["spend"] >= T:
                a = max(1, s["actions"])
                rec["snap"][T] = {
                    "actions": s["actions"],
                    "distinct_states": len(s["states"]),
                    "novelty": len(s["states"]) / a,
                    "recent_novelty": (sum(s["recent"]) / len(s["recent"])) if s["recent"] else 0.0,
                    "actions_since_new_state": s["since_new"],
                    "noop_fraction": s["noops"] / a,
                    "game_overs": s["deaths"],
                    "tokens_per_action": T / a,
                    "zero_action_turn_fraction": s["zero_turns"] / max(1, s["turns"]),
                }
    return out


def auc(pos, neg):
    """P(score of a random positive > a random negative); ties count half (rank form)."""
    if not pos or not neg:
        return float("nan")
    allv = sorted((v, k) for k, vs in ((1, pos), (0, neg)) for v in vs)
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            ranks[k] = (i + j + 1) / 2.0
        i = j
    r_pos = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 1)
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def max_auc_null(feats, n_pos, perms=500, seed=7):
    """The 95th percentile of the best |AUC - 0.5| across features when labels are shuffled."""
    import random
    rng = random.Random(seed)
    n = len(feats[0])
    best = []
    for _ in range(perms):
        idx = list(range(n))
        rng.shuffle(idx)
        pos_i, neg_i = set(idx[:n_pos]), idx[n_pos:]
        best.append(max(abs(auc([f[i] for i in pos_i], [f[i] for i in neg_i]) - 0.5) for f in feats))
    best.sort()
    return 0.5 + best[int(0.95 * len(best))]


def main():
    rows = []
    for run in sys.argv[1:]:
        summary = open(os.path.join(run, "summary.txt"), encoding="utf-8").read()
        game_tokens = {g.split("-")[0]: int(t) for g, t in PER_GAME.findall(summary)}
        passes = int((re.search(r"^passes:\s+(\d+)", summary, re.M) or [None, "1"])[1])
        files = sorted(glob.glob(os.path.join(run, "artifacts", "*_events.jsonl")))
        by_game = defaultdict(list)
        for p in files:
            by_game[os.path.basename(p).split("-")[0]].append(p)
        for game, paths in by_game.items():
            chars = sum(c for p in paths for _, c, _ in load(p))
            if not chars or game not in game_tokens:
                continue
            tpc = game_tokens[game] * passes / chars
            for p in paths:
                for lv, rec in instance_levels(p, tpc).items():
                    rows.append((run, game, os.path.basename(p), lv, rec))
    print("level records: %d from %d instances in %d runs" % (
        len(rows), len({(r[0], r[2]) for r in rows}), len({r[0] for r in rows})))
    for T in THRESHOLDS:
        reached = [r for r in rows if T in r[4]["snap"]]
        done = [r[4]["snap"][T] for r in reached if r[4]["completed"]]
        walls = [r[4]["snap"][T] for r in reached if not r[4]["completed"]]
        print("\nT = %dK tokens on the level: %d later completed, %d never completed" % (T // 1000, len(done), len(walls)))
        if not done or not walls:
            continue
        print("  %-26s %6s %12s %12s" % ("feature", "AUC", "median done", "median wall"))
        for k in done[0]:
            a = auc([w[k] for w in walls], [d[k] for d in done])
            med = lambda xs: sorted(xs)[len(xs) // 2]
            print("  %-26s %6.2f %12.2f %12.2f" % (k, a, med([d[k] for d in done]), med([w[k] for w in walls])))
        feats = [[x[k] for x in walls + done] for k in done[0]]
        print("  chance level for the best of %d features (95th pct of shuffled labels): AUC %.2f or %.2f"
              % (len(feats), max_auc_null(feats, len(walls)), 1 - (max_auc_null(feats, len(walls)) - 0.5) - 0.5))


if __name__ == "__main__":
    main()
