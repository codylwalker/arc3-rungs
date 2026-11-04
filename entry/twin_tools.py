# twin_tools: zero-token world-model checks and planning for an ARC-AGI-3 REPL harness.
#
# The model writes predict(frame, action) -> the board it expects next, and goal(frame) -> bool.
# These helpers replay predict over every logged transition (twin_check_model), test goal on
# every logged board (twin_check_goal), search the model's own simulator for the shortest
# action list (twin_plan), execute a plan one action at a time, stopping at the first board the
# model did not foresee (twin_run_plan), and pick the one action whose outcome best tells rival
# hypotheses apart (twin_best_probe). None of them spends an environment action.
#
# Standard library only, and no `from __future__` line: the harness pastes this source into a
# sandbox bootstrap that allows no third-party imports. Names are prefixed (twin_, _tw_, Twin)
# so they cannot collide with the bootstrap's own.
#
# Clean-room code for an ARC Prize 2026 entry. Licence: MIT-0 (see LICENSE).

import math
import re
import time
from collections import deque

TWIN_BORDER = 4  # the harness's gameplay_changed test excludes the same four-cell border
TWIN_CHARS = "WwgGcBMPRbSYOrNp"  # colour index -> letter, as the harness renders .ascii
TWIN_SIMPLE = ("UP", "DOWN", "LEFT", "RIGHT", "SPACE")
TWIN_MAX_SECONDS = 12.0  # a snippet has 30 s in all, and plan is usually followed by run_plan in it

_TW_ROW = re.compile(r"row\s*=\s*(-?\d+)")
_TW_COL = re.compile(r"col\s*=\s*(-?\d+)")
_TW_TERMINAL = ("game_over", "level_completed", "run_complete", "done")
# an explicit `ignore` (a cell set, 'border' or a function) given to check_model carries over to a
# later run_plan / best_probe in the same snippet that says ignore="auto" (decision 4, 2026-10-28)
_TW_LAST_IGNORE = [None]


def _tw_carried_ignore(ignore):
    if ignore == "auto" and _TW_LAST_IGNORE[0] is not None:
        return _TW_LAST_IGNORE[0]
    return ignore


# ---------------------------------------------------------------------------------------------
# boards and actions


def _tw_char(value):
    if isinstance(value, str):
        if len(value) != 1:
            raise ValueError("a cell is one colour letter, got %r" % (value,))
        return value
    return TWIN_CHARS[max(0, min(15, int(value)))]


def twin_rows(board):
    """The board as a tuple of row strings, from a frame, ascii text, or a list of rows."""
    if isinstance(board, TwinFrame):
        return board.rows
    if isinstance(board, tuple) and (not board or isinstance(board[0], str)):
        return board
    text = getattr(board, "ascii", None)
    if text is not None:
        board = text
    if isinstance(board, str):
        return tuple(board.strip("\n").split("\n"))
    if isinstance(board, (list, tuple)):
        return tuple(row if isinstance(row, str) else "".join(_tw_char(v) for v in row) for row in board)
    raise TypeError("expected a frame, ascii text or a list of rows, got %s" % type(board).__name__)


def twin_parse_action(action):
    """(NAME, row, col) for a click; (NAME, None, None) for any other action."""
    if isinstance(action, dict):
        name = str(action.get("action", "")).strip().upper()
        if name == "MOUSE" and "row" in action and "col" in action:
            return ("MOUSE", int(action["row"]), int(action["col"]))
        return (name, None, None)
    text = str(action or "").strip()
    name = text.split("(", 1)[0].strip().upper()
    if name == "MOUSE":
        row, col = _TW_ROW.search(text), _TW_COL.search(text)
        if row and col:
            return ("MOUSE", int(row.group(1)), int(col.group(1)))
    return (name, None, None)


def twin_action_text(action):
    """An action as history shows it: 'LEFT' or 'MOUSE(row=4, col=7)'."""
    name, row, col = twin_parse_action(action)
    if name == "MOUSE" and row is not None:
        return "MOUSE(row=%d, col=%d)" % (row, col)
    return name


def twin_action_spec(action):
    """An action as action() takes it: 'LEFT' or {'action': 'MOUSE', 'row': r, 'col': c}."""
    name, row, col = twin_parse_action(action)
    if name == "MOUSE" and row is not None:
        return {"action": "MOUSE", "row": row, "col": col}
    return name


class TwinFrame:
    """A board as predict() and goal() see it, real or simulated.

    .rows (tuple of row strings), .ascii, .shape, .level, .step, .grid (colour indices) and
    .segmentation (the harness's object view, computed on first use).
    """

    def __init__(self, rows, level=0, step=0, source=None):
        self.rows = twin_rows(rows)
        self.level = level
        self.step = step
        self.shape = (len(self.rows), len(self.rows[0]) if self.rows else 0)
        self._source = source
        self._grid_cache = None
        self._seg_cache = None

    @property
    def ascii(self):
        return "\n".join(self.rows)

    @property
    def grid(self):
        if self._grid_cache is None:
            index = {ch: i for i, ch in enumerate(TWIN_CHARS)}
            self._grid_cache = [[index.get(ch, 0) for ch in row] for row in self.rows]
        return self._grid_cache

    _grid = grid  # the harness's frame views keep their colour indices here; frame_diff reads it

    @property
    def segmentation(self):
        if self._seg_cache is None:
            if self._source is not None and hasattr(self._source, "segmentation"):
                self._seg_cache = self._source.segmentation
            else:
                segment = globals().get("segment_layer")
                if segment is None:
                    raise RuntimeError("segmentation is not available outside the harness")
                self._seg_cache = segment(self.grid, globals().get("COLOR_CHARS") or TWIN_CHARS)
        return self._seg_cache

    def cell(self, row, col):
        return self.rows[row][col]

    def __repr__(self):
        return "TwinFrame(level=%s, step=%s, shape=%dx%d)" % (self.level, self.step, self.shape[0], self.shape[1])


def _tw_frame(frame):
    if isinstance(frame, TwinFrame):
        return frame
    return TwinFrame(twin_rows(frame), getattr(frame, "level", 0), getattr(frame, "step", 0), source=frame)


def _tw_apply(before, pred):
    """The predicted board: a full board, or a {(row, col): colour} dict of changed cells."""
    if pred is None:
        return None
    if isinstance(pred, dict):
        if not pred:
            return before
        rows = list(before)
        by_row = {}
        for key, value in pred.items():
            r, c = int(key[0]), int(key[1])
            by_row.setdefault(r, []).append((c, _tw_char(value)))
        for r, items in by_row.items():
            if not 0 <= r < len(rows):
                raise IndexError("predicted row %d is off the board" % r)
            line = list(rows[r])
            for c, ch in items:
                if not 0 <= c < len(line):
                    raise IndexError("predicted col %d is off the board" % c)
                line[c] = ch
            rows[r] = "".join(line)
        return tuple(rows)
    return twin_rows(pred)


def _tw_level_of(pred, default):
    return pred.level if isinstance(pred, TwinFrame) else default


# ---------------------------------------------------------------------------------------------
# ignored cells (timers, HUD bars)


def _tw_mask(ignore, shape, transitions=None, level=None):
    """A frozenset of ignored (row, col) cells, or None for a strict comparison."""
    if ignore is None or ignore is False:
        return None
    rows, cols = shape
    if ignore == "border":
        b = TWIN_BORDER
        return frozenset((r, c) for r in range(rows) for c in range(cols)
                         if r < b or c < b or r >= rows - b or c >= cols - b)
    if ignore == "auto":
        return _tw_auto_mask(transitions or [], level, shape)
    if callable(ignore):
        return frozenset((r, c) for r in range(rows) for c in range(cols) if ignore(r, c))
    return frozenset((int(p[0]), int(p[1])) for p in ignore)


def _tw_usable(t, level):
    b, a = t.before_frame, t.after_frame
    if b is None or a is None:
        return False
    res = t.result or {}
    if twin_parse_action(t.action)[0] == "RESET" or res.get("automatic"):
        return False
    if any(res.get(flag) for flag in _TW_TERMINAL):
        return False
    if level is not None and level != "all":
        if getattr(b, "level", None) != level or getattr(a, "level", None) != level:
            return False
    return True


def _tw_band(r, c, h, w):
    b = TWIN_BORDER
    if r < b:
        return "top"
    if r >= h - b:
        return "bottom"
    if c < b:
        return "left"
    if c >= w - b:
        return "right"
    return None


def _tw_band_cells(h, w):
    return [(r, c, _tw_band(r, c, h, w)) for r in range(h) for c in range(w) if _tw_band(r, c, h, w)]


def _tw_band_census(rows, band_cells):
    """{(band, colour): cell count} over the four edge bands of one board."""
    counts = {}
    for r, c, band in band_cells:
        key = (band, rows[r][c])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _tw_census(rows):
    """{colour: cell count} over one board."""
    counts = {}
    for line in rows:
        for ch in line:
            counts[ch] = counts.get(ch, 0) + 1
    return counts


def _tw_extend_strip(cells, colour, frames, horizontal):
    """The whole bar, not just the cells seen changing: along the strip's axis, the runs of the bar's
    colour in the given frames (the first frame holds all of a shrinking bar, the last all of a
    growing one) that touch a changed cell of the same row (or column)."""
    out = set()
    if horizontal:
        for r in sorted(set(r for r, _ in cells)):
            changed = set(c for rr, c in cells if rr == r)
            for frame in frames:
                line = frame[r]
                c = 0
                while c < len(line):
                    if line[c] != colour:
                        c += 1
                        continue
                    start = c
                    while c < len(line) and line[c] == colour:
                        c += 1
                    if any(x in changed or x - 1 in changed or x + 1 in changed for x in range(start, c)):
                        out.update((r, x) for x in range(start, c))
    else:
        for c in sorted(set(c for _, c in cells)):
            changed = set(r for r, cc in cells if cc == c)
            for frame in frames:
                column = [line[c] for line in frame]
                r = 0
                while r < len(column):
                    if column[r] != colour:
                        r += 1
                        continue
                    start = r
                    while r < len(column) and column[r] == colour:
                        r += 1
                    if any(x in changed or x - 1 in changed or x + 1 in changed for x in range(start, r)):
                        out.update((x, c) for x in range(start, r))
    return out


def _tw_auto_mask(transitions, level, shape):
    """Timer cells, judged over the logged actions of a level (of the game, if it has fewer than 4).

    A budget bar or clock is a colour whose board-wide cell count moves one way (down or up on at
    least 90% of its changes) on at least 30% of the actions, and whose changing cells form a thin
    strip (three rows or three columns at most) or lie in the four-cell edge band: a moving sprite
    keeps its count, a shrinking bar does not. The 2026-10-28 measurement of the 25 public games
    put 23 bars in the band and one (sb26) at row 53, and wa30's moves on 37% of the actions, so
    the bar is judged anywhere on the board and at a rate below one half. Masked: the cells seen
    changing, and the whole run of the bar's colour along the strip (its unchanged remainder would
    change next); for a colour drawn around the frame, every band cell that holds it. The board's
    majority colour (the background a bar gives way to) is never extended. Any single cell that
    changes on at least 80% of the actions (a blinker) is masked too; a sprite pacing back and forth
    crosses one cell on half of them, so the per-cell bar is set higher.
    """
    pool = [t for t in transitions if _tw_usable(t, level)]
    if len(pool) < 4:
        pool = [t for t in transitions if _tw_usable(t, "all")]
    if len(pool) < 4:
        return None
    pool = pool[-200:]
    h, w = shape
    index = _tw_band_cells(h, w)
    count_changes = {}
    band_changes = {}
    band_cells = {}
    band_background = {}
    cell_changes = {}
    colour_cells = {}
    first = last = None
    used = 0
    for t in pool:
        before, after = twin_rows(t.before_frame), twin_rows(t.after_frame)
        if len(before) != h or len(after) != h or any(len(x) != w for x in before + after):
            continue
        used += 1
        if first is None:
            first = before
        last = after
        census_b, census_a = _tw_census(before), _tw_census(after)
        for ch in set(census_b) | set(census_a):
            d = census_a.get(ch, 0) - census_b.get(ch, 0)
            if d:
                rec = count_changes.setdefault(ch, [0, 0, 0])
                rec[0] += 1
                rec[1 if d < 0 else 2] += 1
        # the edge band on its own: a bar's colour may also move about in sprites inside the
        # board (ft09, lp85, su15), which hides its one-way count from the board-wide census
        band_b, band_a = _tw_band_census(before, index), _tw_band_census(after, index)
        if not band_background:
            for (band, ch), n in band_b.items():
                if n > band_background.get(band, (None, -1))[1]:
                    band_background[band] = (ch, n)
        for key in set(band_b) | set(band_a):
            if band_b.get(key, 0) != band_a.get(key, 0):
                band_changes[key] = band_changes.get(key, 0) + 1
        for rows in (before, after):
            for r, c, band in index:
                band_cells.setdefault((band, rows[r][c]), set()).add((r, c))
        for r, (x, y) in enumerate(zip(before, after)):
            if x != y:
                for c in range(w):
                    if x[c] != y[c]:
                        cell_changes[(r, c)] = cell_changes.get((r, c), 0) + 1
                        colour_cells.setdefault(x[c], set()).add((r, c))
                        colour_cells.setdefault(y[c], set()).add((r, c))
    if used < 4:
        return None
    masked = set(cell for cell, n in cell_changes.items() if n * 5 >= used * 4)
    for (band, ch), n in band_changes.items():
        if n * 10 >= used * 3 and ch != band_background.get(band, (None, 0))[0]:
            masked |= band_cells.get((band, ch), set())
    background = max(_tw_census(first).items(), key=lambda kv: kv[1])[0]
    for ch, (n, down, up) in count_changes.items():
        if n < 2 or n * 10 < used * 3 or max(down, up) * 10 < n * 9:
            continue
        cells = colour_cells.get(ch, set())
        if not cells:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        thin_h = max(rows) - min(rows) <= 2
        thin_v = max(cols) - min(cols) <= 2
        in_band = sum(1 for r, c in cells if _tw_band(r, c, h, w))
        if not (thin_h or thin_v or in_band * 10 >= len(cells) * 9):
            continue
        masked |= cells
        if ch == background:
            continue
        if thin_h or thin_v:
            horizontal = thin_h and (not thin_v or max(cols) - min(cols) >= max(rows) - min(rows))
            masked |= _tw_extend_strip(cells, ch, (first, last), horizontal)
        else:
            masked |= set((r, c) for frame in (first, last) for r, c, _ in _tw_band_cells(h, w) if frame[r][c] == ch)
    return frozenset(masked) or None


def _tw_keyer(mask):
    """A function from board rows to a hashable key that ignores the masked cells."""
    if not mask:
        return lambda rows: rows
    runs = {}
    for r, c in sorted(mask):
        cols = runs.setdefault(r, [])
        if cols and cols[-1][1] == c:
            cols[-1][1] = c + 1
        else:
            cols.append([c, c + 1])

    def key(rows):
        out = list(rows)
        for r, spans in runs.items():
            if r >= len(out):
                continue
            line = out[r]
            parts = []
            at = 0
            for start, end in spans:
                parts.append(line[at:start])
                parts.append("." * max(0, min(end, len(line)) - start))
                at = end
            parts.append(line[at:])
            out[r] = "".join(parts)
        return tuple(out)

    return key


def _tw_diff(pred, real, mask=None, limit=8):
    """None if the boards agree off the mask; else the count, bounding box and first cells."""
    if len(pred) != len(real) or any(len(x) != len(y) for x, y in zip(pred, real)):
        return {"n_cells": -1, "note": "predicted board has shape %dx%d, actual %dx%d" % (
            len(pred), len(pred[0]) if pred else 0, len(real), len(real[0]) if real else 0)}
    n = 0
    cells = []
    r0 = c0 = 1 << 30
    r1 = c1 = -1
    for r, (x, y) in enumerate(zip(pred, real)):
        if x == y:
            continue
        for c in range(len(x)):
            if x[c] != y[c] and not (mask and (r, c) in mask):
                n += 1
                r0, r1, c0, c1 = min(r0, r), max(r1, r), min(c0, c), max(c1, c)
                if len(cells) < limit:
                    cells.append([r, c, x[c], y[c]])
    if not n:
        return None
    return {"n_cells": n, "bbox": [r0, c0, r1, c1], "cells": cells}


def _tw_error(exc):
    return "%s: %s" % (type(exc).__name__, str(exc)[:200])


# ---------------------------------------------------------------------------------------------
# the four tools


def twin_check_model(predict, transitions, level=None, ignore="auto", include_terminal=False, limit=8):
    """Replay predict(frame, action) over the logged transitions of a level.

    level: None = the latest board's level; "all" = every level. ignore: "auto" (cells that
    change on at least half the actions, i.e. timers), "border", None (strict), a set of
    (row, col) cells, or a function (row, col) -> bool.
    """
    transitions = list(transitions or [])
    if level is None and transitions:
        level = getattr(transitions[-1].after_frame, "level", None)
    if ignore is not None and not isinstance(ignore, str):
        _TW_LAST_IGNORE[0] = ignore
    elif ignore == "border":
        _TW_LAST_IGNORE[0] = ignore
    checked = matched = abstained = errors = 0
    mismatch_at = []
    first = None
    first_error = None
    masks = {}
    for i, t in enumerate(transitions):
        res = t.result or {}
        terminal = any(res.get(flag) for flag in _TW_TERMINAL)
        if terminal and include_terminal:
            if t.before_frame is None or t.after_frame is None or res.get("automatic"):
                continue
            if level not in (None, "all") and getattr(t.before_frame, "level", None) != level:
                continue
        elif not _tw_usable(t, level):
            continue
        before = _tw_frame(t.before_frame)
        text = twin_action_text(t.action)
        try:
            got = _tw_apply(before.rows, predict(before, text))
        except Exception as exc:  # the model's code: report it, keep replaying
            errors += 1
            if first_error is None:
                first_error = {"transition": i, "action": text, "error": _tw_error(exc)}
            continue
        if got is None:
            abstained += 1
            continue
        checked += 1
        real = twin_rows(t.after_frame)
        shape = (len(real), len(real[0]) if real else 0)
        if shape not in masks:
            masks[shape] = _tw_mask(ignore, shape, transitions, level)
        diff = _tw_diff(got, real, masks[shape], limit)
        if diff is None:
            matched += 1
            continue
        mismatch_at.append(i)
        if first is None:
            first = dict(diff, transition=i, action=text)
    ignored = sum(len(m) for m in masks.values() if m) if masks else 0
    out = {
        "level": level,
        "checked": checked,
        "matched": matched,
        "mismatched": len(mismatch_at),
        "abstained": abstained,
        "first_mismatch": first,
        "mismatch_at": mismatch_at[:12],
        "ignored_cells": ignored,
    }
    if errors:
        out["errors"] = errors
        out["first_error"] = first_error
    if not checked and not errors:
        summary = "no transitions checked (none logged on this level, or predict returned None for all)"
    else:
        summary = "predict reproduces %d of %d checked transitions" % (matched, checked)
        if abstained:
            summary += " (%d abstained)" % abstained
        if first:
            if first.get("n_cells", 0) > 0:
                b = first["bbox"]
                summary += "; first mismatch at transitions[%d] %s: %d cells in rows %d-%d, cols %d-%d" % (
                    first["transition"], first["action"], first["n_cells"], b[0], b[2], b[1], b[3])
            else:
                summary += "; first mismatch at transitions[%d] %s: %s" % (
                    first["transition"], first["action"], first.get("note", "shape differs"))
        if errors:
            summary += "; predict raised on %d (first: %s)" % (errors, first_error["error"])
    out["summary"] = summary
    return out


def twin_check_goal(goal, history, level=None):
    """goal(frame) on every logged board of a level; a correct win test is False on all of them."""
    history = list(history or [])
    if level is None and history:
        level = getattr(history[-1].frame, "level", None)
    checked = 0
    true_on = []
    errors = 0
    first_error = None
    for i, entry in enumerate(history):
        frame = getattr(entry, "frame", None)
        if frame is None:
            continue
        if level not in (None, "all") and getattr(frame, "level", None) != level:
            continue
        checked += 1
        try:
            hit = bool(goal(_tw_frame(frame)))
        except Exception as exc:
            errors += 1
            if first_error is None:
                first_error = {"history_index": i, "error": _tw_error(exc)}
            continue
        if hit:
            true_on.append(i)
    out = {"level": level, "checked": checked, "n_true": len(true_on), "true_on": true_on[:12],
           "ok": not true_on and not errors}
    if errors:
        out["errors"] = errors
        out["first_error"] = first_error
    if true_on:
        out["summary"] = ("goal is True on %d of %d logged boards of this level (first history[%d]); "
                          "the level did not end there, so the test is too loose unless winning needs "
                          "one more action" % (len(true_on), checked, true_on[0]))
    else:
        out["summary"] = "goal is False on all %d logged boards of this level" % checked
    if errors:
        out["summary"] += "; goal raised on %d (first: %s)" % (errors, first_error["error"])
    return out


def _tw_click_targets(frame, mask=None, background=None):
    """One cell per same-colour 4-connected object (not the most common colour)."""
    rows = frame.rows
    h, w = frame.shape
    if background is None:
        counts = {}
        for row in rows:
            for ch in row:
                counts[ch] = counts.get(ch, 0) + 1
        background = max(counts, key=counts.get) if counts else None
    seen = set()
    targets = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or rows[r][c] == background or (mask and (r, c) in mask):
                continue
            colour = rows[r][c]
            stack = [(r, c)]
            seen.add((r, c))
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and rows[ny][nx] == colour:
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            cy = sum(p[0] for p in cells) / len(cells)
            cx = sum(p[1] for p in cells) / len(cells)
            targets.append(min(cells, key=lambda p: (p[0] - cy) ** 2 + (p[1] - cx) ** 2))
    return targets


def twin_plan(predict, goal, start, moves=None, clicks=None, max_nodes=20000, time_limit=8.0,
              ignore="auto", max_depth=None, valid_actions=None, transitions=None):
    """Breadth-first search inside predict for the shortest action list that makes goal True.

    moves: action names to try (default: UP/DOWN/LEFT/RIGHT/SPACE among valid_actions).
    clicks: a list of (row, col) cells, or "objects" for one click per object on each board.
    A predict that returns None for an action prunes it; an action that leaves the board
    unchanged is never part of a plan.
    """
    t0 = time.perf_counter()
    time_limit = min(float(time_limit), TWIN_MAX_SECONDS)
    root = _tw_frame(start)
    if moves is None:
        offered = [str(v).strip().upper() for v in (valid_actions or TWIN_SIMPLE)]
        moves = [m for m in TWIN_SIMPLE if m in offered]
    base = [twin_action_text(m) for m in moves]
    per_board_clicks = clicks == "objects"
    if clicks and not per_board_clicks:
        base += ["MOUSE(row=%d, col=%d)" % (int(r), int(c)) for r, c in clicks]
    mask = _tw_mask(ignore, root.shape, transitions, root.level)
    keyer = _tw_keyer(mask)
    errors = 0
    first_error = None

    def done(status, actions=None, nodes=0, visited=0):
        out = {"status": status, "actions": actions, "length": None if actions is None else len(actions),
               "nodes": nodes, "visited": visited, "seconds": round(time.perf_counter() - t0, 2)}
        if errors:
            out["errors"] = errors
            out["first_error"] = first_error
        return out

    try:
        if goal(root):
            return done("found", [], 0, 1)
    except Exception as exc:
        return dict(done("goal_error"), error=_tw_error(exc))
    root_key = keyer(root.rows)
    parent = {root_key: None}
    queue = deque([(root, root_key, 0)])
    nodes = 0
    status = "exhausted"
    while queue:
        frame, key, depth = queue.popleft()
        if max_depth is not None and depth >= max_depth:
            continue
        actions = base + ["MOUSE(row=%d, col=%d)" % p for p in _tw_click_targets(frame, mask)] \
            if per_board_clicks else base
        for text in actions:
            if nodes >= max_nodes:
                status = "node_limit"
                break
            if time.perf_counter() - t0 > time_limit:
                status = "time_limit"
                break
            nodes += 1
            try:
                pred = predict(frame, text)
                rows = _tw_apply(frame.rows, pred)
            except Exception as exc:
                errors += 1
                if first_error is None:
                    first_error = {"action": text, "depth": depth, "error": _tw_error(exc)}
                if errors >= 25 and errors == nodes:
                    return done("predict_error", None, nodes, len(parent))
                continue
            if rows is None:
                continue
            child_key = keyer(rows)
            if child_key in parent:
                continue
            parent[child_key] = (key, text)
            child = TwinFrame(rows, _tw_level_of(pred, frame.level), frame.step + 1)
            try:
                hit = goal(child)
            except Exception as exc:
                errors += 1
                if first_error is None:
                    first_error = {"action": text, "depth": depth + 1, "error": "goal: " + _tw_error(exc)}
                hit = False
            if hit:
                path = []
                at = child_key
                while parent[at] is not None:
                    at, step_text = parent[at]
                    path.append(twin_action_spec(step_text))
                path.reverse()
                return done("found", path, nodes, len(parent))
            queue.append((child, child_key, depth + 1))
        if status != "exhausted":
            break
    return done(status, None, nodes, len(parent))


def twin_run_plan(actions, predict, act, get_frame, ignore="auto", transitions=None, limit=8):
    """Execute actions one at a time through act(); stop at the first board predict did not foresee.

    act is the harness's action(); get_frame returns the current board after each call.
    Stops early on game over or level completion, which act reports in its result.
    """
    run = []
    mask = None
    ignore = _tw_carried_ignore(ignore)
    for spec in list(actions or []):
        text = twin_action_text(spec)
        before = get_frame()
        expect = None
        if predict is not None:
            try:
                expect = _tw_apply(twin_rows(before), predict(_tw_frame(before), text))
            except Exception as exc:
                return {"executed": len(run), "stopped": "predict_error", "actions_run": run,
                        "error": _tw_error(exc), "next_action": text}
        result = act([twin_action_spec(spec)])
        run.append(text)
        res = result if isinstance(result, dict) else {}
        for flag in _TW_TERMINAL:
            if res.get(flag):
                return {"executed": len(run), "stopped": "run_complete" if flag == "done" else flag,
                        "actions_run": run}
        if expect is not None:
            after = twin_rows(get_frame())
            if mask is None:
                mask = _tw_mask(ignore, (len(after), len(after[0]) if after else 0), transitions,
                                getattr(before, "level", None)) or frozenset()
            diff = _tw_diff(expect, after, mask, limit)
            if diff is not None:
                diff.update(at=len(run) - 1, action=text)
                return {"executed": len(run), "stopped": "mismatch", "actions_run": run, "mismatch": diff,
                        "remaining": [twin_action_text(a) for a in list(actions)[len(run):]][:20]}
    return {"executed": len(run), "stopped": "done", "actions_run": run}


def twin_best_probe(predicts, start, moves=None, clicks=None, ignore="auto", weights=None,
                    valid_actions=None, transitions=None, limit=5):
    """The one action from `start` whose outcome best tells rival hypotheses apart.

    predicts: a list of predict(frame, action) functions, or a dict name -> function. For each
    candidate action the hypotheses are grouped by the board they predict, and the action whose
    grouping has the most entropy (bits of expected information, `weights` as the prior, uniform
    by default) wins: OPINE-World's and experimental design's rule, on the CPU. A hypothesis
    that returns None or raises for an action sits that action out.
    """
    if isinstance(predicts, dict):
        names = [str(k) for k in predicts]
        funcs = list(predicts.values())
    else:
        funcs = list(predicts or [])
        names = [getattr(f, "__name__", "h%d" % i) for i, f in enumerate(funcs)]
        if len(set(names)) < len(names):
            names = ["h%d" % i for i in range(len(funcs))]
    if len(funcs) < 2:
        return {"action": None, "bits": 0.0, "split": None, "ranking": [],
                "summary": "best_probe needs at least two rival predict functions"}
    prior = [float(x) for x in (weights or [1.0] * len(funcs))]
    if len(prior) != len(funcs):
        raise ValueError("weights has %d entries for %d hypotheses" % (len(prior), len(funcs)))
    frame = _tw_frame(start)
    if moves is None:
        offered = [str(v).strip().upper() for v in (valid_actions or TWIN_SIMPLE)]
        moves = [m for m in TWIN_SIMPLE if m in offered]
    actions = [twin_action_text(m) for m in moves]
    mask = _tw_mask(_tw_carried_ignore(ignore), frame.shape, transitions, frame.level)
    if clicks == "objects":
        actions += ["MOUSE(row=%d, col=%d)" % p for p in _tw_click_targets(frame, mask)]
    elif clicks:
        actions += ["MOUSE(row=%d, col=%d)" % (int(r), int(c)) for r, c in clicks]
    keyer = _tw_keyer(mask)
    errors = {}
    ranking = []
    for order, text in enumerate(actions):
        groups = {}
        for i, f in enumerate(funcs):
            try:
                rows = _tw_apply(frame.rows, f(frame, text))
            except Exception as exc:
                errors.setdefault(names[i], "%s on %s" % (_tw_error(exc), text))
                continue
            if rows is not None:
                groups.setdefault(keyer(rows), []).append(i)
        mass = [sum(prior[i] for i in g) for g in groups.values()]
        total = sum(mass)
        bits = 0.0
        if total > 0 and len(groups) > 1:
            bits = -sum(m / total * math.log2(m / total) for m in mass if m > 0)
        ranking.append((bits, len(groups), order, text, list(groups.values())))
    ranking.sort(key=lambda item: (-item[0], -item[1], item[2]))
    best = ranking[0] if ranking and ranking[0][0] > 1e-9 else None
    out = {
        "action": twin_action_spec(best[3]) if best else None,
        "bits": round(best[0], 3) if best else 0.0,
        "split": [[names[i] for i in g] for g in best[4]] if best else None,
        "ranking": [{"action": t, "bits": round(b, 3), "outcomes": n} for b, n, _, t, _ in ranking[:limit]],
    }
    if errors:
        out["errors"] = errors
    if best:
        out["summary"] = "%s splits the %d hypotheses into %s (%.2f bits)" % (
            best[3], len(funcs), " | ".join(",".join(part) for part in out["split"]), best[0])
    else:
        out["summary"] = ("no single action from here tells these hypotheses apart: they predict the same "
                          "board for every candidate action")
    return out


# ---------------------------------------------------------------------------------------------
# certification unasked: a predict the model defined, replayed over the level's log after its snippet

_TW_CERTIFY_NAMES = re.compile(r"(predict|step|simulate|transition|next_frame|world|model|dynamics)", re.I)
_TW_ARG_FRAME = re.compile(r"(frame|grid|board|state|obs)", re.I)
_TW_ARG_ACTION = re.compile(r"(action|move|act|cmd)", re.I)


def _tw_certify_candidate(name, func):
    """Does this look like a world model? By its name, or by two parameters named like a board and an action."""
    code = getattr(func, "__code__", None)
    if code is None or not callable(func):
        return False
    params = list(code.co_varnames[:code.co_argcount])
    if len(params) < 2 or len(params) > 3:
        return False
    if _TW_CERTIFY_NAMES.search(name):
        return True
    return bool(_TW_ARG_FRAME.search(params[0]) and _TW_ARG_ACTION.search(params[1]))


class _TwBudget(Exception):
    pass


def _tw_adapted(func, deadline):
    """predict(frame, text) over the model's own signature: (TwinFrame, text) first, then the grid
    of ints, then the row strings; the first form that does not raise on a call is kept."""
    forms = [lambda f, a: func(f, a), lambda f, a: func(f.grid, a), lambda f, a: func(f.rows, a)]
    chosen = []

    def predict(frame, text):
        if time.monotonic() > deadline:
            raise _TwBudget("budget")
        if chosen:
            return chosen[0](frame, text)
        last = None
        for form in forms:
            try:
                out = form(frame, text)
            except (TypeError, AttributeError, IndexError, KeyError, ValueError) as exc:
                last = exc
                continue
            chosen.append(form)
            return out
        raise last if last is not None else TypeError("no usable signature")
    return predict


def twin_auto_certify(names, namespace, transitions, level=None, budget=2.0, recent=60, ignore="auto"):
    """One line per world-model-like function the snippet defined, from its replay over the last
    `recent` logged transitions of the level; None when nothing qualifies or too little is logged
    (fewer than 4 transitions checked). The model is told without having to ask: on the generated
    games it wrote its own search in 40 of 41 games and called check_model in 8.
    """
    transitions = list(transitions or [])
    if level is None and transitions:
        level = getattr(transitions[-1].after_frame, "level", None)
    pool = [t for t in transitions if _tw_usable(t, level)]
    if len(pool) < 4:
        return None
    pool = pool[-recent:]
    deadline = time.monotonic() + budget
    lines = []
    for name in names:
        func = namespace.get(name)
        if not _tw_certify_candidate(name, func):
            continue
        try:
            out = twin_check_model(_tw_adapted(func, deadline), pool, level=level, ignore=ignore)
        except Exception:
            continue
        if out.get("errors") and out["checked"] == 0:
            continue
        if out.get("first_error") and "budget" in str(out["first_error"].get("error", "")):
            continue
        if out["checked"] < 4:
            continue
        if out["matched"] == out["checked"]:
            line = "[certify] %s reproduces all %d logged transitions of this level" % (name, out["checked"])
        else:
            line = "[certify] %s reproduces %d of %d logged transitions of this level" % (name, out["matched"], out["checked"])
            first = out.get("first_mismatch")
            if first:
                if first.get("n_cells", 0) > 0:
                    b = first["bbox"]
                    line += "; first mismatch at transitions[%d] %s: %d cells in rows %d-%d, cols %d-%d" % (
                        first["transition"], first["action"], first["n_cells"], b[0], b[2], b[1], b[3])
                else:
                    line += "; first mismatch at transitions[%d] %s: %s" % (
                        first["transition"], first["action"], first.get("note", "shape differs"))
        if out.get("ignored_cells"):
            line += " (%d timer or bar cells ignored)" % out["ignored_cells"]
        if out.get("abstained"):
            line += " (%d abstained)" % out["abstained"]
        lines.append(line)
        if time.monotonic() > deadline:
            break
    return "\n".join(lines) if lines else None
