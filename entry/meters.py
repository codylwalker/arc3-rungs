"""meters: two perception lines the harness computes so the model need not spend tokens on them.

1. Meters (the opener, once per analysis turn): a colour whose board-wide cell count moves one way
   on at least 30% of the level's actions, along a thin strip or the edge band, is a budget bar or
   a clock. The line names it, where it is, how it moved and roughly how many actions it has left:
       meter: colour 9 along row 63 (cols 20-63): 44 -> 38 cells, -1.0 per action, about 38 actions left
   A HUD-only state is the rung the harness fails most (a key whose value shows at a door, a bar
   whose meaning is a budget); naming the meter turns a display into a readable variable.
2. Far changes (the per-action echo and trace): objects that appeared, vanished or changed colour
   farther than three cells from what the action moved (or from the click), i.e. consequences the
   model's contact heuristics would not connect to its move:
       far: colour 5 appeared at (10, 40) 6px, 28 cells from the moved object
   Not a far change: a resize under a quarter of the object (the bar ticking), an object under
   MIN_FAR_PIXELS or a resized one under MIN_RESIZE_PIXELS (sprite fragments, a bar's end cell).

Host side only (the harness imports it as inference.utils.meters); standard library. Grids are
tuples of int rows as the harness's Frame.grid. The object diff comes from the harness's own
frame_diff / segmentation, passed in as a callable so this module has no harness import at the top.
Clean-room code for an ARC Prize 2026 entry. Licence: MIT-0 (see LICENSE).
"""
import math

RATE = 0.3        # a meter moves on at least this share of the level's actions
ONE_WAY = 0.9     # ... and in one direction on at least this share of its moves
BAND = 4          # the harness's edge band
FAR = 3           # cells; a change farther than this from the move is a consequence
MAX_METERS = 2
MAX_FAR = 3
RESIZE_SHARE = 0.25  # a resized object counts as a far change only when it changed by this share or more
# (arm M, 2026-10-28: 1,423 of 1,616 far lines were the bar or its background resizing by a cell)
MIN_FAR_PIXELS = 3     # an appeared / vanished / recoloured object under this is a sprite fragment or a bar tick
MIN_RESIZE_PIXELS = 8  # ... and a resized one under this (the D' + M demo, 2026-10-29: 568 of 1,459 far
# entries on the public ten were 1-2 px objects, 206 more were 3-8 px resizes; the generated games' 6x6 cells
# put nearly every real object above both)


def _counts(grid):
    out = {}
    for row in grid:
        for v in row:
            out[v] = out.get(v, 0) + 1
    return out


def level_transitions(history, level):
    """(before, after) grids of the actions taken on `level`, from the harness's history entries
    (.action text, .frame.grid, .frame.level, .result dict): RESETs and automatic entries are skipped."""
    out = []
    for i in range(1, len(history)):
        entry, prev = history[i], history[i - 1]
        action = str(getattr(entry, "action", "") or "").strip().upper()
        if not action or action.startswith("RESET"):
            continue
        result = getattr(entry, "result", None) or {}
        if result.get("automatic"):
            continue
        fb, fa = getattr(prev, "frame", None), getattr(entry, "frame", None)
        if fb is None or fa is None:
            continue
        if getattr(fb, "level", level) != level or getattr(fa, "level", level) != level:
            continue
        if len(fb.grid) != len(fa.grid) or any(len(x) != len(y) for x, y in zip(fb.grid, fa.grid)):
            continue
        out.append((fb.grid, fa.grid))
    return out


def find_meters(transitions):
    """[{colour, direction, changes, actions, rows, cols, start, now, per_action, left}] sorted by
    how often the colour moved; the rule is twin_tools' mask rule (2026-10-28), on int grids."""
    if len(transitions) < 4:
        return []
    h = len(transitions[0][0])
    w = len(transitions[0][0][0]) if h else 0
    changes = {}
    cells = {}
    first = transitions[0][0]
    last = transitions[-1][1]
    for before, after in transitions:
        cb, ca = _counts(before), _counts(after)
        for v in set(cb) | set(ca):
            d = ca.get(v, 0) - cb.get(v, 0)
            if d:
                rec = changes.setdefault(v, [0, 0, 0])
                rec[0] += 1
                rec[1 if d < 0 else 2] += 1
        for r in range(h):
            x, y = before[r], after[r]
            if x != y:
                for c in range(w):
                    if x[c] != y[c]:
                        cells.setdefault(x[c], set()).add((r, c))
                        cells.setdefault(y[c], set()).add((r, c))
    n = len(transitions)
    background = max(_counts(first).items(), key=lambda kv: kv[1])[0]
    out = []
    for v, (moves, down, up) in changes.items():
        if v == background or moves < 2 or moves < RATE * n or max(down, up) < ONE_WAY * moves:
            continue
        ch = cells.get(v, set())
        if not ch:
            continue
        rows = [r for r, _ in ch]
        cols = [c for _, c in ch]
        thin = (max(rows) - min(rows) <= 2) or (max(cols) - min(cols) <= 2)
        in_band = sum(1 for r, c in ch if r < BAND or c < BAND or r >= h - BAND or c >= w - BAND)
        if not (thin or in_band >= 0.9 * len(ch)):
            continue
        start, now = _counts(first).get(v, 0), _counts(last).get(v, 0)
        per_action = (now - start) / float(n)
        left = None
        if per_action < 0 and now > 0:
            left = int(math.ceil(now / -per_action))
        out.append({"colour": v, "direction": "down" if down >= up else "up", "changes": moves, "actions": n,
                    "rows": (min(rows), max(rows)), "cols": (min(cols), max(cols)), "start": start, "now": now,
                    "per_action": per_action, "left": left})
    # a growing colour whose strip overlaps a shrinking one is the bar's background filling in behind
    # it (arm M, 2026-10-28: 347 of 674 meter lines were that complement); the shrinking one is the meter
    shrinking = [m for m in out if m["direction"] == "down"]
    out = [m for m in out if m["direction"] == "down" or not any(_overlaps(m, s) for s in shrinking)]
    out.sort(key=lambda m: -m["changes"])
    return out


def _overlaps(a, b):
    return (a["rows"][0] <= b["rows"][1] and b["rows"][0] <= a["rows"][1]
            and a["cols"][0] <= b["cols"][1] and b["cols"][0] <= a["cols"][1])


def meter_line(m):
    r0, r1 = m["rows"]
    c0, c1 = m["cols"]
    if r0 == r1:
        where = "along row %d (cols %d-%d)" % (r0, c0, c1)
    elif c0 == c1:
        where = "along col %d (rows %d-%d)" % (c0, r0, r1)
    else:
        where = "at rows %d-%d, cols %d-%d" % (r0, r1, c0, c1)
    line = "meter: colour %d %s: %d -> %d cells, %+.1f per action" % (
        m["colour"], where, m["start"], m["now"], m["per_action"])
    if m["left"] is not None:
        line += ", about %d actions left" % m["left"]
    line += " (moved on %d of %d actions this level)" % (m["changes"], m["actions"])
    return line


def meters_lines(history, level):
    """At most MAX_METERS lines for the opener, from the level's logged actions."""
    found = find_meters(level_transitions(history, level))
    return [meter_line(m) for m in found[:MAX_METERS]]


def _point(entry, key):
    p = entry.get(key)
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        try:
            return (int(p[0]), int(p[1]))
        except (TypeError, ValueError):
            return None
    return None


def _radius(entry, key="pixels"):
    px = entry.get(key) or 0
    try:
        return int(math.sqrt(max(0, int(px))))
    except (TypeError, ValueError):
        return 0


def _size(e, *keys):
    """The object's pixel count, the largest of the given keys; 0 when none parses."""
    best = 0
    for key in keys:
        try:
            best = max(best, int(e.get(key) or 0))
        except (TypeError, ValueError):
            pass
    return best


def _resized_enough(e, share=RESIZE_SHARE):
    try:
        before, after = int(e.get("before_pixels") or 0), int(e.get("after_pixels") or 0)
    except (TypeError, ValueError):
        return False
    return abs(after - before) >= share * max(before, after, 1)


def far_changes(diff, origin=None, far=FAR):
    """Appeared, disappeared, recoloured or resized objects farther than `far` cells from every
    point the action touched: the moved objects' from/to top-lefts (their size allowed for), or the
    click `origin` (row, col). Returns [(entry_kind, entry, distance)], farthest first."""
    anchors = []
    for e in diff.get("moved") or []:
        rad = _radius(e)
        for key in ("from", "to"):
            p = _point(e, key)
            if p:
                anchors.append((p, rad))
    for e in diff.get("rotated") or []:
        p = _point(e, "at")
        if p:
            anchors.append((p, _radius(e)))
    if origin is not None:
        anchors.append(((int(origin[0]), int(origin[1])), 0))
    out = []
    for kind in ("appeared", "disappeared", "changed_color", "resized"):
        for e in diff.get(kind) or []:
            p = _point(e, "at") or _point(e, "at_before")
            if p is None:
                continue
            if kind == "resized":
                if not _resized_enough(e) or _size(e, "before_pixels", "after_pixels") < MIN_RESIZE_PIXELS:
                    continue  # the bar or its background ticking by a cell, or a sprite fragment
            elif _size(e, "pixels") < MIN_FAR_PIXELS:
                continue  # a fragment of a sprite, or a bar's end cell
            rad = _radius(e, "pixels" if kind != "resized" else "after_pixels")
            if not anchors:
                dist = None
            else:
                dist = min(max(0, max(abs(p[0] - a[0]), abs(p[1] - a[1])) - rad - arad) for a, arad in anchors)
            if dist is None or dist > far:
                out.append((kind, e, dist))
    out.sort(key=lambda t: -(t[2] if t[2] is not None else 10 ** 6))
    return out


def _entry_text(kind, e):
    p = _point(e, "at") or _point(e, "at_before") or ("?", "?")
    if kind == "changed_color":
        return "colour %s->%s recoloured at %s %spx" % (e.get("before_color"), e.get("after_color"), tuple(p), e.get("pixels"))
    if kind == "resized":
        return "colour %s resized %s->%spx at %s" % (e.get("color"), e.get("before_pixels"), e.get("after_pixels"), tuple(p))
    return "colour %s %s at %s %spx" % (e.get("color"), kind, tuple(p), e.get("pixels"))


def far_line(diff, origin=None, far=FAR, limit=MAX_FAR):
    """One line, or None: the far changes of one action, up to `limit`, farthest first."""
    found = far_changes(diff, origin, far)
    if not found:
        return None
    parts = []
    for kind, e, dist in found[:limit]:
        text = _entry_text(kind, e)
        if dist is not None:
            text += ", %d cells from %s" % (dist, "the click" if origin is not None and not (diff.get("moved") or diff.get("rotated")) else "the moved object")
        parts.append(text)
    line = "far: " + "; ".join(parts)
    if len(found) > limit:
        line += "; +%d more" % (len(found) - limit)
    if not (diff.get("moved") or diff.get("rotated")) and origin is None:
        line += " (nothing moved)"
    return line


def action_note(previous_grid, new_grid, diff_fn, origin=None):
    """The far-change line for one executed action, or None. diff_fn(grid_a, grid_b) -> the
    harness's frame diff dict (compute_frame_diff over segment_layer)."""
    if previous_grid is None or new_grid is None or previous_grid == new_grid:
        return None
    if len(previous_grid) != len(new_grid) or any(len(x) != len(y) for x, y in zip(previous_grid, new_grid)):
        return None
    diff = diff_fn(previous_grid, new_grid)
    if not isinstance(diff, dict) or diff.get("changed_cell_count", 0) in (0, -1):
        return None
    return far_line(diff, origin)


def harness_diff_fn():
    """The harness's object diff, imported lazily (host side only)."""
    from inference.utils.frame_diff import compute_frame_diff
    from inference.utils.grid_utils import ARC_COLOR_CHARS
    from inference.utils.segmentation import segment_layer

    def diff(a, b):
        return compute_frame_diff(a, b, segment_layer(a, ARC_COLOR_CHARS).get("nodes", []),
                                  segment_layer(b, ARC_COLOR_CHARS).get("nodes", []), max_group_match=40)
    return diff
