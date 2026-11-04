"""S1 - object-centric state for ARC-AGI-3 frames.

A frame (64x64 ints 0-15) -> background colour + objects (4-connected same-colour components).
Two frames -> a list of object deltas (moved / vanished / appeared / recolour / reshaped / bg).
A stream of (action, frame) -> the agent object (the one whose moves follow ACTION1-4).

Pure Python (numpy only for speed on the diff); deterministic.
"""
from collections import Counter, defaultdict

import numpy as np

DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}  # action -> (dx, dy) semantic mapping (up/down/left/right)
MAX_OBJECTS = 400  # frames with more components than this are "textured"; we keep the largest


class Obj:
    __slots__ = ("color", "cells", "x0", "y0", "x1", "y1", "size", "shape", "tid")

    def __init__(self, color, cells):
        self.color = color
        self.cells = cells
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        self.x0, self.x1, self.y0, self.y1 = min(xs), max(xs), min(ys), max(ys)
        self.size = len(cells)
        self.shape = hash(frozenset((x - self.x0, y - self.y0) for x, y in cells))
        self.tid = None

    @property
    def pos(self):
        return (self.x0, self.y0)

    @property
    def w(self):
        return self.x1 - self.x0 + 1

    @property
    def h(self):
        return self.y1 - self.y0 + 1

    def key(self):
        return (self.color, self.shape, self.size)

    def __repr__(self):
        return f"Obj(c{self.color} {self.w}x{self.h}@{self.x0},{self.y0} n{self.size})"


def to_np(frame):
    return np.asarray(frame, dtype=np.int8)


def background(a):
    vals, counts = np.unique(a, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def segment(frame, bg=None):
    """Connected components of non-background cells (4-connectivity, same colour)."""
    a = to_np(frame)
    if bg is None:
        bg = background(a)
    H, W = a.shape
    seen = np.zeros((H, W), dtype=bool)
    objs = []
    for y in range(H):
        row = a[y]
        for x in range(W):
            if seen[y, x] or row[x] == bg:
                continue
            c = int(row[x])
            stack = [(x, y)]
            seen[y, x] = True
            cells = []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < W and 0 <= ny < H and not seen[ny, nx] and a[ny, nx] == c:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            objs.append(Obj(c, cells))
    if len(objs) > MAX_OBJECTS:
        objs.sort(key=lambda o: -o.size)
        objs = objs[:MAX_OBJECTS]
    return bg, objs


def match(prev, cur):
    """Greedy correspondence prev->cur. Returns (pairs, vanished, appeared, ambiguous_count).
    Priority: same key (colour, shape, size) at same position; then same key nearest; then same
    position any colour (recolour) ; then same colour+size nearest (reshape)."""
    pairs = []
    used_c = set()
    used_p = set()
    ambiguous = 0

    def take(pi, ci, kind):
        pairs.append((prev[pi], cur[ci], kind))
        used_p.add(pi)
        used_c.add(ci)

    by_key_c = defaultdict(list)
    for ci, o in enumerate(cur):
        by_key_c[o.key()].append(ci)
    by_pos_c = {o.pos: ci for ci, o in enumerate(cur)}
    # pass 1: identical key + position
    for pi, p in enumerate(prev):
        ci = by_pos_c.get(p.pos)
        if ci is not None and ci not in used_c and cur[ci].key() == p.key():
            take(pi, ci, "same")
    # pass 2: identical key, nearest
    for pi, p in enumerate(prev):
        if pi in used_p:
            continue
        cands = [ci for ci in by_key_c.get(p.key(), []) if ci not in used_c]
        if cands:
            cands.sort(key=lambda ci: abs(cur[ci].x0 - p.x0) + abs(cur[ci].y0 - p.y0))
            if len(cands) > 1:
                d0 = abs(cur[cands[0]].x0 - p.x0) + abs(cur[cands[0]].y0 - p.y0)
                d1 = abs(cur[cands[1]].x0 - p.x0) + abs(cur[cands[1]].y0 - p.y0)
                if d0 == d1:
                    ambiguous += 1
            take(pi, cands[0], "moved")
    # pass 3: same position + shape, different colour
    for pi, p in enumerate(prev):
        if pi in used_p:
            continue
        ci = by_pos_c.get(p.pos)
        if ci is not None and ci not in used_c and cur[ci].shape == p.shape:
            take(pi, ci, "recolour")
    # pass 4: same colour, overlapping bbox (reshape)
    for pi, p in enumerate(prev):
        if pi in used_p:
            continue
        best = None
        for ci, o in enumerate(cur):
            if ci in used_c or o.color != p.color:
                continue
            if o.x0 <= p.x1 and p.x0 <= o.x1 and o.y0 <= p.y1 and p.y0 <= o.y1:
                d = abs(o.size - p.size)
                if best is None or d < best[0]:
                    best = (d, ci)
        if best is not None:
            take(pi, best[1], "reshape")
    # pass 5: same colour and size, nearest within a radius (a sprite that moved AND rotated)
    for pi, p in enumerate(prev):
        if pi in used_p:
            continue
        best = None
        for ci, o in enumerate(cur):
            if ci in used_c or o.color != p.color or o.size != p.size:
                continue
            d = abs(o.x0 - p.x0) + abs(o.y0 - p.y0)
            if d <= 16 and (best is None or d < best[0]):
                best = (d, ci)
        if best is not None:
            take(pi, best[1], "moved")
    vanished = [p for pi, p in enumerate(prev) if pi not in used_p]
    appeared = [o for ci, o in enumerate(cur) if ci not in used_c]
    return pairs, vanished, appeared, ambiguous


def deltas(prev_frame, cur_frame, prev_objs=None, cur_objs=None, bg=None):
    """Object-level deltas between two frames. Each delta: dict(kind, obj, dx, dy, from, to)."""
    if prev_objs is None:
        bg, prev_objs = segment(prev_frame, bg)
    if cur_objs is None:
        _, cur_objs = segment(cur_frame, bg)
    pairs, vanished, appeared, amb = match(prev_objs, cur_objs)
    out = []
    for p, c, kind in pairs:
        if kind == "same":
            continue
        if kind == "moved":
            out.append({"kind": "moved", "obj": p, "to": c, "dx": c.x0 - p.x0, "dy": c.y0 - p.y0})
        elif kind == "recolour":
            out.append({"kind": "recolour", "obj": p, "to": c, "from": p.color, "color": c.color})
        else:
            out.append({"kind": "reshape", "obj": p, "to": c, "dsize": c.size - p.size})
    for v in vanished:
        out.append({"kind": "vanished", "obj": v})
    for a in appeared:
        out.append({"kind": "appeared", "obj": a})
    a1 = to_np(prev_frame)
    a2 = to_np(cur_frame)
    ncells = int((a1 != a2).sum())
    return out, ncells, amb, cur_objs


def summarize(ds):
    """Compress a delta list to a multiset of (kind, colour, dx, dy) for census tables."""
    c = Counter()
    for d in ds:
        if d["kind"] == "moved":
            c[("moved", d["obj"].color, int(np.sign(d["dx"])), int(np.sign(d["dy"])))] += 1
        elif d["kind"] == "recolour":
            c[("recolour", d["from"], d["color"])] += 1
        else:
            c[(d["kind"], d["obj"].color)] += 1
    return c


class Tracker:
    """Persist object identities across a stream; find the agent."""

    def __init__(self):
        self.next_id = 0
        self.objs = None
        self.bg = None
        self.moves = defaultdict(Counter)  # tid -> Counter of (action, consistent)
        self.colors = {}
        self.agent_shape = {}
        self.agent_size = {}
        self.agent_key = None
        self.ambiguous = 0
        self.transitions = 0
        self.class_moves = defaultdict(Counter)  # (colour, size) -> Counter of (action, consistent)
        self.last_agent_pos = None
        self.banned = set()  # tids proven not to be the agent (stuck in every direction)

    def start(self, frame):
        """(Re)start on a frame; keeps the agent's identity across level resets by its key."""
        ag = self.agent()
        agent_key = self.agent_key if ag is None else (self.colors.get(ag[0]), self.agent_shape.get(ag[0]), self.agent_size.get(ag[0]))
        cur = next((o for o in (self.objs or []) if ag and o.tid == ag[0]), None)
        if cur is not None:
            self.last_agent_pos = (cur.x0, cur.y0)
        self.bg, self.objs = segment(frame)
        for o in self.objs:
            o.tid = self._new()
        if agent_key and agent_key[0] is not None and ag is not None:
            cands = [o for o in self.objs if o.color == agent_key[0] and (o.shape == agent_key[1] or o.size == agent_key[2])]
            if cands:
                if self.last_agent_pos is not None:
                    cands.sort(key=lambda o: abs(o.x0 - self.last_agent_pos[0]) + abs(o.y0 - self.last_agent_pos[1]))
                cands[0].tid = ag[0]
            self.agent_key = agent_key
        return self.objs

    def _new(self):
        self.next_id += 1
        return self.next_id

    def step(self, action, frame):
        _, cur = segment(frame, self.bg)
        pairs, vanished, appeared, amb = match(self.objs, cur)
        self.ambiguous += amb
        self.transitions += 1
        for p, c, kind in pairs:
            c.tid = p.tid
            if kind == "moved" and action in DIRS:
                dx, dy = c.x0 - p.x0, c.y0 - p.y0
                if abs(dx) > 32:  # wrapped around the screen edge
                    dx -= 64 * int(np.sign(dx))
                if abs(dy) > 32:
                    dy -= 64 * int(np.sign(dy))
                ex, ey = DIRS[action]
                # axis-consistent: the object moves along the action's axis (the sign may wrap or be mirrored)
                consistent = bool((ex != 0 and dx != 0 and dy == 0) or (ey != 0 and dy != 0 and dx == 0))
                self.moves[p.tid][(action, consistent)] += 1
                self.colors[p.tid] = p.color
                self.agent_shape[p.tid] = p.shape
                self.agent_size[p.tid] = p.size
                self.class_moves[(p.color, p.size)][(action, consistent)] += 1
        for a in appeared:
            a.tid = self._new()
        self.objs = cur
        ag = self.agent()
        if ag:
            o = next((x for x in cur if x.tid == ag[0]), None)
            if o is not None:
                self.last_agent_pos = (o.x0, o.y0)
        return cur

    def agent(self, min_moves=3):
        best = None
        live = {o.tid for o in (self.objs or [])}
        for tid, cnt in self.moves.items():
            if tid in self.banned:
                continue
            cons = sum(v for (a, ok), v in cnt.items() if ok)
            incons = sum(v for (a, ok), v in cnt.items() if not ok)
            if cons >= min_moves and tid in live and (best is None or cons - incons > best[1]):
                best = (tid, cons - incons, cons, incons)
        if best is not None:
            return best  # (tid, score, consistent, inconsistent)
        # fallback: the object CLASS (colour, size) whose moves follow the actions, located nearest the last position
        bestk = None
        for key, cnt in self.class_moves.items():
            cons = sum(v for (a, ok), v in cnt.items() if ok)
            incons = sum(v for (a, ok), v in cnt.items() if not ok)
            if cons >= min_moves and cons > 2 * incons and (bestk is None or cons - incons > bestk[1]):
                bestk = (key, cons - incons, cons, incons)
        if bestk is None:
            return None
        cands = [o for o in (self.objs or []) if (o.color, o.size) == bestk[0] and o.tid not in self.banned]
        if not cands:
            return None
        if self.last_agent_pos is not None:
            cands.sort(key=lambda o: abs(o.x0 - self.last_agent_pos[0]) + abs(o.y0 - self.last_agent_pos[1]))
        o = cands[0]
        self.colors[o.tid] = o.color
        self.agent_shape[o.tid] = o.shape
        self.agent_size[o.tid] = o.size
        return (o.tid, bestk[1], bestk[2], bestk[3])

    def consistency(self):
        return 1.0 - self.ambiguous / max(self.transitions, 1)
