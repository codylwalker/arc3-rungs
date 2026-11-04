"""Compose a game plan: a seeded choice of mechanics, a palette, a direction map and >= 6 level
grids of growing size and content. The oracle (certify.py) decides whether a plan is solvable;
compose only places things.

Mechanics (CONFIG keys): walls (always), hazard, keydoor, pushable (+plates), teleporter, oneway,
ice, wrap, snake (trail length), switch (two agents, ACTION5), state_flip (n glyphs, crosses),
budget (a step bar; set by certify from the optimal length).
"""
import random
import string

MECHANICS = ["hazard", "keydoor", "pushable", "teleporter", "oneway", "ice", "wrap", "snake", "switch", "state_flip"]
EXCLUDES = {("snake", "pushable"), ("snake", "switch"), ("wrap", "oneway"), ("switch", "state_flip"), ("ice", "snake"),
            ("teleporter", "wrap")}
COLORS = list(range(16))


def _ok(ms):
    return not any((a, b) in EXCLUDES or (b, a) in EXCLUDES for a in ms for b in ms if a != b)


GOAL_MECH = {"keys": "keydoor", "plates": "pushable"}  # a goal kind (runtime_move GOAL_ITEMS) and the mechanic it needs
GOAL_EXCLUDE = ("switch",)  # switch places its own G tiles and a second agent


def pick_mechanics(rng, k=None, must=None, exclude=()):
    k = k or rng.choice([2, 2, 3, 3, 4])
    for _ in range(200):
        ms = list(must or [])
        pool = [m for m in MECHANICS if m not in ms and m not in exclude]
        rng.shuffle(pool)
        ms += pool[: k - len(ms)]
        if _ok(ms):
            return sorted(ms)
    return ["hazard", "keydoor"]


ROLES = ["wall", "band", "letterbox", "goal", "hazard", "key", "door", "block", "plate", "oneway", "ice", "cross",
         "trail", "budget"]
MECH_ROLES = {"hazard": ["hazard"], "keydoor": ["key", "door"], "pushable": ["block", "plate"], "oneway": ["oneway"],
              "ice": ["ice"], "state_flip": ["cross"], "snake": ["trail"]}


def palette(rng):
    """Every role a distinct colour; glyph and teleporter colours drawn from all non-background colours, so
    they can repeat a role's. The click family's palette (it draws no glyphs); movement games use
    move_palette, which keeps the agent and every glyph apart from what the game draws."""
    cols = COLORS[:]
    rng.shuffle(cols)
    bg = cols.pop()
    pal = {"bg": bg}
    for r in ROLES:
        pal[r] = cols.pop() if cols else rng.choice(COLORS)
    pal["glyph"] = [c for c in rng.sample([c for c in COLORS if c != bg], 4)]
    pal["tele"] = [c for c in rng.sample([c for c in COLORS if c != bg], 4)]
    return pal


def move_palette(rng, mechanics):
    """Distinct colours, in priority order, for walls, the goal, the agent's glyph, each role the game's
    mechanics draw, the other glyphs (state_flip), the teleporter pads, then the HUD band, letterbox and budget
    bar. 16 colours cover every 4-mechanic mix but the fullest (state_flip with 4 glyphs plus keydoor, pushable
    and teleporter); there the HUD roles reuse a colour, never the background's, the walls' or a glyph's.
    (10-27: with glyphs drawn from all colours, 5 of 31 games drew the agent in a role's colour: invisible
    on ice of its own colour.)"""
    cols = COLORS[:]
    rng.shuffle(cols)
    bg = cols.pop()
    pal = {"bg": bg}
    keep = {bg}

    def take():
        if cols:
            c = cols.pop()
        else:
            c = rng.choice([c for c in COLORS if c not in keep] or [c for c in COLORS if c != bg])
        return c

    pal["wall"] = take()
    pal["goal"] = take()
    glyph = [take()]
    keep.update([pal["wall"], glyph[0]])
    for m in mechanics:
        for r in MECH_ROLES.get(m, []):
            pal[r] = take()
    glyph += [take() for _ in range(3 if "state_flip" in mechanics else 0)]
    keep.update(glyph)
    tele = [take() for _ in range(2 if "teleporter" in mechanics else 0)]
    for r in ("band", "letterbox", "budget"):  # the HUD glyph and the budget bar sit on the band
        pal[r] = take()
        keep.add(pal[r])
    others = [c for c in COLORS if c != bg]
    pal["glyph"] = glyph + [rng.choice(others) for _ in range(4 - len(glyph))]  # unused slots are never drawn
    pal["tele"] = tele + [rng.choice(others) for _ in range(4 - len(tele))]
    for r in ROLES:
        pal.setdefault(r, rng.choice(others))  # roles this game never draws
    return pal


def direction_map(rng):
    acts = [1, 2, 3, 4]
    rng.shuffle(acts)
    return {str(a): d for a, d in zip(acts, [0, 1, 2, 3])}


def game_id(rng):
    return "g" + "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(3))


def _empty(w, h):
    g = [["#"] * w for _ in range(h)]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            g[y][x] = "."
    return g


def _carve_walls(g, rng, density):
    h, w = len(g), len(g[0])
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if rng.random() < density:
                g[y][x] = "#"
    # a few wall segments
    for _ in range(rng.randint(1, max(1, (w + h) // 6))):
        x, y = rng.randint(1, w - 2), rng.randint(1, h - 2)
        horiz = rng.random() < 0.5
        for i in range(rng.randint(2, max(2, min(w, h) // 2))):
            xx, yy = (x + i, y) if horiz else (x, y + i)
            if 1 <= xx < w - 1 and 1 <= yy < h - 1:
                g[yy][xx] = "#"


def _free_cells(g):
    return [(x, y) for y, row in enumerate(g) for x, ch in enumerate(row) if ch == "."]


def _place(g, rng, ch, n=1, avoid=()):
    cells = [c for c in _free_cells(g) if c not in avoid]
    rng.shuffle(cells)
    out = []
    for c in cells[:n]:
        g[c[1]][c[0]] = ch
        out.append(c)
    return out


def _lane_free(g, c, used):
    x, y = c
    return 0 <= y < len(g) and 0 <= x < len(g[0]) and g[y][x] == "." and c not in used


def _place_push(g, rng, n, used):
    """Up to n block/plate pairs. Each plate lies 1-3 cells from its block along a free lane, and the
    cell behind the block is free for the agent: every pair is solvable on its own, so the search's
    work is the interaction (a block dropped against a wall can never reach its plate). The lane and the
    push cell are reserved, so later placements leave them clear."""
    placed = []
    cells = [c for c in _free_cells(g) if c not in used]
    rng.shuffle(cells)
    for bx, by in cells:
        if len(placed) == n:
            break
        if (bx, by) in used:
            continue
        ds = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        rng.shuffle(ds)
        for dx, dy in ds:
            back = (bx - dx, by - dy)
            lane = [(bx + dx * i, by + dy * i) for i in range(1, rng.randint(1, 3) + 1)]
            if not _lane_free(g, back, used) or not all(_lane_free(g, c, used) for c in lane):
                continue
            g[by][bx] = "B"
            g[lane[-1][1]][lane[-1][0]] = "P"
            used.update([(bx, by), back] + lane)
            placed.append((bx, by))
            break
    return placed


def _far_pair(g, rng):
    cells = _free_cells(g)
    best, bd = None, -1
    for _ in range(40):
        a, b = rng.sample(cells, 2)
        d = abs(a[0] - b[0]) + abs(a[1] - b[1])
        if d > bd:
            best, bd = (a, b), d
    return best


def level_rows(rng, mechanics, idx, n_levels, goal=None):
    """One level grid for level `idx` (0-based). Content grows with idx. A goal kind (`goal`, GOAL_MECH) places no
    G tile: the level completes when its items are done (runtime_move), and a keys goal gets one key more than
    doors, so collecting every key is always possible."""
    t = idx / max(1, n_levels - 1)
    w = int(7 + round(t * 7)) + rng.randint(0, 2)
    h = int(7 + round(t * 5)) + rng.randint(0, 2)
    w, h = min(w, 18), min(h, 14)
    g = _empty(w, h)
    _carve_walls(g, rng, 0.08 + 0.12 * t)
    a, gcell = _far_pair(g, rng)
    g[a[1]][a[0]] = "A"
    if not goal:
        g[gcell[1]][gcell[0]] = "G"
    used = {a, gcell}
    # level 0 is the tutorial: the first mechanic only; later levels add the rest
    active = mechanics[: 1 + min(len(mechanics) - 1, idx)] if idx < len(mechanics) else mechanics
    k = 1 + int(t * 3)
    for m in active:
        if m == "hazard":
            used.update(_place(g, rng, "X", k + 1, used))
        elif m == "keydoor":
            used.update(_place(g, rng, "K", k + (1 if goal == "keys" else 0), used))
            used.update(_place(g, rng, "D", k, used))
        elif m == "pushable":
            _place_push(g, rng, min(k, 2), used)  # 3+ random boxes is Sokoban past the search caps
        elif m == "teleporter":
            for ch in "12"[: 1 + (idx >= 2)]:
                used.update(_place(g, rng, ch, 2, used))
        elif m == "oneway":
            for _ in range(k + 1):
                used.update(_place(g, rng, rng.choice("^>v<"), 1, used))
        elif m == "ice":
            used.update(_place(g, rng, "I", 3 + 3 * k, used))
        elif m == "snake":
            pass
        elif m == "switch":
            s = _place(g, rng, "S", 1, used)
            used.update(s)
            used.update(_place(g, rng, "G", 1, used))
        elif m == "state_flip":
            used.update(_place(g, rng, "C", 1 + (k > 1), used))
        elif m == "wrap":
            # open the border on one axis so wrapping is possible
            if rng.random() < 0.5:
                for y in range(1, h - 1):
                    if rng.random() < 0.4:
                        g[y][0] = "."
                        g[y][w - 1] = "."
            else:
                for x in range(1, w - 1):
                    if rng.random() < 0.4:
                        g[0][x] = "."
                        g[h - 1][x] = "."
    return ["".join(r) for r in g]


def compose_game(seed, k=None, must=None, n_levels=6, goal=None):
    """A movement plan. `goal` (keys | plates) makes a goal-kind game: the goal's mechanic is required and LEADS
    (level 1 is it alone), switch is excluded, no level has a G tile, and cfg["goal"] names the kind;
    cfg["show_goal"] is set at emission (corpus.py --pair emits both variants from one plan)."""
    rng = random.Random(seed)
    if goal:
        gm = GOAL_MECH[goal]
        ms = pick_mechanics(rng, k, list(must or []) + [gm], exclude=GOAL_EXCLUDE)
        mechanics = [gm] + [m for m in ms if m != gm]
    else:
        mechanics = pick_mechanics(rng, k, must)
    cfg = {"palette": move_palette(rng, mechanics), "dir_map": direction_map(rng), "mechanics": mechanics, "seed": seed}
    if goal:
        cfg["goal"] = goal
    for m in mechanics:
        cfg[m] = True
    if "snake" in mechanics:
        cfg["snake"] = rng.choice([3, 4, 6])
    if "state_flip" in mechanics:
        cfg["state_flip"] = rng.choice([2, 3, 4])
        cfg["start_glyph"] = 0
    if "pushable" in mechanics:
        cfg["plates"] = True
    levels = [level_spec(rng, mechanics, cfg, i, n_levels) for i in range(n_levels)]
    return {"game_id": game_id(rng), "config": cfg, "levels": levels}


def level_spec(rng, mechanics, cfg, i, n_levels):
    spec = {"rows": level_rows(rng, mechanics, i, n_levels, goal=cfg.get("goal")), "budget": 0}
    if "state_flip" in mechanics and any("C" in r for r in spec["rows"]):
        spec["need_glyph"] = rng.randrange(1, cfg["state_flip"])
    return spec


# ---- click family ----------------------------------------------------------------------------
def _apply_rule(state, cx, cy, rule):
    h, w = len(state), len(state[0])
    cells = [(cx, cy)]
    if rule == "plus":
        cells += [(cx + dx, cy + dy) for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]]
    elif rule == "rowcol":
        cells = [(x, cy) for x in range(w)] + [(cx, y) for y in range(h) if y != cy]
    for x, y in cells:
        if 0 <= x < w and 0 <= y < h:
            state[y][x] ^= 1


def click_level_spec(rng, cfg, i, n_levels):
    t = i / max(1, n_levels - 1)
    if cfg["kind"] == "toggle":
        # >= 3x3 (512+ states): on 2x2 boards random clicking matched the target within ~60 clicks (T1 smoke 10-27)
        w = 3 + int(round(t * 3)) + rng.randint(0, 1)
        h = 3 + int(round(t * 2)) + rng.randint(0, 1)
        w, h = min(w, 6), min(h, 5)
        target = [[rng.randint(0, 1) for _ in range(w)] for _ in range(h)]
        state = [list(r) for r in target]
        for _ in range(2 + i + rng.randint(0, 1)):
            _apply_rule(state, rng.randrange(w), rng.randrange(h), cfg["rule"])
        if state == target:
            _apply_rule(state, 0, 0, cfg["rule"])
        return {"target": target, "state": state, "budget": 0}
    w = int(6 + round(t * 5)) + rng.randint(0, 1)
    h = int(5 + round(t * 4)) + rng.randint(0, 1)
    g = _empty(w, h)
    _carve_walls(g, rng, 0.06 + 0.1 * t)
    a, goal = _far_pair(g, rng)
    g[a[1]][a[0]] = "O"
    g[goal[1]][goal[0]] = "G"
    used = {a, goal}
    for _ in range(1 + int(t * 3)):
        used.update(_place(g, rng, rng.choice("^>v<"), 1, used))
    return {"rows": ["".join(r) for r in g], "budget": 0}


def compose_click_game(seed, kind=None, n_levels=6):
    rng = random.Random(seed)
    kind = kind or rng.choice(["toggle", "move"])
    pal = palette(rng)
    pal.update({"on": pal["key"], "off": pal["door"], "ball": pal["cross"], "floor": pal["ice"]})
    cfg = {"palette": pal, "kind": kind, "mechanics": ["click_" + kind], "seed": seed}
    if kind == "toggle":
        cfg["rule"] = rng.choice(["plus", "plus", "rowcol", "self"])
        cfg["mechanics"] = ["click_toggle_" + cfg["rule"]]
    levels = [click_level_spec(rng, cfg, i, n_levels) for i in range(n_levels)]
    return {"game_id": game_id(rng), "config": cfg, "levels": levels}
