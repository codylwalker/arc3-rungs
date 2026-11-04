# Generated ARC-AGI-3-style game (house-side, uncertified unless cert.json says otherwise).
# MIT-0 for the emitted file; mechanics are ideas, not code, from the public set.
CONFIG = {'dir_map': {'1': 3, '2': 1, '3': 2, '4': 0},
 'goal': 'keys',
 'keydoor': True,
 'mechanics': ['keydoor', 'teleporter'],
 'palette': {'band': 3,
             'bg': 1,
             'block': 0,
             'budget': 0,
             'cross': 0,
             'door': 9,
             'glyph': [14, 2, 14, 9],
             'goal': 12,
             'hazard': 2,
             'ice': 9,
             'key': 2,
             'letterbox': 10,
             'oneway': 5,
             'plate': 14,
             'tele': [5, 15, 6, 7],
             'trail': 4,
             'wall': 11},
 'seed': 1004,
 'show_goal': False,
 'teleporter': True}

LEVELS = [{'budget': 24,
  'budget_unit': 2,
  'rows': ['########',
           '##.#...#',
           '#..#..##',
           '#..#..K#',
           '#..#...#',
           '#..KD..#',
           '#....A##',
           '########']},
 {'budget': 21,
  'budget_unit': 2,
  'rows': ['#########',
           '#.......#',
           '##.#....#',
           '#...#...#',
           '#...#KK.#',
           '#D...1..#',
           '#.A.#.1.#',
           '#########']},
 {'budget': 18,
  'budget_unit': 1,
  'rows': ['##########',
           '###21....#',
           '#........#',
           '#..#D.D.##',
           '#.##.2...#',
           '##.#.K...#',
           '#..#.K.1K#',
           '#......A.#',
           '#....#..##',
           '##########']},
 {'budget': 33,
  'budget_unit': 2,
  'rows': ['###########',
           '#..#....2.#',
           '#1.####...#',
           '#..#...1..#',
           '#..##.##..#',
           '####D.....#',
           '#.K...K#..#',
           '#.D...2.#.#',
           '#...#.K...#',
           '#....#..A.#',
           '###########']},
 {'budget': 45,
  'budget_unit': 2,
  'rows': ['#############',
           '#..D1..2A..##',
           '#.......#...#',
           '#.#....######',
           '#...#.#D#.#.#',
           '#.KK.....##.#',
           '#K.2#.#.#...#',
           '#D#.##......#',
           '#.....#..#1##',
           '#....#K...#.#',
           '#############']},
 {'budget': 75,
  'budget_unit': 2,
  'rows': ['################',
           '#.D.#.#.....#..#',
           '#A..#.#...#.##D#',
           '##..#..######..#',
           '#...#..#1..K#..#',
           '#...#..#.#.K..K#',
           '#1..#..K...D##.#',
           '#..#.....#.#####',
           '#...#K..##...#.#',
           '#.#...#.D..##2.#',
           '#.##.....#...2.#',
           '################']}]

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, GameState, Level, Sprite

CELL = 3
HUD = 4
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
ONEWAY = {"^": 0, ">": 1, "v": 2, "<": 3}
GLYPHS = [
    [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
    [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
]
ACTIONS = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3, 4: GameAction.ACTION4}
# Goal kinds (CONFIG "goal"; 2026-10-30, the goal-rung set). Absent: the tile goal (an agent on G). "keys": the
# level completes when every key is collected; "plates": when every plate is covered by a block. Neither has a G
# tile. CONFIG "show_goal" SIGNALS the goal on the HUD's last row (one pip per item, filled as it is done); without
# it the goal shows only by its consequence, the level completing. Both read state export_state already carries.
GOAL_ITEMS = {"keys": "K", "plates": "P"}


def _solid(color, w=CELL, h=CELL):
    return [[color] * w for _ in range(h)]


def _mask(pattern, color):
    return [[color if v else -1 for v in row] for row in pattern]


def _ring(color):
    return [[color, color, color], [color, -1, color], [color, color, color]]


def _dot(color):
    return [[-1, -1, -1], [-1, color, -1], [-1, -1, -1]]


def _px(cx, cy):
    return cx * CELL, HUD + cy * CELL


def _pips(n, done, pal, goal):
    """The goal row: one 2-px pip per item, the done ones in the item's colour, the rest in the goal colour."""
    row = []
    for i in range(n):
        row += [pal["key" if goal == "keys" else "plate"] if i < done else pal["goal"]] * 2 + [-1]
    return [row[:-1] if row else [-1]]


def build_level(spec, cfg):
    """A Level from one LEVELS entry: {"rows": [...], "budget": n, "need_glyph": k}."""
    rows = spec["rows"]
    h, w = len(rows), len(rows[0])
    pal = cfg["palette"]
    sprites = [Sprite(_solid(pal["band"], w * CELL, HUD), name="band", x=0, y=0, layer=-2,
                      blocking=BlockingMode.NOT_BLOCKED, collidable=False, tags=["hud"])]
    tele_colors = {}
    n_agent = n_block = n_key = n_door = n_goal = n_tp = 0
    for cy, row in enumerate(rows):
        cx = 0
        while cx < w:
            ch = row[cx]
            x, y = _px(cx, cy)
            if ch == "#":
                run = 1
                while cx + run < w and row[cx + run] == "#":
                    run += 1
                sprites.append(Sprite(_solid(pal["wall"], run * CELL), name="wall_%d_%d" % (cx, cy), x=x, y=y,
                                      tags=["wall"]))
                cx += run
                continue
            if ch == "A" or ch == "S":
                k = cfg.get("start_glyph", 0)
                sprites.append(Sprite(_mask(GLYPHS[k], pal["glyph"][k]), name="agent_%d" % n_agent, x=x, y=y, layer=3,
                                      blocking=BlockingMode.BOUNDING_BOX, tags=["agent"]))
                n_agent += 1
            elif ch == "G":
                k = spec.get("need_glyph", -1)
                col = pal["glyph"][k] if k >= 0 else pal["goal"]
                sprites.append(Sprite(_ring(col), name="goal_%d" % n_goal, x=x, y=y, layer=0, collidable=False,
                                      tags=["goal"]))
                n_goal += 1
            elif ch == "X":
                sprites.append(Sprite(_mask([[1, 0, 1], [0, 1, 0], [1, 0, 1]], pal["hazard"]), name="hz_%d_%d" % (cx, cy),
                                      x=x, y=y, layer=0, collidable=False, tags=["hazard"]))
            elif ch == "K":
                sprites.append(Sprite(_mask([[0, 1, 0], [1, 1, 1], [0, 1, 1]], pal["key"]), name="key_%d" % n_key, x=x,
                                      y=y, layer=1, collidable=False, tags=["key"]))
                n_key += 1
            elif ch == "D":
                sprites.append(Sprite(_mask([[1, 1, 1], [1, 0, 1], [1, 1, 1]], pal["door"]), name="door_%d" % n_door,
                                      x=x, y=y, layer=1, tags=["door"]))
                n_door += 1
            elif ch == "B":
                sprites.append(Sprite(_solid(pal["block"]), name="block_%d" % n_block, x=x, y=y, layer=2,
                                      blocking=BlockingMode.BOUNDING_BOX, tags=["block"]))
                n_block += 1
            elif ch == "P":
                sprites.append(Sprite(_dot(pal["plate"]), name="plate_%d_%d" % (cx, cy), x=x, y=y, layer=0,
                                      collidable=False, tags=["plate"]))
            elif ch in "1234":
                col = pal["tele"][int(ch) - 1]
                sprites.append(Sprite(_mask([[1, 1, 1], [1, 0, 1], [1, 1, 1]], col), name="tp_%s_%d" % (ch, n_tp), x=x,
                                      y=y, layer=0, collidable=False, tags=["tele"]))
                n_tp += 1
            elif ch in ONEWAY:
                d = ONEWAY[ch]
                arrow = {0: [[0, 1, 0], [1, 1, 1], [0, 0, 0]], 1: [[0, 1, 0], [0, 1, 1], [0, 1, 0]],
                         2: [[0, 0, 0], [1, 1, 1], [0, 1, 0]], 3: [[0, 1, 0], [1, 1, 0], [0, 1, 0]]}[d]
                sprites.append(Sprite(_mask(arrow, pal["oneway"]), name="ow_%d_%d" % (cx, cy), x=x, y=y, layer=0,
                                      collidable=False, tags=["oneway"]))
            elif ch == "I":
                sprites.append(Sprite(_solid(pal["ice"]), name="ice_%d_%d" % (cx, cy), x=x, y=y, layer=-1,
                                      collidable=False, tags=["ice"]))
            elif ch == "C":
                sprites.append(Sprite(_mask([[0, 1, 0], [1, 1, 1], [0, 1, 0]], pal["cross"]), name="cross_%d_%d" % (cx, cy),
                                      x=x, y=y, layer=0, collidable=False, tags=["cross"]))
            cx += 1
    # HUD: glyph indicator, key slots, budget bar
    if cfg.get("state_flip"):
        k = cfg.get("start_glyph", 0)
        sprites.append(Sprite(_mask(GLYPHS[k], pal["glyph"][k]), name="hud_glyph", x=1, y=0, layer=5, collidable=False,
                              tags=["hud"]))
    if cfg.get("keydoor"):
        sprites.append(Sprite([[-1]], name="hud_keys", x=6, y=1, layer=5, collidable=False, tags=["hud"]))
    if spec.get("budget", 0) > 0:
        unit = spec.get("budget_unit", 1)
        n = -(-spec["budget"] // unit)
        sprites.append(Sprite([[pal["budget"]] * n], name="hud_budget", x=max(0, w * CELL - n - 1), y=1,
                              layer=5, collidable=False, tags=["hud"]))
    goal = cfg.get("goal")
    if cfg.get("show_goal") and goal in GOAL_ITEMS:
        n = sum(r.count(GOAL_ITEMS[goal]) for r in rows)
        sprites.append(Sprite(_pips(n, 0, pal, goal), name="hud_goal", x=1, y=HUD - 1, layer=5, collidable=False,
                              tags=["hud"]))
    return Level(sprites=sprites, grid_size=(w * CELL, HUD + h * CELL), data={"rows": list(rows), "budget": spec.get("budget", 0),
                                                           "budget_unit": spec.get("budget_unit", 1),
                                                           "need_glyph": spec.get("need_glyph", -1)})


class GenGame(ARCBaseGame):
    def __init__(self, game_id, cfg, level_specs):
        self._cfg = cfg
        self._st = {}
        self._dir_actions = {ACTIONS[int(a)]: d for a, d in cfg["dir_map"].items()}
        levels = [build_level(s, cfg) for s in level_specs]
        w = len(level_specs[0]["rows"][0])
        h = len(level_specs[0]["rows"])
        for s in level_specs:
            w = max(w, len(s["rows"][0]))
            h = max(h, len(s["rows"]))
        cam = Camera(width=w * CELL, height=HUD + h * CELL, background=cfg["palette"]["bg"],
                     letter_box=cfg["palette"]["letterbox"])
        acts = sorted(int(a) for a in cfg["dir_map"])
        if cfg.get("switch"):
            acts.append(5)
        super().__init__(game_id=game_id, levels=levels, camera=cam, available_actions=acts)

    # ---- level state -------------------------------------------------------------------------
    def on_set_level(self, level):
        rows = level.get_data("rows")
        self._rows = rows
        self._w, self._h = len(rows[0]), len(rows)
        self._tele = {}
        for cy, row in enumerate(rows):
            for cx, ch in enumerate(row):
                if ch in "1234":
                    self._tele.setdefault(ch, []).append((cx, cy))
        self._st = {"active": 0, "glyph": self._cfg.get("start_glyph", 0), "keys": 0, "removed": set(),
                    "trail": [], "budget": level.get_data("budget"), "done": False}
        self._refresh_hud()

    def _cell(self, cx, cy):
        if 0 <= cx < self._w and 0 <= cy < self._h:
            return self._rows[cy][cx]
        return "#"

    def _agents(self):
        return sorted(self.current_level.get_sprites_by_tag("agent"), key=lambda s: s.name)

    def _pos(self, s):
        return s.x // CELL, (s.y - HUD) // CELL

    def _by_name(self, name):
        s = self.current_level.get_sprites_by_name(name)
        return s[0] if s else None

    def _live(self, tag, cx, cy):
        """The first non-removed sprite with `tag` at the cell, else None."""
        x, y = _px(cx, cy)
        for s in self.current_level.get_sprites_by_tag(tag):
            if s.x == x and s.y == y and s.name not in self._st["removed"]:
                return s
        return None

    def _remove(self, s):
        from arcengine import InteractionMode
        s.set_interaction(InteractionMode.REMOVED)
        self._st["removed"].add(s.name)

    # ---- the action ---------------------------------------------------------------------------
    def step(self):
        a = self.action.id
        st = self._st
        acted = False
        if a in self._dir_actions:
            self._move(self._dir_actions[a])
            acted = True
        elif a == GameAction.ACTION5 and self._cfg.get("switch"):
            st["active"] = (st["active"] + 1) % max(1, len(self._agents()))
            acted = True
        if self._state != GameState.GAME_OVER:
            # the engine runs step() after a RESET too: only a move or a switch spends the budget
            if acted and st["budget"] > 0 and not st["done"]:
                st["budget"] -= 1
            self._refresh_hud()
            if self._goal_met():
                st["done"] = True
                self.next_level()
            elif st["budget"] == 0 and self.current_level.get_data("budget") > 0:
                self.lose()
        self.complete_action()

    def _move(self, d):
        agents = self._agents()
        if not agents:
            return
        ag = agents[self._st["active"] % len(agents)]
        slid = 0
        while True:
            cx, cy = self._pos(ag)
            dx, dy = DIRS[d]
            tx, ty = cx + dx, cy + dy
            wrapped = False
            if self._cfg.get("wrap") and not (0 <= tx < self._w and 0 <= ty < self._h):
                tx, ty = tx % self._w, ty % self._h
                wrapped = True
            tch = self._cell(tx, ty)
            if tch == "#":
                return
            if tch in ONEWAY and ONEWAY[tch] != d:
                return
            if self._cfg.get("snake") and (tx, ty) in self._st["trail"]:
                return
            # a door opens with a key; a block is pushed first
            door = self._live("door", tx, ty)
            if door is not None:
                if self._st["keys"] <= 0:
                    return
                self._st["keys"] -= 1
                self._remove(door)
            block = self._block_at(tx, ty)
            if block is not None:
                bx, by = tx + dx, ty + dy
                bch = self._cell(bx, by)
                if bch == "#" or bch in ONEWAY or self._block_at(bx, by) is not None or self._agent_at(bx, by) is not None \
                        or self._live("door", bx, by) is not None or (self._cfg.get("wrap") and not (0 <= bx < self._w and 0 <= by < self._h)):
                    return
                if not (0 <= bx < self._w and 0 <= by < self._h):
                    return
                block.set_position(*_px(bx, by))
            if wrapped:
                if self._agent_at(tx, ty) is not None:
                    return
                old = (cx, cy)
                ag.set_position(*_px(tx, ty))
            else:
                old = (cx, cy)
                if self.try_move_sprite(ag, dx * CELL, dy * CELL):
                    return
            if self._cfg.get("snake"):
                self._st["trail"].append(old)
                self._st["trail"] = self._st["trail"][-self._cfg["snake"]:]
                self._rebuild_trail()
            if not self._on_enter(ag, tx, ty):
                return
            if self._cfg.get("ice") and self._cell(*self._pos(ag)) == "I" and slid < 64:
                slid += 1
                continue
            return

    def _on_enter(self, ag, cx, cy):
        """Cell effects after a successful step; False ends the move (death or teleport)."""
        ch = self._cell(cx, cy)
        if ch == "X":
            self.lose()
            return False
        key = self._live("key", cx, cy)
        if key is not None:
            self._remove(key)
            self._st["keys"] += 1
        if ch == "C" and self._cfg.get("state_flip"):
            self._st["glyph"] = (self._st["glyph"] + 1) % self._cfg["state_flip"]
            self._set_glyph(ag)
        if ch in "1234" and self._cfg.get("teleporter"):
            pads = self._tele.get(ch, [])
            others = [p for p in pads if p != (cx, cy)]
            if others:
                dest = others[0]
                if self._agent_at(*dest) is None and self._block_at(*dest) is None:
                    ag.set_position(*_px(*dest))
            return False
        return True

    def _block_at(self, cx, cy):
        x, y = _px(cx, cy)
        for s in self.current_level.get_sprites_by_tag("block"):
            if s.x == x and s.y == y:
                return s
        return None

    def _agent_at(self, cx, cy):
        x, y = _px(cx, cy)
        for s in self.current_level.get_sprites_by_tag("agent"):
            if s.x == x and s.y == y:
                return s
        return None

    def _goal_progress(self):
        """(items, done) for a keys or plates goal: keys collected are the removed key sprites (a door spends a
        key from the count but not from the collection), plates covered are those under a block."""
        goal = self._cfg.get("goal")
        if goal == "keys":
            keys = [s.name for s in self.current_level.get_sprites_by_tag("key")]
            return len(keys), sum(1 for k in keys if k in self._st["removed"])
        plates = {self._pos(p) for p in self.current_level.get_sprites_by_tag("plate")}
        blocks = {self._pos(b) for b in self.current_level.get_sprites_by_tag("block")}
        return len(plates), len(plates & blocks)

    def _goal_met(self):
        if self._cfg.get("goal") in GOAL_ITEMS:
            n, done = self._goal_progress()
            return n > 0 and done == n
        goals = {self._pos(g) for g in self.current_level.get_sprites_by_tag("goal")}
        if not goals:
            return False
        if self._cfg.get("plates"):
            plates = {self._pos(p) for p in self.current_level.get_sprites_by_tag("plate")}
            blocks = {self._pos(b) for b in self.current_level.get_sprites_by_tag("block")}
            if plates and not plates <= blocks:
                return False
        need = self.current_level.get_data("need_glyph")
        if need >= 0 and self._st["glyph"] != need:
            return False
        agents = self._agents()
        if self._cfg.get("switch"):
            return all(self._pos(a) in goals for a in agents)
        return self._pos(agents[self._st["active"] % len(agents)]) in goals

    # ---- visuals -------------------------------------------------------------------------------
    def _set_glyph(self, ag):
        k = self._st["glyph"]
        ag.pixels = np.array(_mask(GLYPHS[k], self._cfg["palette"]["glyph"][k]), dtype=np.int8)

    def _refresh_hud(self):
        pal = self._cfg["palette"]
        g = self._by_name("hud_glyph")
        if g is not None:
            k = self._st["glyph"]
            g.pixels = np.array(_mask(GLYPHS[k], pal["glyph"][k]), dtype=np.int8)
        kk = self._by_name("hud_keys")
        if kk is not None:
            n = self._st["keys"]
            kk.pixels = np.array([[pal["key"]] * (2 * n) if n else [-1]], dtype=np.int8)
        b = self._by_name("hud_budget")
        if b is not None:
            unit = self.current_level.get_data("budget_unit")
            n = -(-self._st["budget"] // unit)
            b.pixels = np.array([[pal["budget"]] * n if n > 0 else [-1]], dtype=np.int8)
        hg = self._by_name("hud_goal")
        if hg is not None:
            n, done = self._goal_progress()
            hg.pixels = np.array(_pips(n, done, pal, self._cfg.get("goal")), dtype=np.int8)

    def _rebuild_trail(self):
        lv = self.current_level
        for s in list(lv.get_sprites_by_tag("trail")):
            lv.remove_sprite(s)
        col = self._cfg["palette"]["trail"]
        for i, (cx, cy) in enumerate(self._st["trail"]):
            x, y = _px(cx, cy)
            lv.add_sprite(Sprite(_solid(col), name="trail_%d" % i, x=x, y=y, layer=2, blocking=BlockingMode.BOUNDING_BOX,
                                 tags=["trail"]))

    # ---- snapshots (used by the oracle; not by play) --------------------------------------------
    def export_state(self):
        st = self._st
        return (self._current_level_index, self._score, self._state.value,
                tuple(self._pos(a) for a in self._agents()), st["active"], st["glyph"], st["keys"],
                tuple(sorted(st["removed"])),
                tuple(self._pos(b) for b in sorted(self.current_level.get_sprites_by_tag("block"), key=lambda s: s.name)),
                tuple(st["trail"]), st["budget"], st["done"], self._action_count)

    def import_state(self, snap):
        from arcengine import InteractionMode
        (li, score, state, agents, active, glyph, keys, removed, blocks, trail, budget, done, count) = snap
        if li != self._current_level_index:
            self.set_level(li)
        self._score = score
        self._state = GameState(state)
        self._action_count = count
        st = self._st
        st.update({"active": active, "glyph": glyph, "keys": keys, "removed": set(removed), "trail": list(trail),
                   "budget": budget, "done": done})
        for a, p in zip(self._agents(), agents):
            a.set_position(*_px(*p))
            self._set_glyph(a)
        for b, p in zip(sorted(self.current_level.get_sprites_by_tag("block"), key=lambda s: s.name), blocks):
            b.set_position(*_px(*p))
        for tag in ("key", "door"):
            for s in self.current_level.get_sprites_by_tag(tag):
                s.set_interaction(InteractionMode.REMOVED if s.name in st["removed"] else
                                  (InteractionMode.TANGIBLE if tag == "door" else InteractionMode.INTANGIBLE))
        if self._cfg.get("snake"):
            self._rebuild_trail()
        self._refresh_hud()


class Gsg6(GenGame):
    def __init__(self):
        super().__init__('gsg6', CONFIG, LEVELS)
