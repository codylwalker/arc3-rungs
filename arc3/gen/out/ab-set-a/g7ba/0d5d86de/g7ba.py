# Generated ARC-AGI-3-style game (house-side, uncertified unless cert.json says otherwise).
# MIT-0 for the emitted file; mechanics are ideas, not code, from the public set.
CONFIG = {'kind': 'move',
 'mechanics': ['click_move'],
 'palette': {'ball': 2,
             'band': 13,
             'bg': 15,
             'block': 9,
             'budget': 5,
             'cross': 2,
             'door': 0,
             'floor': 8,
             'glyph': [7, 10, 9, 13],
             'goal': 12,
             'hazard': 10,
             'ice': 8,
             'key': 14,
             'letterbox': 3,
             'off': 0,
             'on': 14,
             'oneway': 7,
             'plate': 11,
             'tele': [8, 3, 6, 5],
             'trail': 4,
             'wall': 6},
 'seed': 202}

LEVELS = [{'budget': 9, 'budget_unit': 1, 'rows': ['#######', '#.#..O#', '#.#...#', '#G.^..#', '#######']},
 {'budget': 9,
  'budget_unit': 1,
  'rows': ['########', '#......#', '#O###..#', '#......#', '#^#.#.G#', '########']},
 {'budget': 21,
  'budget_unit': 2,
  'rows': ['#########',
           '##.....G#',
           '#.####..#',
           '#.....#.#',
           '#.......#',
           '#......v#',
           '#.O>...##',
           '#########']},
 {'budget': 12,
  'budget_unit': 1,
  'rows': ['#########',
           '#..#.^..#',
           '#G...#..#',
           '#.>.....#',
           '##..#...#',
           '##......#',
           '####.O..#',
           '#########']},
 {'budget': 21,
  'budget_unit': 1,
  'rows': ['##########',
           '#..O.#..##',
           '#...>#v..#',
           '##...#...#',
           '#.##.....#',
           '##...^...#',
           '#..#.....#',
           '#.#.#..G##',
           '##########']},
 {'budget': 12,
  'budget_unit': 1,
  'rows': ['############',
           '#..#...#..v#',
           '#.v#..#..#G#',
           '#.#.#......#',
           '#....#.#^..#',
           '#...#.#...##',
           '##..v.##..##',
           '#.O.......##',
           '############']}]

import numpy as np
from arcengine import ARCBaseGame, BlockingMode, Camera, GameAction, GameState, Level, Sprite

CELL = 3
HUD = 6
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ARROWS = {"^": 0, ">": 1, "v": 2, "<": 3}


def _solid(color, w=CELL, h=CELL):
    return [[color] * w for _ in range(h)]


def _mask(pattern, color):
    return [[color if v else -1 for v in row] for row in pattern]


def _px(cx, cy):
    return cx * CELL, HUD + cy * CELL


def build_level(spec, cfg):
    pal = cfg["palette"]
    kind = cfg["kind"]
    if kind == "toggle":
        h, w = len(spec["target"]), len(spec["target"][0])
    else:
        h, w = len(spec["rows"]), len(spec["rows"][0])
    sprites = [Sprite(_solid(pal["band"], w * CELL, HUD), name="band", x=0, y=0, layer=-2,
                      blocking=BlockingMode.NOT_BLOCKED, collidable=False, tags=["hud"])]
    if kind == "toggle":
        for cy in range(h):
            for cx in range(w):
                x, y = _px(cx, cy)
                on = spec["state"][cy][cx]
                sprites.append(Sprite(_solid(pal["on"] if on else pal["off"]), name="tile_%d_%d" % (cx, cy), x=x, y=y,
                                      layer=0, blocking=BlockingMode.BOUNDING_BOX, tags=["tile", "sys_click"]))
        mini = [[pal["on"] if v else pal["off"] for v in row] for row in spec["target"]]
        sprites.append(Sprite(mini, name="hud_target", x=1, y=1, layer=5, collidable=False, tags=["hud"]))
    else:
        for cy, row in enumerate(spec["rows"]):
            for cx, ch in enumerate(row):
                x, y = _px(cx, cy)
                if ch == "#":
                    sprites.append(Sprite(_solid(pal["wall"]), name="wall_%d_%d" % (cx, cy), x=x, y=y, tags=["wall"]))
                elif ch == "G":
                    sprites.append(Sprite([[pal["goal"], pal["goal"], pal["goal"]], [pal["goal"], -1, pal["goal"]],
                                           [pal["goal"], pal["goal"], pal["goal"]]], name="goal", x=x, y=y, layer=0,
                                          collidable=False, tags=["goal"]))
                elif ch == "O":
                    sprites.append(Sprite(_mask([[0, 1, 0], [1, 1, 1], [0, 1, 0]], pal["ball"]), name="ball", x=x, y=y,
                                          layer=3, blocking=BlockingMode.BOUNDING_BOX, tags=["ball", "sys_click"]))
                elif ch in ARROWS:
                    d = ARROWS[ch]
                    arrow = {0: [[0, 1, 0], [1, 1, 1], [1, 0, 1]], 1: [[1, 1, 0], [0, 1, 1], [1, 1, 0]],
                             2: [[1, 0, 1], [1, 1, 1], [0, 1, 0]], 3: [[0, 1, 1], [1, 1, 0], [0, 1, 1]]}[d]
                    sprites.append(Sprite(_mask(arrow, pal["block"]), name="arrow_%d_%d" % (cx, cy), x=x, y=y, layer=2,
                                          blocking=BlockingMode.BOUNDING_BOX, tags=["arrow", "sys_click", "d%d" % d]))
                if ch != "#":
                    sprites.append(Sprite(_solid(pal["floor"]), name="floor_%d_%d" % (cx, cy), x=x, y=y, layer=-1,
                                          collidable=False, tags=["floor", "sys_click"]))
    if spec.get("budget", 0) > 0:
        unit = spec.get("budget_unit", 1)
        n = -(-spec["budget"] // unit)
        sprites.append(Sprite([[pal["budget"]] * n], name="hud_budget", x=max(0, w * CELL - n - 1), y=HUD - 2, layer=5,
                              collidable=False, tags=["hud"]))
    data = {"w": w, "h": h, "budget": spec.get("budget", 0), "budget_unit": spec.get("budget_unit", 1)}
    if kind == "toggle":
        data["target"] = [list(r) for r in spec["target"]]
        data["state0"] = [list(r) for r in spec["state"]]
    else:
        data["rows"] = list(spec["rows"])
    return Level(sprites=sprites, grid_size=(w * CELL, HUD + h * CELL), data=data)


class GenClickGame(ARCBaseGame):
    def __init__(self, game_id, cfg, level_specs):
        self._cfg = cfg
        self._st = {}
        levels = [build_level(s, cfg) for s in level_specs]
        w = max(l.grid_size[0] for l in levels)
        h = max(l.grid_size[1] for l in levels)
        cam = Camera(width=w, height=h, background=cfg["palette"]["bg"], letter_box=cfg["palette"]["letterbox"])
        super().__init__(game_id=game_id, levels=levels, camera=cam, available_actions=[6])

    def on_set_level(self, level):
        self._w, self._h = level.get_data("w"), level.get_data("h")
        self._st = {"budget": level.get_data("budget"), "done": False}
        if self._cfg["kind"] == "toggle":
            self._st["state"] = [list(r) for r in level.get_data("state0")]
        else:
            self._rows = level.get_data("rows")
        self._refresh()

    def _by_name(self, name):
        s = self.current_level.get_sprites_by_name(name)
        return s[0] if s else None

    def _pos(self, s):
        return s.x // CELL, (s.y - HUD) // CELL

    def _click_cell(self):
        d = self.action.data or {}
        g = self.camera.display_to_grid(int(d.get("x", 0)), int(d.get("y", 0)))
        if g is None:
            return None
        gx, gy = g
        if gy < HUD:
            return None
        cx, cy = gx // CELL, (gy - HUD) // CELL
        if 0 <= cx < self._w and 0 <= cy < self._h:
            return cx, cy
        return None

    def step(self):
        st = self._st
        if self.action.id == GameAction.ACTION6:
            cell = self._click_cell()
            if cell is not None:
                if self._cfg["kind"] == "toggle":
                    self._flip(*cell)
                else:
                    self._click_move(*cell)
            if st["budget"] > 0 and not st["done"]:
                st["budget"] -= 1
        self._refresh()
        if self._goal_met():
            st["done"] = True
            self.next_level()
        elif st["budget"] == 0 and self.current_level.get_data("budget") > 0:
            self.lose()
        self.complete_action()

    # ---- toggle ---------------------------------------------------------------------------------
    def _flip(self, cx, cy):
        rule = self._cfg.get("rule", "self")
        cells = [(cx, cy)]
        if rule == "plus":
            cells += [(cx + dx, cy + dy) for dx, dy in DIRS]
        elif rule == "rowcol":
            cells = [(x, cy) for x in range(self._w)] + [(cx, y) for y in range(self._h) if y != cy]
        s = self._st["state"]
        for x, y in cells:
            if 0 <= x < self._w and 0 <= y < self._h:
                s[y][x] ^= 1

    # ---- move -----------------------------------------------------------------------------------
    def _cell(self, cx, cy):
        if 0 <= cx < self._w and 0 <= cy < self._h:
            return self._rows[cy][cx]
        return "#"

    def _occupied(self, cx, cy):
        if self._cell(cx, cy) == "#":
            return True
        x, y = _px(cx, cy)
        for s in self.current_level.get_sprites_by_tag("arrow") + self.current_level.get_sprites_by_tag("ball"):
            if s.x == x and s.y == y:
                return True
        return False

    def _slide(self, sprite, d):
        dx, dy = DIRS[d]
        cx, cy = self._pos(sprite)
        while not self._occupied(cx + dx, cy + dy):
            cx, cy = cx + dx, cy + dy
        sprite.set_position(*_px(cx, cy))

    def _click_move(self, cx, cy):
        x, y = _px(cx, cy)
        for s in self.current_level.get_sprites_by_tag("arrow"):
            if s.x == x and s.y == y:
                d = int([t for t in s.tags if t.startswith("d")][0][1:])
                self._slide(s, d)
                return
        ball = self._by_name("ball")
        bx, by = self._pos(ball)
        if (cx, cy) == (bx, by):
            return
        if cy == by:
            self._slide(ball, 1 if cx > bx else 3)
        elif cx == bx:
            self._slide(ball, 2 if cy > by else 0)

    # ---- goal and visuals -----------------------------------------------------------------------
    def _goal_met(self):
        if self._cfg["kind"] == "toggle":
            return self._st["state"] == self.current_level.get_data("target")
        g = self._by_name("goal")
        return g is not None and self._pos(self._by_name("ball")) == self._pos(g)

    def _refresh(self):
        pal = self._cfg["palette"]
        if self._cfg["kind"] == "toggle":
            for cy in range(self._h):
                for cx in range(self._w):
                    t = self._by_name("tile_%d_%d" % (cx, cy))
                    on = self._st["state"][cy][cx]
                    t.pixels = np.array(_solid(pal["on"] if on else pal["off"]), dtype=np.int8)
        b = self._by_name("hud_budget")
        if b is not None:
            unit = self.current_level.get_data("budget_unit")
            n = -(-self._st["budget"] // unit)
            b.pixels = np.array([[pal["budget"]] * n if n > 0 else [-1]], dtype=np.int8)

    # ---- snapshots ------------------------------------------------------------------------------
    def export_state(self):
        st = self._st
        if self._cfg["kind"] == "toggle":
            core = tuple(tuple(r) for r in st["state"])
        else:
            core = (self._pos(self._by_name("ball")),
                    tuple(self._pos(s) for s in sorted(self.current_level.get_sprites_by_tag("arrow"), key=lambda s: s.name)))
        return (self._current_level_index, self._score, self._state.value, core, st["budget"], st["done"], self._action_count)

    def import_state(self, snap):
        li, score, state, core, budget, done, count = snap
        if li != self._current_level_index:
            self.set_level(li)
        self._score = score
        self._state = GameState(state)
        self._action_count = count
        self._st.update({"budget": budget, "done": done})
        if self._cfg["kind"] == "toggle":
            self._st["state"] = [list(r) for r in core]
        else:
            self._by_name("ball").set_position(*_px(*core[0]))
            for s, p in zip(sorted(self.current_level.get_sprites_by_tag("arrow"), key=lambda s: s.name), core[1]):
                s.set_position(*_px(*p))
        self._refresh()


class G7ba(GenClickGame):
    def __init__(self):
        super().__init__('g7ba', CONFIG, LEVELS)
