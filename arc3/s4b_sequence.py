"""S4 v1 - sequenced goal hypotheses with state-change events (movement games).

Per game:
  survey   enter (overlap) every candidate object once, nearest first; record EVENTS: a contact
           after which some object away from the agent changed (recolour/reshape/move/appear/
           vanish), ignoring 'clock' objects that change on most steps (energy bars, counters).
  pairs    for A in event-producing objects (then the rest), for B in the others, nearest first:
           enter A then enter B (press into it). A level-up fixes the goal SEQUENCE by object
           keys (colour, shape); later levels replay it by key, falling back to survey.
Planning: optimistic cost search in the induced passability model (per-colour roles + failed
edges + recently occupied cells as a body). Budget per game: 3x human baseline sum, cap 600.
Usage: python s4b_sequence.py [--budget N] [--debug K] [game ...]
"""
import argparse
import heapq
import os
import json
import time
from collections import Counter, deque

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcobj import DIRS, Tracker, deltas, segment

ACT = {a.value: a for a in GameAction}
ACT[5] = GameAction.ACTION5
MOVE_GAMES = ["ls20", "m0r0", "tu93", "g50t", "re86", "tr87", "wa30", "ar25", "bp35", "dc22", "ka59", "sp80", "cn04", "sk48"]


def human_baselines(game):
    try:
        return json.load(open("arc3_t1.json", encoding="utf-8"))["baselines"].get(game)
    except Exception:  # noqa: BLE001
        return None


class Model:
    def __init__(self, bg):
        self.bg = bg
        self.role = {}
        self.evidence = Counter()
        self.blocked_edges = set()
        self.step = 1
        self.body = deque(maxlen=12)  # recently occupied agent positions (snake body proxy)
        self.trail = False  # set once the agent is seen to leave its colour behind
        self.wraps = False
        self.teleports = {}  # (x0, y0, action) -> (x1, y1): observed non-lattice landings
        self.step_hist = Counter()

    def cell_cost(self, colour):
        r = self.role.get(colour)
        if r == "kill":
            return None
        if r in ("pass", "goal"):
            return 1
        if r == "block":
            return 12
        return 2


def overlap(ax0, ay0, aw, ah, o):
    return ax0 <= o.x1 and o.x0 <= ax0 + aw - 1 and ay0 <= o.y1 and o.y0 <= ay0 + ah - 1


def plan_to(frame, model, agent, target, max_nodes=30000, enter=True):
    H = W = 64
    aw, ah = agent.w, agent.h
    start = (agent.x0, agent.y0)
    goal_set = set(target.cells)
    body = set(model.body) if model.trail else set()

    def move_cost(x, y):
        if x < 0 or y < 0 or x + aw > W or y + ah > H:
            return None
        worst = 1
        for i in range(aw):
            for j in range(ah):
                cx, cy = x + i, y + j
                if (cx, cy) in goal_set:
                    continue
                if (cx, cy) in body:
                    return None
                c = model.cell_cost(int(frame[cy][cx]))
                if c is None:
                    return None
                worst = max(worst, c)
        return worst

    def done(x, y):
        if enter:
            return overlap(x, y, aw, ah, target)
        ddx = max(target.x0 - (x + aw - 1), x - target.x1, 0)
        ddy = max(target.y0 - (y + ah - 1), y - target.y1, 0)
        return max(ddx, ddy) <= model.step

    prev = {start: None}
    dist = {start: 0}
    heap = [(0, start)]
    n = 0
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist.get(cur, 1e18):
            continue
        n += 1
        if n > max_nodes:
            return None
        if cur != start and done(*cur):
            path, c = [], cur
            while prev[c] is not None:
                p, a = prev[c]
                path.append(a)
                c = p
            return path[::-1]
        for a, (dx, dy) in DIRS.items():
            if (cur[0], cur[1], a) in model.blocked_edges:
                continue
            tp = model.teleports.get((cur[0], cur[1], a))
            nxt = tp if tp else (cur[0] + dx * model.step, cur[1] + dy * model.step)
            mc = 1 if tp else move_cost(*nxt)
            if mc is None:
                continue
            nd = d + mc
            if nd < dist.get(nxt, 1e18):
                dist[nxt] = nd
                prev[nxt] = (cur, a)
                heapq.heappush(heap, (nd, nxt))
    return None


def reachable_positions(frame, model, agent, max_nodes=4000):
    """Flood over the move lattice from the agent: every position the induced model allows."""
    H = W = 64
    aw, ah = agent.w, agent.h
    start = (agent.x0, agent.y0)
    seen = {start}
    q = deque([start])
    while q and len(seen) < max_nodes:
        cur = q.popleft()
        for a, (dx, dy) in DIRS.items():
            if (cur[0], cur[1], a) in model.blocked_edges:
                continue
            tp = model.teleports.get((cur[0], cur[1], a))
            x, y = tp if tp else (cur[0] + dx * model.step, cur[1] + dy * model.step)
            if (x, y) in seen or x < 0 or y < 0 or x + aw > W or y + ah > H:
                continue
            if tp:
                seen.add((x, y))
                q.append((x, y))
                continue
            ok = True
            for i in range(aw):
                for j in range(ah):
                    if model.cell_cost(int(frame[y + j][x + i])) is None:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                seen.add((x, y))
                q.append((x, y))
    return seen


def norm_cells(o):
    return frozenset((x - o.x0, y - o.y0) for x, y in o.cells)


def shape_eq_scaled(a, b):
    """Same shape up to an integer scale (HUD glyphs are often drawn at 2x the in-game pattern)."""
    if a.shape == b.shape:
        return True
    small, big = (a, b) if a.size <= b.size else (b, a)
    if small.w == 0 or small.h == 0 or big.w % small.w or big.h % small.h:
        return False
    k = big.w // small.w
    if k < 1 or big.h // small.h != k or big.size != small.size * k * k:
        return False
    up = frozenset((x * k + i, y * k + j) for x, y in norm_cells(small) for i in range(k) for j in range(k))
    return up == norm_cells(big)


def push_plan(frame, model, agent, block, goal_cells, max_nodes=40000):
    """Joint search over (agent position, block position). The agent moves one lattice step; if its
    next bbox overlaps the block, the block is pushed by the same displacement provided its own
    next bbox is passable and does not leave the screen. Goal: the block's bbox overlaps goal_cells."""
    W = H = 64
    aw, ah, bw, bh = agent.w, agent.h, block.w, block.h
    st = model.step
    goal = set(goal_cells)
    bcells = frozenset((x - block.x0, y - block.y0) for x, y in block.cells)

    def free(x, y, w, h, ignore_block_at=None):
        if x < 0 or y < 0 or x + w > W or y + h > H:
            return False
        for i in range(w):
            for j in range(h):
                cx, cy = x + i, y + j
                if ignore_block_at and ignore_block_at[0] <= cx < ignore_block_at[0] + bw and ignore_block_at[1] <= cy < ignore_block_at[1] + bh:
                    continue
                c = model.cell_cost(int(frame[cy][cx]))
                if c is None or c >= 12:
                    return False
        return True

    def bbox_overlap(x, y, w, h, bx, by):
        return x <= bx + bw - 1 and bx <= x + w - 1 and y <= by + bh - 1 and by <= y + h - 1

    def done(bx, by):
        return any((bx + dx, by + dy) in goal for dx, dy in bcells)

    start = (agent.x0, agent.y0, block.x0, block.y0)
    prev = {start: None}
    q = deque([start])
    n = 0
    while q:
        cur = q.popleft()
        n += 1
        if n > max_nodes:
            return None
        ax, ay, bx, by = cur
        if done(bx, by):
            path, c = [], cur
            while prev[c] is not None:
                pc, a = prev[c]
                path.append(a)
                c = pc
            return path[::-1]
        for a, (dx, dy) in DIRS.items():
            if (ax, ay, a) in model.blocked_edges:
                continue
            nax, nay = ax + dx * st, ay + dy * st
            nbx, nby = bx, by
            if bbox_overlap(nax, nay, aw, ah, bx, by):
                nbx, nby = bx + dx * st, by + dy * st  # push
                if not free(nbx, nby, bw, bh):
                    continue
                if not free(nax, nay, aw, ah, ignore_block_at=(bx, by)):
                    continue
            elif not free(nax, nay, aw, ah, ignore_block_at=(bx, by)):
                continue
            nxt = (nax, nay, nbx, nby)
            if nxt in prev:
                continue
            prev[nxt] = (cur, a)
            q.append(nxt)
    return None


def band(o):
    """Screen-edge band an object lives in (HUDs live at the edges), else None."""
    if o.y1 <= 5:
        return "top"
    if o.y0 >= 58:
        return "bottom"
    if o.x1 <= 5:
        return "left"
    if o.x0 >= 58:
        return "right"
    return None


def okey(o):
    return (o.color, o.shape)


class Agent:
    def __init__(self, game, budget, debug=0):
        self.game, self.budget, self.debug = game, budget, debug
        self.diag = Counter()
        self.goal_seq = None  # list of object keys

    def candidates(self, objs, agent, attached, reach=None):
        out = []
        for o in objs:
            if overlap(agent.x0, agent.y0, agent.w, agent.h, o) or okey(o) in attached:
                continue
            if reach is not None and not any(overlap(x, y, agent.w, agent.h, o) for (x, y) in reach):
                continue  # nothing reachable overlaps it: not a target
            if self.model.role.get(o.color) == "kill":
                continue
            if self.steps > 6 and band(o) is not None and self.clock_col.get((o.color, band(o)), 0) > 0.5 * self.steps:
                continue  # a clock: this colour in this edge band changes on most steps (energy bars, counters)
            if self.steps > 6 and self.clock_any.get(o.color, 0) > 0.8 * self.steps:
                continue  # a colour that changes on nearly every step anywhere: the floor the agent reveals
            out.append(o)
        return out

    def run(self, arc):
        env = arc.make(self.game, seed=0)
        obs = env.reset()
        tr = Tracker()
        tr.start(obs.frame[-1])
        self.model = Model(tr.bg)
        model = self.model
        self.clock = Counter()
        self.clock_any = Counter()
        self.steps = 0
        attached = set()
        events = {}  # object key -> count of state changes caused
        contacted = set()  # object keys entered this level
        tried_pairs = set()
        plan = []
        target = None
        queue = []  # list of object keys to visit in order (the current hypothesis)
        press_left = 0
        target_fails = 0
        pressing_done = False
        seq_tried_this_level = False
        flips = 0
        switch_budget = 6
        op_left = []
        op_tried = set()
        tried_actions = Counter()
        collect_tried = set()
        pushable = set()
        push_tried = set()
        push_goal = None
        unreachable = set()
        unreach_col = Counter()
        marker_colours = {}  # event producer key -> colours of the far objects it changes
        marker_colours = __import__("collections").defaultdict(set)
        self.goal_relation = getattr(self, "goal_relation", None)  # (A key, marker colour, B key)
        self.clock_col = Counter()
        last_level = obs.levels_completed
        level_actions = []
        cur_level_actions = 0
        actions = 0
        contacted_now = None
        while actions < self.budget:
            frame = obs.frame[-1]
            ag = tr.agent()
            agent = next((o for o in tr.objs if ag and o.tid == ag[0]), None)
            if agent is None:
                a = [1, 2, 3, 4][actions % 4]
                self.diag["no_agent"] += 1
            else:
                if not plan and target is not None and op_left:
                    plan = [op_left.pop(0)]
                    self.diag["operator_tries"] += 1
                if not plan and target is not None and press_left > 0:
                    pressing_done = press_left == 1
                    tcx, tcy = (target.x0 + target.x1) // 2, (target.y0 + target.y1) // 2
                    acx, acy = (agent.x0 + agent.x1) // 2, (agent.y0 + agent.y1) // 2
                    ddx, ddy = tcx - acx, tcy - acy
                    plan = [(4 if ddx > 0 else 3) if abs(ddx) >= abs(ddy) else (2 if ddy > 0 else 1)]
                    press_left -= 1
                if not plan:
                    press_left = 0
                    pressing_done = False
                    _, objs = segment(frame, tr.bg)
                    reach = reachable_positions(frame, model, agent)
                    cands = self.candidates(objs, agent, attached, reach)
                    bykey = {}
                    for o in cands:
                        bykey.setdefault(okey(o), []).append(o)
                    # 0. keep the current target across replans (a blocked step is not a reason to change goal)
                    chosen = None
                    if target is not None and okey(target) in bykey and okey(target) not in contacted and target_fails < 3:
                        chosen = min(bykey[okey(target)], key=lambda o: abs(o.x0 - target.x0) + abs(o.y0 - target.y0))
                        self.diag["retarget"] += 1
                    # 1. a goal sequence known from an earlier level (or forced for a test: S4FORCE="0,5" = colours in order)
                    if chosen is None and self.goal_relation and not queue and flips < 8:
                        ka, m, kb = self.goal_relation
                        akeys = [k for k in bykey if k == ka] or [k for k in bykey if k[0] == ka[0]]
                        # B on this level: an object of B's colour that HOLDS a pattern of the marker colour
                        def pattern_inside(o):
                            return [p for p in objs if p.color == m and p is not o and p.size <= 64 and okey(p) not in attached
                                    and not overlap(agent.x0 - 2, agent.y0 - 2, agent.w + 4, agent.h + 4, p)
                                    and o.x0 <= p.x0 and o.y0 <= p.y0 and o.x1 >= p.x1 and o.y1 >= p.y1]
                        bcands = [o for o in objs if o.color == kb[0] and o.size <= 200 and pattern_inside(o)]
                        bkeys = [okey(o) for o in bcands]
                        if bcands and akeys:
                            bobj = min(bcands, key=lambda o: o.size)
                            inside = pattern_inside(bobj)
                            markers = [o for o in objs if o.color == m and not overlap(bobj.x0 - 2, bobj.y0 - 2, bobj.w + 4, bobj.h + 4, o)
                                       and not overlap(agent.x0 - 2, agent.y0 - 2, agent.w + 4, agent.h + 4, o)]
                            matched = bool(markers and inside) and shape_eq_scaled(max(markers, key=lambda o: o.size), max(inside, key=lambda o: o.size))
                            if self.debug:
                                pm = max(markers, key=lambda o: o.size) if markers else None
                                pi = max(inside, key=lambda o: o.size) if inside else None
                                print(f"  RELATION t={actions} B={(bobj.color, bobj.x0, bobj.y0, bobj.w, bobj.h)} marker={(pm.w, pm.h, sorted(norm_cells(pm))) if pm else None} pattern={(pi.w, pi.h, sorted(norm_cells(pi))) if pi else None} matched={matched} flips={flips}", flush=True)
                            if matched:
                                queue = [okey(bobj)]
                                bykey.setdefault(okey(bobj), []).append(bobj)
                                self.diag["relation_enter"] += 1
                            else:
                                queue = [akeys[0]]
                                self.diag["relation_flip"] += 1
                    if chosen is None and self.goal_seq and not queue and not seq_tried_this_level and not self.goal_relation:
                        queue = []
                        for k in self.goal_seq:  # exact key, else the same colour (the next level draws the box differently)
                            if k in bykey:
                                queue.append(k)
                            else:
                                same = [kk for kk in bykey if kk[0] == k[0] and kk not in queue]
                                if same:
                                    queue.append(max(same, key=lambda kk: max(o.size for o in bykey[kk])))
                        seq_tried_this_level = True
                        self.diag["seq_replays"] += 1
                    if chosen is None and not queue and os.environ.get("S4FORCE") and not getattr(self, "forced", False):
                        queue = []
                        for spec in os.environ["S4FORCE"].split(","):
                            col, _, x0 = spec.partition("@")
                            ks = [k for k in bykey if k[0] == int(col) and (not x0 or any(o.x0 == int(x0) for o in bykey[k]))]
                            if ks:
                                queue.append(ks[0])
                        self.forced = True
                        print("  FORCED queue", queue, flush=True)
                    # 2. the current hypothesis queue
                    while queue and chosen is None:
                        k = queue.pop(0)
                        if k in bykey:
                            chosen = min(bykey[k], key=lambda o: abs(o.x0 - agent.x0) + abs(o.y0 - agent.y0))
                    # 3c. push: a pushable block onto a candidate target (Sokoban)
                    if chosen is None and not plan and pushable:
                        blocks = [o for o in objs if okey(o) in pushable]
                        goals = [o for o in cands if okey(o) not in pushable and okey(o) not in attached]
                        goals.sort(key=lambda o: abs(o.x0 - agent.x0) + abs(o.y0 - agent.y0))
                        for b in blocks:
                            for gobj in goals:
                                if (okey(b), (gobj.color, gobj.x0, gobj.y0)) in push_tried:
                                    continue
                                push_tried.add((okey(b), (gobj.color, gobj.x0, gobj.y0)))
                                pp = push_plan(frame, model, agent, b, gobj.cells)
                                self.diag["push_plans"] += 1
                                if pp:
                                    plan, target, press_left = pp, gobj, 0
                                    push_goal = (okey(b), gobj)
                                    self.diag["push_found"] += 1
                                    if self.debug:
                                        print(f"  PUSH t={actions} agent={(agent.x0, agent.y0, agent.w, agent.h, agent.color)} block={(b.color, b.x0, b.y0, b.w, b.h)} goal={(gobj.color, gobj.x0, gobj.y0, gobj.w, gobj.h)} plan={pp}", flush=True)
                                    break
                            if plan:
                                break
                    # 3. survey: nearest uncontacted object
                    if chosen is None:
                        unvisited = [o for o in cands if okey(o) not in contacted and (o.color, o.x0, o.y0) not in unreachable and unreach_col[okey(o)] < 2]
                        if unvisited:
                            chosen = min(unvisited, key=lambda o: abs(o.x0 - agent.x0) + abs(o.y0 - agent.y0))
                            self.diag["survey"] += 1
                    # 3b. collect-all: several objects of one colour produce events -> visit every one of that colour
                    if chosen is None and not queue:
                        ev_cols = Counter(k[0] for k in events)
                        for col, n in ev_cols.most_common():
                            same = [o for o in cands if o.color == col]
                            if n >= 1 and len(same) >= 2 and col not in collect_tried:
                                collect_tried.add(col)
                                order = []
                                px, py = agent.x0, agent.y0
                                pool = same[:]
                                while pool:  # greedy nearest-first tour
                                    nxt = min(pool, key=lambda o: abs(o.x0 - px) + abs(o.y0 - py))
                                    pool.remove(nxt)
                                    order.append(nxt)
                                    px, py = nxt.x0, nxt.y0
                                for o in order:
                                    queue.append(okey(o))
                                    bykey.setdefault(okey(o), []).append(o)
                                self.diag["collect_all"] += 1
                                break
                        while queue and chosen is None:
                            k = queue.pop(0)
                            if k in bykey:
                                chosen = min(bykey[k], key=lambda o: abs(o.x0 - agent.x0) + abs(o.y0 - agent.y0))
                    # 4. pairs: A (event producers first) then B
                    if chosen is None and not plan:
                        keys = [k for k in bykey if unreach_col[k] < 2 and not all((o.color, o.x0, o.y0) in unreachable for o in bykey[k])]
                        ordered_a = sorted(keys, key=lambda k: (-events.get(k, 0), 0))
                        pair = None
                        for ka in ordered_a:
                            for kb in sorted(keys, key=lambda k: abs(bykey[k][0].x0 - agent.x0) + abs(bykey[k][0].y0 - agent.y0)):
                                if ka != kb and (ka, kb) not in tried_pairs:
                                    pair = (ka, kb)
                                    break
                            if pair:
                                break
                        if pair:
                            tried_pairs.add(pair)
                            queue = [pair[1]]
                            chosen = min(bykey[pair[0]], key=lambda o: abs(o.x0 - agent.x0) + abs(o.y0 - agent.y0))
                            self.diag["pairs"] += 1
                    if chosen is None and GameAction.ACTION5 in env.action_space and switch_budget > 0:
                        switch_budget -= 1
                        tr.moves.clear()
                        tr.class_moves.clear()  # a switch may hand control to another sprite
                        plan, target = [5], None
                        self.diag["switch"] += 1
                    if chosen is None and not plan:
                        contacted = set()  # start over
                        chosen = cands[0] if cands else None
                    if plan:
                        p = plan  # a push plan already chosen
                        chosen = target
                    else:
                        p = plan_to(frame, model, agent, chosen, enter=True) if chosen else None
                        if p is None and chosen is not None:
                            p = plan_to(frame, model, agent, chosen, enter=False)
                    if all((agent.x0, agent.y0, d) in model.blocked_edges for d in (1, 2, 3, 4)):
                        # stuck in every direction: this is not the agent (a twin sprite swapped identities)
                        tr.banned.add(agent.tid)
                        for d in (1, 2, 3, 4):
                            model.blocked_edges.discard((agent.x0, agent.y0, d))
                        self.diag["agent_swap"] += 1
                        p = None
                    if p:
                        if target is None or okey(chosen) != okey(target):
                            target_fails = 0
                        plan, target, press_left = p, chosen, 0  # presses start only on arrival
                        self.diag["plans"] += 1
                    else:
                        untried = [d for d in (1, 2, 3, 4) if (agent.x0, agent.y0, d) not in model.blocked_edges]
                        plan = [(untried or [1, 2, 3, 4])[(actions * 7) % len(untried or [1, 2, 3, 4])]]
                        if chosen is not None:
                            unreachable.add((chosen.color, chosen.x0, chosen.y0))
                            unreach_col[okey(chosen)] += 1
                        target = None
                        self.diag["no_path"] += 1
                        if self.debug:
                            print(f"  NO_PATH t={actions} agent={(agent.x0, agent.y0)} chosen={(chosen.color, chosen.x0, chosen.y0) if chosen else None} blocked={sorted(model.blocked_edges)[:12]} n_blocked={len(model.blocked_edges)} roles={model.role} step={model.step}", flush=True)
                a = plan.pop(0)
            prev_objs = tr.objs
            nobs = env.step(ACT[a], data={})
            actions += 1
            cur_level_actions += 1
            self.steps += 1
            tried_actions[a] += 1
            if nobs is None or not nobs.frame:
                if nobs is not None and nobs.state == GameState.GAME_OVER:
                    nobs = env.reset()
                    tr.start(nobs.frame[-1])
                    obs, plan, target = nobs, [], None
                    continue
                break
            cur_objs = tr.step(a, nobs.frame[-1])
            if self.debug and actions <= self.debug:
                print(f"  t={actions} agent={(agent.x0, agent.y0) if agent else None} a={a} target={(target.color, target.x0, target.y0) if target else None} "
                      f"plan_left={len(plan)} lv={nobs.levels_completed} queue={queue} events={dict(events)} seq={self.goal_seq}", flush=True)
            if agent is None or a not in DIRS:
                obs = nobs
                if nobs.state == GameState.GAME_OVER:
                    nobs = env.reset()
                    tr.start(nobs.frame[-1])
                    obs = nobs
                continue
            c = next((o for o in cur_objs if o.tid == agent.tid), None)
            dx, dy = DIRS[a]
            ahead = [(agent.x1 + 1 if dx > 0 else agent.x0 - 1, y) for y in range(agent.y0, agent.y1 + 1)] if dx else \
                    [(x, agent.y1 + 1 if dy > 0 else agent.y0 - 1) for x in range(agent.x0, agent.x1 + 1)]
            ahead_cols = {int(frame[y][x]) for x, y in ahead if 0 <= x < 64 and 0 <= y < 64}
            # state-change events: objects away from the agent that changed
            if c is not None:
                ds, ncells, _, _ = deltas(frame, nobs.frame[-1], prev_objs=prev_objs, cur_objs=cur_objs, bg=tr.bg)
                far_changes = []
                for d in ds:
                    o = d["obj"]
                    near_prev = overlap(agent.x0 - 1, agent.y0 - 1, agent.w + 2, agent.h + 2, o)
                    near_cur = overlap(c.x0 - 1, c.y0 - 1, c.w + 2, c.h + 2, o)
                    if not near_prev and not near_cur and okey(o) not in attached:
                        far_changes.append(o)
                for key in {(o.color, band(o)) for o in far_changes}:
                    self.clock_col[key] += 1
                for col in {o.color for o in far_changes}:
                    self.clock_any[col] += 1
                real = [o for o in far_changes if not (self.steps > 6 and ((band(o) is not None and self.clock_col[(o.color, band(o))] > 0.5 * self.steps) or self.clock_any[o.color] > 0.8 * self.steps))]
                consumed = False
                if target is not None and overlap(c.x0, c.y0, c.w, c.h, target):
                    # occlusion-tolerant: any object of the target's colour within 2 cells of its bbox still counts as present
                    still = [o for o in cur_objs if o.color == target.color and overlap(target.x0 - 2, target.y0 - 2, target.w + 4, target.h + 4, o)]
                    consumed = False  # disabled: occlusion by the agent makes this unreliable; revisit with an off-target check
                if (real or consumed) and target is not None and overlap(c.x0, c.y0, c.w, c.h, target):
                    events[okey(target)] = events.get(okey(target), 0) + 1
                    self.diag["events"] += 1
                    if consumed:
                        self.diag["consumed"] += 1
                    for o in real:
                        marker_colours[okey(target)].add(o.color)
            if nobs.levels_completed > last_level:
                hist = list(getattr(self, "history_keys", [])) + ([okey(target)] if target is not None else [])
                seq = []
                for k in hist[-3:]:
                    if k not in seq:
                        seq.append(k)
                self.goal_seq = seq
                # relation: an event producer A in the history whose marker (far object of colour m)
                # now has the same shape as an object of colour m inside/near the target B
                if target is not None:
                    _, before_objs = segment(frame, tr.bg)
                    # B is the CONTAINER of the target: the smallest object whose bbox contains it, else the target
                    holders = [o for o in before_objs if o is not target and o.color != target.color and o.size <= 200
                               and o.x0 <= target.x0 and o.y0 <= target.y0 and o.x1 >= target.x1 and o.y1 >= target.y1 and o.size > target.size]
                    B = min(holders, key=lambda o: o.size) if holders else target
                    for ka, cols in marker_colours.items():
                        if ka not in hist:
                            continue
                        for m in cols:
                            inside = [o for o in before_objs if o.color == m and overlap(B.x0, B.y0, B.w, B.h, o)]
                            markers = [o for o in before_objs if o.color == m and not overlap(B.x0, B.y0, B.w, B.h, o)
                                       and not overlap(agent.x0 - 2, agent.y0 - 2, agent.w + 4, agent.h + 4, o)]
                            if markers and inside and shape_eq_scaled(max(markers, key=lambda o: o.size), max(inside, key=lambda o: o.size)):
                                self.goal_relation = (ka, m, okey(B))
                                self.diag["relation"] = str((ka[0], m, okey(B)[0]))
                level_actions.append(cur_level_actions)
                cur_level_actions = 0
                last_level = nobs.levels_completed
                contacted, tried_pairs, queue, plan, target = set(), set(), [], [], None
                collect_tried = set()
                push_tried = set()
                push_goal = None
                seq_tried_this_level = False
                flips = 0
                unreachable = set()
                unreach_col = Counter()
                model.blocked_edges = set()  # the layout changed: edge memory is per level, colour roles are per game
                model.teleports = {}
                model.body.clear()
                tr.start(nobs.frame[-1])
                model.body.clear()
                self.diag["level_ups"] += 1
            elif nobs.state == GameState.GAME_OVER:
                for col in ahead_cols:
                    model.role[col] = "kill"
                plan, target = [], None
                nobs = env.reset()
                tr.start(nobs.frame[-1])
                model.body.clear()
                self.diag["deaths"] += 1
            elif c is None:
                # tracking lost (the sprite changed shape or jumped); not a death
                plan, target = [], None
                tr.start(nobs.frame[-1])
                self.diag["track_lost"] += 1
            elif (c.x0, c.y0) == (agent.x0, agent.y0):
                model.blocked_edges.add((agent.x0, agent.y0, a))
                for col in ahead_cols:
                    model.evidence[(col, "stay")] += 1
                    if model.evidence[(col, "stay")] >= 2 and model.evidence[(col, "pass")] == 0:
                        model.role[col] = "block"
                if press_left > 0 and target is not None:
                    press_left = 0
                    pressing_done = True
                    contacted.add(okey(target))
                    self.history_keys = (getattr(self, "history_keys", []) + [okey(target)])[-4:]
                else:
                    target_fails += 1
                plan = []
            else:
                wdx, wdy = c.x0 - agent.x0, c.y0 - agent.y0
                if abs(wdx) > 32:
                    wdx -= 64 * (1 if wdx > 0 else -1)
                    model.wraps = True
                if abs(wdy) > 32:
                    wdy -= 64 * (1 if wdy > 0 else -1)
                    model.wraps = True
                step = max(abs(wdx), abs(wdy))
                model.step_hist[step] += 1
                common, n_common = model.step_hist.most_common(1)[0]
                if n_common >= 3 and (abs(wdx) + abs(wdy)) >= 3 * common:
                    model.teleports[(agent.x0, agent.y0, a)] = (c.x0, c.y0)  # landed far away: a teleport edge
                    self.diag["teleports"] += 1
                    plan = []  # replan from the new place
                elif 0 < step <= 32:
                    model.step = common if n_common >= 3 else step
                model.body.append((agent.x0, agent.y0))
                # trail detection: the agent's previous cells still carry a non-floor colour after the move
                f2 = nobs.frame[-1]
                left = [int(f2[y][x]) for x in range(agent.x0, agent.x1 + 1) for y in range(agent.y0, agent.y1 + 1)
                        if not (c.x0 <= x <= c.x1 and c.y0 <= y <= c.y1)]
                if left and all(v != tr.bg and model.role.get(v) != "pass" for v in left):
                    model.trail = True
                for col in ahead_cols:
                    model.evidence[(col, "pass")] += 1
                    if model.role.get(col) != "goal":
                        model.role[col] = "pass"
                adx, ady = c.x0 - agent.x0, c.y0 - agent.y0
                ahead_set = set(ahead)
                for o in cur_objs:
                    po = next((q for q in prev_objs if q.tid == o.tid), None)
                    if po is not None and o.tid != agent.tid and (o.x0 - po.x0, o.y0 - po.y0) == (adx, ady):
                        if any((x, y) in ahead_set for x, y in po.cells):
                            pushable.add(okey(po))  # it was ahead of the agent and moved with it: pushed
                            attached.discard(okey(po))
                            self.diag["pushable"] += 1
                        elif okey(po) not in pushable:
                            attached.add(okey(o))
                if target is not None and not plan and press_left == 0 and not pressing_done and not overlap(c.x0, c.y0, c.w, c.h, target):
                    press_left = 3  # arrived next to an impassable target: press into it up to three times
                if target is not None and overlap(c.x0, c.y0, c.w, c.h, target):
                    contacted.add(okey(target))
                    self.history_keys = (getattr(self, "history_keys", []) + [okey(target)])[-4:]
                    # operators: available actions that never moved the agent (e.g. tr87's vertical actions)
                    moving = {act for key, cnt in tr.class_moves.items() for (act, ok), v in cnt.items() if ok and v > 0}
                    dir_avail = [x.value for x in env.action_space if x.value in DIRS]
                    settled = all(tried_actions[d] >= 2 for d in dir_avail)  # every direction tried twice before calling any of them an operator
                    ops = [x.value for x in env.action_space if settled and x.value in (1, 2, 3, 4, 5) and x.value not in moving
                           and (okey(target), x.value) not in op_tried and okey(target) not in events]
                    for x in ops:
                        op_tried.add((okey(target), x))
                    op_left = ops[:3]
                    if self.goal_relation and okey(target)[0] == self.goal_relation[0][0]:
                        flips += 1
                        contacted.discard(okey(target))  # A may be needed again
            obs = nobs
            if obs.state == GameState.WIN:
                break
        self.diag["roles"] = str({k: v for k, v in model.role.items()})
        self.diag["clock_col"] = str(dict(self.clock_col))
        self.diag["unreach_col"] = str(dict(unreach_col))
        self.diag["steps"] = self.steps
        return obs.levels_completed, actions, level_actions, model.role, self.goal_seq, dict(events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("games", nargs="*")
    args = ap.parse_args()
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    games = args.games or MOVE_GAMES
    out = {}
    for g in games:
        t0 = time.perf_counter()
        base = human_baselines(g)
        budget = min(args.budget, 3 * sum(base)) if base else args.budget
        ag = Agent(g, budget, args.debug)
        levels, actions, la, roles, seq, events = ag.run(arc)
        scores = [round(min((base[i] / x) ** 2, 1.15), 3) if base and i < len(base) else None for i, x in enumerate(la)]
        n = len(base) if base else None
        gs = round(sum((s or 0) * (i + 1) for i, s in enumerate(scores)) / sum(range(1, n + 1)), 3) if n else None
        out[g] = {"levels": levels, "of": n, "actions": actions, "level_actions": la, "human": base, "scores": scores, "game_score": gs,
                  "seq": seq, "events": {str(k): v for k, v in events.items()}, "diag": dict(ag.diag), "secs": round(time.perf_counter() - t0, 1)}
        print(f"{g}: levels={levels}/{n} actions={actions}/{budget} per-level={la} human={base[:len(la)] if base else None} scores={scores} "
              f"game_score={gs} seq={seq} events={out[g]['events']} diag={dict(ag.diag)}", flush=True)
    json.dump(out, open("arc3_s4b.json", "w"), indent=1)
    done = [g for g in games if out[g]["levels"] >= 2]
    any1 = [g for g in games if out[g]["levels"] >= 1]
    mean = sum(out[g]["game_score"] or 0 for g in games) / len(games)
    print(f"GATE S4: >=2 levels on {len(done)} {done}; >=1 level on {len(any1)} {any1}; mean game score {mean:.3f} over {len(games)} games")


if __name__ == "__main__":
    main()
