"""S4 v0 - goal discovery by systematic contact, planning in the induced model.

Simple-action games only (ACTION1-4 move an agent). Loop per game:
  1. segment the frame; find the agent (Tracker); keep a colour -> role table (pass/block/kill/
     goal/unk) from the agent's experience (S2c Roles) plus a per-cell block memory.
  2. candidate targets = objects of colours not yet contacted (or whose role is unknown), nearest
     first by BFS over the cell grid in the INDUCED model (agent moves one cell per action unless
     the observed step size differs; cells of 'block' colours are impassable, 'kill' colours are
     avoided; unknown colours are passable with a cost).
  3. execute the planned path one real action at a time; on a surprise (predicted move, got
     'stay') mark the cell blocked and re-plan; on 'gone' mark the colour 'kill', reset the level.
  4. a level-up records the contacted colour as the GOAL hypothesis; on later levels the planner
     goes straight to that colour (then falls back to contact-everything if it does not work).
Budget per game: 3x the sum of the human baselines, capped at 600 actions. Score per level =
min((human/agent)^2, 1.15) over completed levels (ARC's formula, with the scorecard's baselines).

Usage: python s4_contact.py [--budget N] [game ...]
"""
import argparse
import json
import os
import time
import heapq
from collections import Counter, deque

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcobj import DIRS, Tracker, segment

ACT = {a.value: a for a in GameAction}
SIMPLE_GAMES = ["g50t", "ls20", "re86", "tr87", "tu93", "wa30", "ar25", "bp35", "dc22", "ka59", "sp80", "cn04", "m0r0", "sk48"]


def human_baselines(arc, game):
    try:
        return json.load(open("arc3_t1.json", encoding="utf-8"))["baselines"].get(game)
    except Exception:  # noqa: BLE001
        return None


class Model:
    """Induced passability model: colour roles + remembered blocked cells."""

    def __init__(self, bg):
        self.bg = bg
        self.role = {}  # colour -> pass|block|kill|goal
        self.evidence = Counter()
        self.blocked_cells = set()
        self.blocked_edges = set()  # (x0, y0, action) moves observed to fail
        self.step = 1  # cells per move, observed

    def cell_cost(self, colour):
        """Optimistic costs: a 'block' colour is expensive, not forbidden (one colour can be both
        a wall and a floor); only 'kill' is forbidden. Unknown colours are slightly dear."""
        r = self.role.get(colour)
        if r == "kill":
            return None
        if r in ("pass", "goal"):
            return 1
        if r == "block":
            return 12
        return 2


def bfs_path(frame, model, agent, goal_cells, max_nodes=20000, enter=True):
    """Path of actions from the agent's bbox origin to a position that ENTERS the target (the agent's
    bbox covers the target's centre) or, with enter=False, merely touches it."""
    H, W = 64, 64
    aw, ah = agent.w, agent.h
    start = (agent.x0, agent.y0)
    goal_set = set(goal_cells)
    gxs = [c[0] for c in goal_cells]
    gys = [c[1] for c in goal_cells]
    centre = ((min(gxs) + max(gxs)) // 2, (min(gys) + max(gys)) // 2)

    def occupies(x, y):
        return [(x + i, y + j) for i in range(aw) for j in range(ah)]

    def move_cost(x, y):
        if x < 0 or y < 0 or x + aw > W or y + ah > H:
            return None
        worst = 1
        for cx, cy in occupies(x, y):
            if (cx, cy) in goal_set:
                continue
            c = model.cell_cost(int(frame[cy][cx]))
            if c is None:
                return None
            worst = max(worst, c)
        return worst

    gx0, gx1, gy0, gy1 = min(gxs), max(gxs), min(gys), max(gys)

    def touches(x, y):
        if enter:
            return x <= centre[0] <= x + aw - 1 and y <= centre[1] <= y + ah - 1
        # within one move's reach of the target's bbox (the agent moves `step` cells per action)
        ddx = max(gx0 - (x + aw - 1), x - gx1, 0)
        ddy = max(gy0 - (y + ah - 1), y - gy1, 0)
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
        if cur != start and touches(*cur):
            path = []
            c = cur
            while prev[c] is not None:
                p, a = prev[c]
                path.append(a)
                c = p
            return path[::-1]
        for a, (dx, dy) in DIRS.items():
            if (cur[0], cur[1], a) in model.blocked_edges:
                continue
            nxt = (cur[0] + dx * model.step, cur[1] + dy * model.step)
            mc = move_cost(*nxt)
            if mc is None:
                continue
            nd = d + mc
            if nd < dist.get(nxt, 1e18):
                dist[nxt] = nd
                prev[nxt] = (cur, a)
                heapq.heappush(heap, (nd, nxt))
    return None


def play(arc, game, budget, log):
    env = arc.make(game, seed=0)
    obs = env.reset()
    tr = Tracker()
    tr.start(obs.frame[-1])
    model = Model(tr.bg)
    contacted = set()
    goal_colour = None
    actions = 0
    level_actions = []
    cur_level_actions = 0
    last_level = obs.levels_completed
    rng_actions = [1, 2, 3, 4]
    plan = []
    target = None
    diag = Counter()
    press_left = 0
    pressing = False
    attached = set()  # (colour, shape) of objects that move with the agent: part of the agent
    while actions < budget:
        frame = obs.frame[-1]
        ag = tr.agent()
        agent = next((o for o in tr.objs if ag and o.tid == ag[0]), None)
        if agent is None:
            a = rng_actions[actions % 4]  # discover the agent by moving
            data = {}
            diag["no_agent"] += 1
        else:
            if not plan and target is not None and press_left > 0:
                # arrived next to / on the target: press into it toward its centre until blocked
                tcx, tcy = (target.x0 + target.x1) // 2, (target.y0 + target.y1) // 2
                acx, acy = (agent.x0 + agent.x1) // 2, (agent.y0 + agent.y1) // 2
                ddx, ddy = tcx - acx, tcy - acy
                a_press = (4 if ddx > 0 else 3) if abs(ddx) >= abs(ddy) else (2 if ddy > 0 else 1)
                plan = [a_press]
                press_left -= 1
                pressing = True
            if not plan:
                press_left = 0
                pressing = False
                bg, objs = segment(frame, tr.bg)
                cands = []
                for o in objs:
                    if o.tid == agent.tid or (o.x0 <= agent.x1 and agent.x0 <= o.x1 and o.y0 <= agent.y1 and agent.y0 <= o.y1):
                        continue  # the agent's own body (fresh segmentation carries no identities)
                    if (o.color, o.shape) in attached:
                        continue  # moves with the agent: part of it
                    if model.role.get(o.color) == "kill":
                        continue
                    okey = (o.color, o.x0, o.y0, o.size)
                    pri = 0 if o.color == goal_colour else (1 if okey not in contacted else 2)
                    d = abs(o.x0 - agent.x0) + abs(o.y0 - agent.y0)
                    cands.append((pri, d, o))
                cands.sort(key=lambda t: (t[0], t[1]))
                plan, target = [], None
                for pri, d, o in cands[:12]:
                    p = bfs_path(frame, model, agent, o.cells, enter=True) or bfs_path(frame, model, agent, o.cells, enter=False)
                    if p:
                        plan, target = p, o
                        break
                if not plan:
                    untried = [d for d in (1, 2, 3, 4) if (agent.x0, agent.y0, d) not in model.blocked_edges]
                    a = (untried or [1, 2, 3, 4])[(actions * 7) % len(untried or [1, 2, 3, 4])]
                    plan = [a]
                    target = None
                    diag["no_path"] += 1
                else:
                    diag["plans"] += 1
                    diag["targets_" + str(target.color)] += 1
                    press_left = 4
            a = plan.pop(0)
            data = {}
        prev_objs = tr.objs
        nobs = env.step(ACT[a], data=data)
        actions += 1
        if os.environ.get("S4DEBUG") and actions <= int(os.environ["S4DEBUG"]):
            nag = None
            if nobs is not None and nobs.frame:
                _, tmp = segment(nobs.frame[-1], tr.bg)
            print(f"  t={actions} agent={(agent.x0, agent.y0, agent.w, agent.h, agent.color) if agent else None} a={a} target={(target.color, target.x0, target.y0) if target else None} plan_left={len(plan)} lv={nobs.levels_completed if nobs else None} state={nobs.state.name if nobs else None} roles={ {k: v for k, v in model.role.items() if k != '_diag'} }", flush=True)
        cur_level_actions += 1
        if nobs is None or not nobs.frame:
            if nobs is not None and nobs.state == GameState.GAME_OVER:
                nobs = env.reset()
                tr.start(nobs.frame[-1])
                obs = nobs
                plan = []
                continue
            break
        cur_objs = tr.step(a, nobs.frame[-1])
        if agent is not None:
            c = next((o for o in cur_objs if o.tid == agent.tid), None)
            dx, dy = DIRS[a]
            ex, ey = agent.x0 + dx * model.step, agent.y0 + dy * model.step
            ahead_cells = [(agent.x1 + 1 if dx > 0 else agent.x0 - 1, y) for y in range(agent.y0, agent.y1 + 1)] if dx else \
                          [(x, agent.y1 + 1 if dy > 0 else agent.y0 - 1) for x in range(agent.x0, agent.x1 + 1)]
            ahead_colours = {int(frame[y][x]) for x, y in ahead_cells if 0 <= x < 64 and 0 <= y < 64}
            ahead_nonbg = list(ahead_colours)  # includes the background colour on purpose
            if nobs.levels_completed > last_level:
                if target is not None:
                    goal_colour = target.color
                    model.role[target.color] = "goal"
                level_actions.append(cur_level_actions)
                cur_level_actions = 0
                last_level = nobs.levels_completed
                contacted = set()
                plan = []
                tr.start(nobs.frame[-1])
            elif nobs.state == GameState.GAME_OVER or c is None:
                for col in ahead_nonbg:
                    model.role[col] = "kill"
                plan = []
                nobs = env.reset()
                tr.start(nobs.frame[-1])
                cur_level_actions += 0
            elif (c.x0, c.y0) == (agent.x0, agent.y0) and pressing:
                model.blocked_edges.add((agent.x0, agent.y0, a))
                press_left = 0
                pressing = False
                plan = []
                if target is not None:
                    contacted.add((target.color, target.x0, target.y0, target.size))
            elif (c.x0, c.y0) == (agent.x0, agent.y0):
                for col in ahead_nonbg:
                    model.evidence[(col, "stay")] += 1
                    if model.evidence[(col, "stay")] >= 2 and model.evidence[(col, "pass")] == 0:
                        model.role[col] = "block"
                    elif model.evidence[(col, "stay")] >= 6 * max(1, model.evidence[(col, "pass")]):
                        model.role[col] = "block"
                model.blocked_edges.add((agent.x0, agent.y0, a))
                plan = []
            else:
                step = max(abs(c.x0 - agent.x0), abs(c.y0 - agent.y0))
                if step > 0:
                    model.step = step
                adx, ady = c.x0 - agent.x0, c.y0 - agent.y0
                for o in cur_objs:
                    po = next((q for q in prev_objs if q.tid == o.tid), None)
                    if po is not None and o.tid != agent.tid and (o.x0 - po.x0, o.y0 - po.y0) == (adx, ady):
                        attached.add((o.color, o.shape))
                for col in ahead_nonbg:
                    model.evidence[(col, "pass")] += 1
                    if model.role.get(col) not in ("goal",):
                        model.role[col] = "pass"
                if target is not None and not plan and press_left == 0:
                    contacted.add((target.color, target.x0, target.y0, target.size))
        obs = nobs
        if obs.state == GameState.WIN:
            break
    model.role["_diag"] = dict(diag)
    return obs.levels_completed, actions, level_actions, model.role, goal_colour


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("games", nargs="*")
    args = ap.parse_args()
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    games = args.games or SIMPLE_GAMES
    out = {}
    for g in games:
        t0 = time.perf_counter()
        base = human_baselines(arc, g)
        budget = min(args.budget, 3 * sum(base)) if base else args.budget
        levels, actions, level_actions, roles, goal = play(arc, g, budget, out)
        scores = []
        for i, la in enumerate(level_actions):
            hb = base[i] if base and i < len(base) else None
            scores.append(round(min((hb / la) ** 2, 1.15), 3) if hb and la else None)
        n_levels = len(base) if base else None
        game_score = (sum(s * (i + 1) for i, s in enumerate(scores) if s) / sum(range(1, n_levels + 1))) if n_levels else None
        out[g] = {"levels": levels, "of": n_levels, "actions": actions, "budget": budget, "level_actions": level_actions,
                  "human": base, "level_scores": scores, "game_score": round(game_score, 3) if game_score is not None else None,
                  "roles": {str(k): v for k, v in roles.items()}, "goal_colour": goal, "secs": round(time.perf_counter() - t0, 1)}
        print(f"{g}: levels={levels}/{n_levels} actions={actions}/{budget} per-level={level_actions} human={base[:len(level_actions)] if base else None} "
              f"scores={scores} game_score={out[g]['game_score']} goal={goal} roles={out[g]['roles']}", flush=True)
    json.dump(out, open("arc3_s4.json", "w"), indent=1)
    done = [g for g in games if out[g]["levels"] >= 2]
    mean = sum(out[g]["game_score"] or 0 for g in games) / len(games)
    print(f"GATE S4: >=2 levels on {len(done)} games {done}; mean game score over these {len(games)} games = {mean:.3f} (ARC mean over 25 would be {mean*len(games)/25:.3f})")


if __name__ == "__main__":
    main()
