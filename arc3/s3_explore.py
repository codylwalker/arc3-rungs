"""S3 - exploration as rule disambiguation, in the real environment.

At each step the agent scores every available action by the inducer's own uncertainty about its
outcome in the current context (1 - p_max under the chosen MDL node, 1.0 for an unseen
context) plus a novelty bonus for (agent position, action) pairs never tried, and takes the
argmax (ties broken by a seeded RNG). Clicks are scored on a coarse lattice of candidate cells
the same way. Compared against the T1 random stream on the same games by the prequential
accuracy of a FRESH inducer trained on each stream (window 50-200), and by the number of real
actions until the inducer first reaches 0.80 accuracy over a trailing window of 40.

Usage: python s3_explore.py [--actions N] [game ...]   (default: the 11 games that passed S2)
"""
import argparse
import json
import random
import time
from collections import Counter

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcobj import DIRS, Tracker
from s2c_roles import MdlTree, Roles, ahead_cells, look, outcome, structure

ACT = {a.value: a for a in GameAction}
PASSED = ["ar25", "bp35", "dc22", "g50t", "ka59", "ls20", "re86", "sp80", "tr87", "tu93", "wa30"]


class Explorer:
    def __init__(self, seed):
        self.tr = Tracker()
        self.roles = Roles()
        self.tree = MdlTree()
        self.moved = set()
        self.tried = Counter()
        self.rng = random.Random(seed)
        self.log = []

    def context(self, agent_obj, action, frame):
        c1, c2 = ahead_cells(agent_obj, action)
        a1 = look(c1, frame, self.tr.bg)
        a2 = look(c2, frame, self.tr.bg)
        return (self.roles.role(a1), structure(a1, self.tr, self.moved), a1, a2, action), a1

    def score(self, ctx, key):
        y, p = self.tree.predict(ctx)
        nd = self.tree.nodes.get(self.tree.chosen(ctx))
        unc = 1.0 if nd is None or nd.n == 0 else 1.0 - p
        nov = 0.5 if self.tried[key] == 0 else 0.0
        return unc + nov + self.rng.random() * 1e-3

    def choose(self, env, obs):
        acts = env.action_space
        frame = obs.frame[-1]
        ag = self.tr.agent()
        agent_obj = next((o for o in self.tr.objs if ag and o.tid == ag[0]), None)
        best = None
        for a in acts:
            if a == GameAction.ACTION6:
                for yy in range(8, 64, 16):
                    for xx in range(8, 64, 16):
                        key = ("c", xx // 16, yy // 16)
                        s = (0.5 if self.tried[key] == 0 else 0.0) + self.rng.random() * 1e-3
                        if best is None or s > best[0]:
                            best = (s, a, {"x": xx, "y": yy}, key, None, None)
                continue
            if a.value in DIRS and agent_obj is not None:
                ctx, a1 = self.context(agent_obj, a.value, frame)
                key = ("m", agent_obj.x0, agent_obj.y0, a.value)
                s = self.score(ctx, key)
                if best is None or s > best[0]:
                    best = (s, a, {}, key, ctx, a1)
            else:
                key = ("s", a.value)
                s = (0.5 if self.tried[key] == 0 else 0.0) + self.rng.random() * 1e-3
                if best is None or s > best[0]:
                    best = (s, a, {}, key, None, None)
        return best, agent_obj

    def run(self, arc, game, n):
        env = arc.make(game, seed=0)
        obs = env.reset()
        self.tr.start(obs.frame[-1])
        levels = 0
        for i in range(n):
            best, agent_obj = self.choose(env, obs)
            if best is None:
                break
            _, a, data, key, ctx, a1 = best
            self.tried[key] += 1
            prev_objs = self.tr.objs
            nobs = env.step(a, data=data)
            cur_objs = self.tr.step(a.value, nobs.frame[-1]) if nobs is not None else []
            for o in cur_objs:
                po = next((q for q in prev_objs if q.tid == o.tid), None)
                if po is not None and (po.x0, po.y0) != (o.x0, o.y0):
                    self.moved.add(str(o.color))
            if ctx is not None and agent_obj is not None:
                y = outcome(a.value, prev_objs, cur_objs, obs, nobs, agent_obj.tid)
                ok = self.tree.observe(ctx, y)
                self.roles.update(a1, y)
                self.log.append((y, ok))
            if nobs is None:
                break
            obs = nobs
            levels = max(levels, obs.levels_completed)
            if obs.state == GameState.WIN:
                break
            if obs.state == GameState.GAME_OVER:
                obs = env.reset()
                self.tr.start(obs.frame[-1])
        return levels


def fresh_accuracy(arc, game, actions):
    """Prequential accuracy of a fresh inducer on a replayed action stream."""
    ex = Explorer(0)
    env = arc.make(game, seed=0)
    obs = env.reset()
    ex.tr.start(obs.frame[-1])
    for a in actions:
        data = {"x": a[1], "y": a[2]} if len(a) == 3 and a[1] is not None else {}
        prev_objs = ex.tr.objs
        ag = ex.tr.agent()
        agent_obj = next((o for o in prev_objs if ag and o.tid == ag[0]), None)
        nobs = env.step(ACT[a[0]], data=data)
        cur_objs = ex.tr.step(a[0], nobs.frame[-1]) if nobs is not None else []
        for o in cur_objs:
            po = next((q for q in prev_objs if q.tid == o.tid), None)
            if po is not None and (po.x0, po.y0) != (o.x0, o.y0):
                ex.moved.add(str(o.color))
        if a[0] in DIRS and agent_obj is not None:
            ctx, a1 = ex.context(agent_obj, a[0], obs.frame[-1])
            y = outcome(a[0], prev_objs, cur_objs, obs, nobs, agent_obj.tid)
            ex.log.append((y, ex.tree.observe(ctx, y)))
            ex.roles.update(a1, y)
        if nobs is None:
            break
        obs = nobs
        if obs.state == GameState.GAME_OVER:
            obs = env.reset()
            ex.tr.start(obs.frame[-1])
    return ex.log


def stats(log):
    w = log[50:200]
    acc = round(sum(ok for _, ok in w) / len(w), 3) if w else None
    first80 = None
    for i in range(40, len(log) + 1):
        if sum(ok for _, ok in log[i - 40:i]) / 40 >= 0.8:
            first80 = i
            break
    return acc, first80, len(log), dict(Counter(y for y, _ in log))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=300)
    ap.add_argument("games", nargs="*")
    args = ap.parse_args()
    t1 = json.load(open("arc3_t1.json", encoding="utf-8"))
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    games = args.games or PASSED
    out = {}
    for g in games:
        t0 = time.perf_counter()
        ex = Explorer(0)
        levels = ex.run(arc, g, args.actions)
        e_acc, e_first, e_n, e_dist = stats(ex.log)
        rnd = next(r for r in t1["rows"] if r["game"] == g and r["policy"] == "random" and r["seed"] == 0)
        rlog = fresh_accuracy(arc, g, rnd["actions"][: args.actions])
        r_acc, r_first, r_n, r_dist = stats(rlog)
        out[g] = {"explore": {"acc50_200": e_acc, "first80": e_first, "agent_steps": e_n, "outcomes": e_dist, "levels": levels, "roles": {c: ex.roles.role(c) for c in ex.roles.tab}},
                  "random": {"acc50_200": r_acc, "first80": r_first, "agent_steps": r_n, "outcomes": r_dist, "levels": rnd["levels"]}, "secs": round(time.perf_counter() - t0, 1)}
        print(f"{g}: explore acc={e_acc} first80@{e_first} steps={e_n} levels={levels} {e_dist} | random acc={r_acc} first80@{r_first} steps={r_n} levels={rnd['levels']}", flush=True)
    json.dump(out, open("arc3_s3.json", "w"), indent=1)
    better = [g for g in games if out[g]["explore"]["first80"] and (out[g]["random"]["first80"] is None or out[g]["explore"]["first80"] * 2 <= out[g]["random"]["first80"])]
    print(f"GATE (explore reaches 0.80 in <= half the actions of random): {len(better)}/{len(games)} {better}")


if __name__ == "__main__":
    main()
