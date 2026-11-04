"""S2c - role-tied, direction-normalised agent-outcome induction with a code-length back-off.

Lesson from an earlier rule-learning experiment (role features generalise where object identities do
not) applied to ARC:
  * the thing ahead of the agent is described by its ROLE (what the agent has experienced against
    that colour so far: pass / block / kill / level / unk), then by pre-contact structure
    (static? size class? border?), and only then by its literal colour;
  * the action is the LAST key, so wall statistics are shared across the four directions;
  * the context tree picks, per observation, the deepest node whose accumulated code length on
    the observations it has seen is no worse than its parent's on the same observations (a
    per-node MDL comparison instead of a fixed support threshold).
Also runs a permutation evaluation: K shuffles of the policy-block order, reporting mean / min.

Outcome classes as in s2b: mv_ok, mv_oth, stay, gone, level.
"""
import json
import math
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcobj import DIRS, Tracker

ACT = {a.value: a for a in GameAction}
WINDOWS = [(0, 10), (10, 50), (50, 200), (200, 10**9)]
CLASSES = ["mv_ok", "mv_oth", "stay", "gone", "level"]


class Node:
    __slots__ = ("counts", "bits", "parent_bits", "n")

    def __init__(self):
        self.counts = Counter()
        self.bits = 0.0
        self.parent_bits = 0.0
        self.n = 0


class MdlTree:
    """Context tree over tuples (general ... specific); prediction uses the deepest node that
    codes its own observations at least as well as its parent did on those same observations."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.nodes = defaultdict(Node)
        self.K = len(CLASSES)

    def p(self, node, y):
        return (node.counts[y] + self.alpha) / (node.n + self.alpha * self.K)

    def chosen(self, ctx):
        best = ()
        for i in range(1, len(ctx) + 1):
            path = ctx[:i]
            nd = self.nodes.get(path)
            if nd is None or nd.n < 2:
                break
            if nd.bits <= nd.parent_bits + 1e-9:
                best = path
            else:
                break
        return best

    def predict(self, ctx):
        path = self.chosen(ctx)
        nd = self.nodes.get(path)  # () is the root node, a valid predictor
        if nd is None or nd.n == 0:
            return "stay", 1.0 / self.K
        y, c = nd.counts.most_common(1)[0]
        return y, self.p(nd, y)

    def observe(self, ctx, y):
        pred, _ = self.predict(ctx)
        parent_p = 1.0 / self.K
        for i in range(0, len(ctx) + 1):
            path = ctx[:i]
            nd = self.nodes[path]
            pp = self.p(nd, y)
            nd.bits += -math.log2(pp)
            nd.parent_bits += -math.log2(parent_p)
            nd.counts[y] += 1
            nd.n += 1
            parent_p = pp
        return pred == y

    def rules(self, min_n=5, min_prob=0.7):
        out = []
        for path, nd in self.nodes.items():
            if nd.n >= min_n and path:
                y, c = nd.counts.most_common(1)[0]
                if c / nd.n >= min_prob and nd.bits <= nd.parent_bits:
                    out.append((path, y, c, nd.n))
        out.sort(key=lambda r: (-r[3], len(r[0])))
        return out


class Mixture:
    """Bayesian mixture of two context trees: weights proportional to 2^-(code length so far)."""

    def __init__(self, trees):
        self.trees = trees
        self.bits = [0.0] * len(trees)

    def dist(self, tree, ctx):
        path = tree.chosen(ctx)
        nd = tree.nodes.get(path)
        if nd is None or nd.n == 0:
            return {y: 1.0 / tree.K for y in CLASSES}
        return {y: tree.p(nd, y) for y in CLASSES}

    def observe(self, ctxs, y):
        dists = [self.dist(t, c) for t, c in zip(self.trees, ctxs)]
        m = min(self.bits)
        w = [2.0 ** (-(b - m)) for b in self.bits]
        tot = sum(w)
        mixed = {c: sum(wi * d[c] for wi, d in zip(w, dists)) / tot for c in CLASSES}
        pred = max(mixed, key=mixed.get)
        for i, d in enumerate(dists):
            self.bits[i] += -math.log2(max(d[y], 1e-9))
        return pred == y


class Roles:
    """Online role of a colour = argmax agent outcome when it was directly ahead."""

    def __init__(self):
        self.tab = defaultdict(Counter)
        self.static = {}  # colour -> ever moved?

    def role(self, colour):
        t = self.tab.get(colour)
        if not t or sum(t.values()) < 2:
            return "unk"
        y = t.most_common(1)[0][0]
        return {"mv_ok": "pass", "mv_oth": "pass", "stay": "block", "gone": "kill", "level": "goal"}[y]

    def update(self, colour, y):
        self.tab[colour][y] += 1


def ahead_cells(agent, action):
    dx, dy = DIRS[action]
    if dx:
        x = agent.x1 + 1 if dx > 0 else agent.x0 - 1
        c1 = [(x, y) for y in range(agent.y0, agent.y1 + 1)]
    else:
        y = agent.y1 + 1 if dy > 0 else agent.y0 - 1
        c1 = [(x, y) for x in range(agent.x0, agent.x1 + 1)]
    c2 = [(x + dx, y + dy) for x, y in c1]
    return c1, c2


def look(cells, frame, bg):
    cols = Counter()
    for x, y in cells:
        if not (0 <= x < 64 and 0 <= y < 64):
            cols["border"] += 1
        else:
            v = int(frame[y][x])
            cols["bg" if v == bg else str(v)] += 1
    nonbg = [(k, v) for k, v in cols.items() if k != "bg"]
    return max(nonbg, key=lambda kv: kv[1])[0] if nonbg else "bg"


def structure(colour, tr, moved_colours):
    if colour in ("bg", "border"):
        return colour
    objs = [o for o in tr.objs if str(o.color) == colour]
    if not objs:
        return "?"
    size = max(o.size for o in objs)
    sz = "s" if size <= 4 else ("m" if size <= 64 else "l")
    mv = "mob" if colour in moved_colours else "static"
    return f"{mv}{sz}"


def outcome(action, prev_objs, cur_objs, prev_obs, cur_obs, tid):
    if cur_obs is None or cur_obs.state == GameState.GAME_OVER:
        return "gone"
    if cur_obs.levels_completed > prev_obs.levels_completed or cur_obs.state == GameState.WIN:
        return "level"
    p = next((o for o in prev_objs if o.tid == tid), None)
    c = next((o for o in cur_objs if o.tid == tid), None)
    if p is None or c is None:
        return "gone"
    dx, dy = c.x0 - p.x0, c.y0 - p.y0
    if (dx, dy) == (0, 0):
        return "stay"
    return "mv_ok" if (np.sign(dx), np.sign(dy)) == DIRS[action] else "mv_oth"


def run_game(arc, game, rows):
    tr = Tracker()
    roles = Roles()
    tree = MdlTree()
    act_only = MdlTree()
    colour_tree = MdlTree()  # the old per-colour key, for comparison
    mix = Mixture([tree, colour_tree])
    moved_colours = set()
    log = []
    for r in rows:
        env = arc.make(game, seed=r["seed"])
        obs = env.reset()
        tr.start(obs.frame[-1])
        for a in r["actions"]:
            data = {"x": a[1], "y": a[2]} if len(a) == 3 and a[1] is not None else {}
            prev_objs = tr.objs
            ag = tr.agent()
            agent_obj = next((o for o in prev_objs if ag and o.tid == ag[0]), None)
            nobs = env.step(ACT[a[0]], data=data)
            cur_objs = tr.step(a[0], nobs.frame[-1]) if nobs is not None else []
            for o in cur_objs:
                po = next((q for q in prev_objs if q.tid == o.tid), None)
                if po is not None and (po.x0, po.y0) != (o.x0, o.y0):
                    moved_colours.add(str(o.color))
            if a[0] in DIRS and agent_obj is not None:
                c1, c2 = ahead_cells(agent_obj, a[0])
                a1 = look(c1, obs.frame[-1], tr.bg)
                a2 = look(c2, obs.frame[-1], tr.bg)
                ctx = (roles.role(a1), structure(a1, tr, moved_colours), a1, a2, a[0])
                y = outcome(a[0], prev_objs, cur_objs, obs, nobs, agent_obj.tid)
                cctx = (a[0], f"{a1}|{a2}", str(agent_obj.color))
                ok_mix = mix.observe([ctx, cctx], y)  # before the trees are updated (prequential)
                ok = tree.observe(ctx, y)
                ok_act = act_only.observe((a[0],), y)
                ok_col = colour_tree.observe(cctx, y)
                roles.update(a1, y)
                log.append((y, ok, ok_act, ok_col, ok_mix))
            if nobs is None:
                break
            obs = nobs
            if obs.state == GameState.GAME_OVER:
                obs = env.reset()
                tr.start(obs.frame[-1])
    return tree, roles, log


def windowed(log, idx):
    out = {}
    for lo, hi in WINDOWS:
        seg = log[lo:hi]
        if seg:
            out[f"{lo}-{min(hi, len(log))}"] = round(sum(x[idx] for x in seg) / len(seg), 3)
    return out


def main():
    t1 = json.load(open("arc3_t1.json", encoding="utf-8"))
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    games = sorted({r["game"] for r in t1["rows"]})
    policies = ["random", "one_probe", "repeat_last_change", "blind_A1", "blind_A2", "blind_A3", "blind_A4", "blind_A5", "click_center", "click_sweep"]
    K = int(os.environ.get('S2C_PERMS', '3'))
    results = {}
    for g in games:
        rows_all = [r for r in t1["rows"] if r["game"] == g and r["steps"] > 0]
        t0 = time.perf_counter()
        per_perm = []
        for k in range(K):
            order = policies[:] if k == 0 else random.Random(k).sample(policies, len(policies))
            rows = sorted(rows_all, key=lambda r: (order.index(r["policy"]) if r["policy"] in order else 99, r["seed"]))
            tree, roles, log = run_game(arc, g, rows)
            per_perm.append((windowed(log, 1), windowed(log, 2), windowed(log, 3), len(log), windowed(log, 4)))
            if k == 0:
                rules = [[list(map(str, p)), y, c, n] for p, y, c, n in tree.rules()[:10]]
                role_map = {c: roles.role(c) for c in roles.tab}
        w = "50-200"
        vals = [p[0].get(w) for p in per_perm if p[0].get(w) is not None]
        vals_act = [p[1].get(w) for p in per_perm if p[1].get(w) is not None]
        vals_col = [p[2].get(w) for p in per_perm if p[2].get(w) is not None]
        vals_mix = [p[4].get(w) for p in per_perm if p[4].get(w) is not None]
        results[g] = {"n_agent_steps": per_perm[0][3], "role_tree_50_200": vals, "act_only_50_200": vals_act, "colour_tree_50_200": vals_col, "mix_50_200": vals_mix,
                      "roles": role_map, "rules": rules, "perm0": {"tree": per_perm[0][0], "act": per_perm[0][1], "colour": per_perm[0][2]},
                      "secs": round(time.perf_counter() - t0, 1)}
        if vals:
            print(f"{g} steps={per_perm[0][3]} w50-200 role_tree mean={np.mean(vals):.3f} min={min(vals):.3f} | act_only {np.mean(vals_act):.3f} | colour_tree {np.mean(vals_col):.3f} | MIX mean={np.mean(vals_mix):.3f} min={min(vals_mix):.3f} | roles={role_map}", flush=True)
        else:
            print(f"{g} steps={per_perm[0][3]} (too few agent steps)", flush=True)
    json.dump(results, open("arc3_s2c.json", "w"), indent=1)
    passed = [g for g in games if results[g]["role_tree_50_200"] and np.mean(results[g]["role_tree_50_200"]) >= 0.8]
    passed_min = [g for g in games if results[g]["role_tree_50_200"] and min(results[g]["role_tree_50_200"]) >= 0.8]
    print(f"GATE role_tree >=0.80 (mean over {K} perms): {len(passed)} {passed}; by min: {len(passed_min)} {passed_min}")
    beat = [g for g in games if results[g]["role_tree_50_200"] and np.mean(results[g]["role_tree_50_200"]) > np.mean(results[g]["colour_tree_50_200"]) + 0.02]
    print(f"role tree beats colour tree by >2pts on: {beat}")
    mixpass = [g for g in games if results[g]["mix_50_200"] and np.mean(results[g]["mix_50_200"]) >= 0.8]
    mixmin = [g for g in games if results[g]["mix_50_200"] and min(results[g]["mix_50_200"]) >= 0.8]
    print(f"GATE MIX >=0.80: by mean {len(mixpass)} {mixpass}; by min {len(mixmin)} {mixmin}")


if __name__ == "__main__":
    main()
