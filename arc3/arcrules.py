"""S2 - prequential MDL rule induction over an object-centric context tree.

The hypothesis space is a context tree: (action, context_1, context_2, ...) -> distribution over
EFFECT SIGNATURES. A signature is the sorted tuple of summarized object deltas for one step
(kind, colour, sign dx, sign dy) plus special tokens (level_up, game_over, none). Prediction backs
off from the most specific context with enough support to the shallowest (PPM-style, Laplace
estimator); the prequential code length of the stream under the tree IS the MDL score, so model
selection is automatic (Rissanen): a context specialisation is used only when its predictions
beat the parent's online.

Contexts (built from arcobj objects; agent from the Tracker):
  c1 = action id (1-7)
  c2 = for simple actions: what is directly ahead of the agent in the action's direction
       ("bg" | "border" | colour) ; for clicks: the colour under the click (or "bg")
  c3 = for simple actions: the agent's colour; for clicks: the size class of the clicked object
Rules are read out as (context path) -> argmax signature with support and probability.
"""
import math
from collections import Counter, defaultdict

from arcobj import DIRS, deltas, segment, summarize

LEVEL_UP = ("level_up",)
GAME_OVER = ("game_over",)
NONE = ("none",)


def signature(ds, ncells, obs_prev, obs_cur):
    if obs_cur is None:
        return (("error",),)
    if obs_cur.state.name == "GAME_OVER":
        return (GAME_OVER,)
    if obs_cur.levels_completed > obs_prev.levels_completed or obs_cur.state.name == "WIN":
        return (LEVEL_UP,)
    if ncells == 0:
        return (NONE,)
    c = summarize(ds)
    if ncells > 400 and len(ds) > 30:
        return (("global",),)
    return tuple(sorted((tuple(map(str, k)) for k in c.keys())))


def ahead(agent, action, frame, bg):
    """What is directly ahead of the agent's bbox in the action's direction."""
    if agent is None or action not in DIRS:
        return "noagent"
    dx, dy = DIRS[action]
    cells = []
    if dx:
        x = agent.x1 + 1 if dx > 0 else agent.x0 - 1
        cells = [(x, y) for y in range(agent.y0, agent.y1 + 1)]
    else:
        y = agent.y1 + 1 if dy > 0 else agent.y0 - 1
        cells = [(x, y) for x in range(agent.x0, agent.x1 + 1)]
    def look(cells):
        cols = Counter()
        for x, y in cells:
            if not (0 <= x < 64 and 0 <= y < 64):
                cols["border"] += 1
            else:
                v = int(frame[y][x])
                cols["bg" if v == bg else str(v)] += 1
        nonbg = [(k, v) for k, v in cols.items() if k != "bg"]
        return max(nonbg, key=lambda kv: kv[1])[0] if nonbg else "bg"
    a1 = look(cells)
    a2 = look([(x + dx, y + dy) for x, y in cells])  # two cells ahead
    return f"{a1}|{a2}"


def click_context(x, y, frame, bg, objs):
    v = int(frame[y][x])
    if v == bg:
        return "bg", "bg"
    for o in objs:
        if (x, y) in o.cells:
            sz = "s" if o.size <= 4 else ("m" if o.size <= 64 else "l")
            return str(v), sz
    return str(v), "?"


class ContextTree:
    def __init__(self, alpha=0.5, min_support=2):
        self.alpha = alpha
        self.min_support = min_support
        self.tables = defaultdict(Counter)  # context path (tuple) -> Counter(signature)
        self.alphabet = set()
        self.codelen = 0.0
        self.n = 0
        self.hits = 0

    def paths(self, ctx):
        return [ctx[:i] for i in range(len(ctx), 0, -1)]  # most specific first

    def predict(self, ctx):
        """Back-off prediction: (signature, prob, path) from the most specific context with support."""
        for p in self.paths(ctx):
            t = self.tables.get(p)
            if t and sum(t.values()) >= self.min_support:
                tot = sum(t.values())
                sig, cnt = t.most_common(1)[0]
                return sig, (cnt + self.alpha) / (tot + self.alpha * max(len(self.alphabet), 1)), p
        if self.alphabet:
            return NONE_SIG, 1.0 / len(self.alphabet), ()
        return NONE_SIG, 1.0, ()

    def prob(self, ctx, sig):
        for p in self.paths(ctx):
            t = self.tables.get(p)
            if t and sum(t.values()) >= self.min_support:
                tot = sum(t.values())
                return (t[sig] + self.alpha) / (tot + self.alpha * max(len(self.alphabet), 1))
        return 1.0 / max(len(self.alphabet), 1)

    def observe(self, ctx, sig):
        """Prequential: score the prediction, then update every prefix table."""
        pred, _, _ = self.predict(ctx)
        self.n += 1
        self.hits += int(pred == sig)
        self.codelen += -math.log2(max(self.prob(ctx, sig), 1e-9))
        self.alphabet.add(sig)
        for p in self.paths(ctx):
            self.tables[p][sig] += 1
        return pred == sig

    def rules(self, min_support=3, min_prob=0.6):
        out = []
        for p, t in self.tables.items():
            tot = sum(t.values())
            if tot < min_support:
                continue
            sig, cnt = t.most_common(1)[0]
            pr = cnt / tot
            if pr >= min_prob:
                out.append((p, sig, cnt, tot, round(pr, 2)))
        out.sort(key=lambda r: (-r[3], len(r[0])))
        return out


NONE_SIG = (NONE,)


class Inducer:
    """Online: feed (action, data, prev_obs, cur_obs); keeps objects, agent, and the tree."""

    def __init__(self, tracker):
        self.tr = tracker
        self.tree = ContextTree()
        self.history = []  # (ctx, sig, predicted_ok)

    def context(self, action, data, prev_obs):
        frame = prev_obs.frame[-1]
        bg = self.tr.bg
        ag = self.tr.agent()
        agent_obj = None
        if ag:
            for o in self.tr.objs:
                if o.tid == ag[0]:
                    agent_obj = o
        if action == 6 and data:
            c2, c3 = click_context(data["x"], data["y"], frame, bg, self.tr.objs)
            return (action, c2, c3)
        c2 = ahead(agent_obj, action, frame, bg)
        c3 = str(agent_obj.color) if agent_obj else "noagent"
        return (action, c2, c3)

    def step(self, action, data, prev_obs, cur_obs):
        ctx = self.context(action, data, prev_obs)
        prev_objs = self.tr.objs
        if cur_obs is None:
            sig = (("error",),)
            ok = self.tree.observe(ctx, sig)
            self.history.append((ctx, sig, ok))
            return ok, sig
        ds, ncells, amb, cur_objs = deltas(prev_obs.frame[-1], cur_obs.frame[-1], prev_objs=prev_objs, bg=self.tr.bg)
        sig = signature(ds, ncells, prev_obs, cur_obs)
        ok = self.tree.observe(ctx, sig)
        self.history.append((ctx, sig, ok))
        self.tr.step(action, cur_obs.frame[-1])
        if cur_obs.state.name == "GAME_OVER":
            pass  # caller resets and calls tracker.start
        return ok, sig
