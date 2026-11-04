"""S2b gate (agent-centric): on every simple-action step (ACTION1-4) with a known agent, predict the
agent's outcome from (action, what-is-ahead, agent colour) with the prequential context tree, vs an
action-only table and a majority baseline. Outcome classes:
  mv_ok   agent moved in the action's direction
  mv_oth  agent moved some other way
  stay    agent did not move (blocked or no-op)
  gone    agent vanished / game over
  level   level up
Stream order: random first, then the rest, so early windows are informative. Also reports the
'any change' prediction (did the frame change at all) for all actions incl. clicks.
"""
import json
import time
from collections import Counter

import numpy as np
from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcobj import DIRS, Tracker, match, segment
from arcrules import ContextTree, ahead

ACT = {a.value: a for a in GameAction}
WINDOWS = [(0, 10), (10, 50), (50, 200), (200, 10**9)]


def agent_outcome(tr, action, prev_objs, cur_objs, prev_obs, cur_obs, agent_tid):
    if cur_obs is None or cur_obs.state == GameState.GAME_OVER:
        return "gone"
    if cur_obs.levels_completed > prev_obs.levels_completed or cur_obs.state == GameState.WIN:
        return "level"
    p = next((o for o in prev_objs if o.tid == agent_tid), None)
    c = next((o for o in cur_objs if o.tid == agent_tid), None)
    if p is None:
        return "noagent"
    if c is None:
        return "gone"
    dx, dy = c.x0 - p.x0, c.y0 - p.y0
    if (dx, dy) == (0, 0):
        return "stay"
    if (np.sign(dx), np.sign(dy)) == DIRS[action]:
        return "mv_ok"
    return "mv_oth"


def run_game(arc, game, rows):
    tr = Tracker()
    tree = ContextTree()
    act_only = ContextTree()
    maj = Counter()
    change_tree = ContextTree()
    change_act = ContextTree()
    log = []
    clog = []
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
            # any-change target for every action
            changed = "chg" if (nobs is not None and (nobs.state == GameState.GAME_OVER or nobs.levels_completed != obs.levels_completed or np.any(np.asarray(nobs.frame[-1]) != np.asarray(obs.frame[-1])))) else "same"
            cctx = (a[0], ahead(agent_obj, a[0], obs.frame[-1], tr.bg)) if a[0] in DIRS else ((a[0], f"{data.get('x',0)//8},{data.get('y',0)//8}") if a[0] == 6 else (a[0],))
            clog.append((changed, change_tree.observe(cctx, changed), change_act.observe((a[0],), changed)))
            if a[0] in DIRS and agent_obj is not None:
                ctx = (a[0], ahead(agent_obj, a[0], obs.frame[-1], tr.bg), str(agent_obj.color))
                y = agent_outcome(tr, a[0], prev_objs, cur_objs, obs, nobs, agent_obj.tid)
                pred_m = maj.most_common(1)[0][0] if maj else "stay"
                ok = tree.observe(ctx, y)
                ok1 = act_only.observe((a[0],), y)
                maj[y] += 1
                log.append((y, ok, ok1, pred_m == y))
            if nobs is None:
                break
            obs = nobs
            if obs.state == GameState.GAME_OVER:
                obs = env.reset()
                tr.start(obs.frame[-1])
    return tree, log, clog


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
    order = ["random", "one_probe", "repeat_last_change", "blind_A1", "blind_A2", "blind_A3", "blind_A4", "blind_A5", "click_center", "click_sweep"]
    results = {}
    for g in games:
        rows = [r for r in t1["rows"] if r["game"] == g and r["steps"] > 0]
        rows.sort(key=lambda r: (order.index(r["policy"]) if r["policy"] in order else 99, r["seed"]))
        t0 = time.perf_counter()
        tree, log, clog = run_game(arc, g, rows)
        dist = Counter(x[0] for x in log)
        res = {"n_agent_steps": len(log), "outcomes": dict(dist),
               "tree": windowed(log, 1), "act_only": windowed(log, 2), "majority": windowed(log, 3),
               "change_tree": windowed(clog, 1), "change_act": windowed(clog, 2), "n_steps": len(clog),
               "rules": [[list(map(str, p)), s, c, t, pr] for p, s, c, t, pr in tree.rules()[:10]], "secs": round(time.perf_counter() - t0, 1)}
        results[g] = res
        w = "50-200" if "50-200" in res["tree"] else ("10-50" if "10-50" in res["tree"] else None)
        if w:
            print(f"{g} agent_steps={len(log)} {dict(dist)} | w{w}: tree={res['tree'][w]} act={res['act_only'][w]} maj={res['majority'][w]} | change w{w}: tree={res['change_tree'].get(w)} act={res['change_act'].get(w)}", flush=True)
        else:
            print(f"{g} agent_steps={len(log)} (no agent on simple actions) | change: {res['change_tree']}", flush=True)
    json.dump(results, open("arc3_s2b.json", "w"), indent=1)
    g80 = [g for g in games if results[g]["tree"].get("50-200", 0) >= 0.8]
    print(f"GATE (agent outcome, window 50-200, >=0.80): {len(g80)} games {g80}")
    better = [g for g in games if results[g]["tree"].get("50-200", 0) > results[g]["act_only"].get("50-200", 0) + 0.02]
    print(f"context beats action-only by >2pts on: {better}")


if __name__ == "__main__":
    main()
