"""S1 gate: replay T1 streams (seed 0, random + one_probe + repeat_last_change) through the
Tracker; report tracking consistency and whether the agent object is found on simple-action games."""
import json
import time

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcobj import Tracker, segment

ACT = {a.value: a for a in GameAction}


def replay(arc, game, seed, actions, tracker):
    env = arc.make(game, seed=seed)
    obs = env.reset()
    tracker.start(obs.frame[-1])
    for a in actions:
        data = {"x": a[1], "y": a[2]} if len(a) == 3 and a[1] is not None else {}
        nobs = env.step(ACT[a[0]], data=data)
        if nobs is None:
            break
        tracker.step(a[0], nobs.frame[-1])
        if nobs.state == GameState.GAME_OVER:
            nobs = env.reset()
            tracker.start(nobs.frame[-1])
    return env


def main():
    t1 = json.load(open("arc3_t1.json", encoding="utf-8"))
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    games = sorted({r["game"] for r in t1["rows"]})
    out = {}
    for g in games:
        rows = [r for r in t1["rows"] if r["game"] == g and r["seed"] == 0 and r["policy"] in ("random", "one_probe", "repeat_last_change")]
        simple = not any(len(a) == 3 for r in rows for a in r["actions"])
        tr = Tracker()
        t0 = time.perf_counter()
        for r in rows:
            replay(arc, g, 0, r["actions"], tr)
        ag = tr.agent()
        env = arc.make(g, seed=0)
        obs = env.reset()
        bg, objs = segment(obs.frame[-1])
        out[g] = {"simple": simple, "transitions": tr.transitions, "consistency": round(tr.consistency(), 4),
                  "agent": (ag[0], ag[2], ag[3], tr.colors.get(ag[0])) if ag else None,
                  "n_objects_start": len(objs), "bg": bg, "secs": round(time.perf_counter() - t0, 1)}
        print(f"{g} simple={simple} T={tr.transitions} consistency={tr.consistency():.3f} objs0={len(objs)} bg={bg} agent={out[g]['agent']}", flush=True)
    json.dump(out, open("arc3_s1.json", "w"), indent=1)
    simple = [g for g in games if out[g]["simple"]]
    found = [g for g in simple if out[g]["agent"]]
    cons = [out[g]["consistency"] for g in simple]
    print(f"simple-action games {len(simple)}: agent found on {len(found)} {found}; mean consistency {sum(cons)/len(cons):.3f}")


if __name__ == "__main__":
    main()
