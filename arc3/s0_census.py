"""S0 - behavioural mechanics census. For each game (seed 0, level 1): from the start frame and
from K random reachable frames, branch (deepcopy) and apply each available action (clicks on a
lattice); classify the object-level effect. Output arc3_s0.json + one line per game.

Classes: none | moved(colour,dir) | vanished(colour) | appeared(colour) | recolour(from,to) |
reshape(colour) | global(ncells) | level_up | game_over.
Usage: python s0_census.py [--frames K] [--lattice N] [game ...]
"""
import argparse
import copy
import json
import random
import time
from collections import Counter

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcobj import deltas, segment, summarize

SIMPLE = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5, GameAction.ACTION7]


def classify(obs0, obs1, bg):
    if obs1 is None:
        return Counter({("error",): 1}), 0
    if obs1.state == GameState.GAME_OVER:
        return Counter({("game_over",): 1}), 0
    if obs1.levels_completed > obs0.levels_completed or obs1.state == GameState.WIN:
        return Counter({("level_up",): 1}), 0
    ds, ncells, amb, _ = deltas(obs0.frame[-1], obs1.frame[-1], bg=bg)
    if ncells == 0:
        return Counter({("none",): 1}), 0
    c = summarize(ds)
    if ncells > 400 and len(ds) > 30:
        c = Counter({("global", ncells // 100 * 100): 1})
    return c, ncells


def probe_frame(env, obs, bg, lattice, rng):
    """Apply every available action from this state (branching by deepcopy). Returns {action_label: Counter}."""
    out = {}
    acts = env.action_space
    for a in acts:
        if a in SIMPLE:
            e = copy.deepcopy(env)
            o = e.step(a, data={})
            c, n = classify(obs, o, bg)
            out[a.name] = c
        elif a == GameAction.ACTION6:
            step = 64 // lattice
            agg = Counter()
            per = {}
            for yi in range(lattice):
                for xi in range(lattice):
                    x, y = xi * step + step // 2, yi * step + step // 2
                    e = copy.deepcopy(env)
                    o = e.step(a, data={"x": x, "y": y})
                    c, n = classify(obs, o, bg)
                    key = next(iter(c))[0]
                    agg[key] += 1
                    if key != "none":
                        per[f"{x},{y}"] = [list(map(str, k)) + [v] for k, v in c.items()]
            out["ACTION6"] = agg
            out["ACTION6_cells"] = per
    return out


def random_walk(env, obs, n, rng):
    for _ in range(n):
        acts = env.action_space
        a = rng.choice(list(acts))
        data = {"x": rng.randint(0, 63), "y": rng.randint(0, 63)} if a == GameAction.ACTION6 else {}
        o = env.step(a, data=data)
        if o is None:
            break
        obs = o
        if obs.state == GameState.GAME_OVER:
            obs = env.reset()
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--lattice", type=int, default=8)
    ap.add_argument("games", nargs="*")
    args = ap.parse_args()
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    ids = sorted(e.game_id.split("-")[0] for e in arc.get_environments())
    games = args.games or ids
    census = {}
    for g in games:
        t0 = time.perf_counter()
        rng = random.Random(0)
        env = arc.make(g, seed=0)
        obs = env.reset()
        bg, objs = segment(obs.frame[-1])
        frames = [("start", copy.deepcopy(env), obs)]
        walk_env = copy.deepcopy(env)
        wobs = obs
        for i in range(args.frames):
            wobs = random_walk(walk_env, wobs, 15, rng)
            frames.append((f"walk{i}", copy.deepcopy(walk_env), wobs))
        table = {}
        cells = {}
        for label, e, o in frames:
            res = probe_frame(e, o, bg, args.lattice, rng)
            for act, c in res.items():
                if act == "ACTION6_cells":
                    cells[label] = c
                    continue
                t = table.setdefault(act, Counter())
                for k, v in c.items():
                    t[k] += v
        census[g] = {"bg": bg, "n_objects": len(objs), "colors": sorted({o.color for o in objs}),
                     "actions": [a.name for a in env.action_space], "win_levels": obs.win_levels,
                     "table": {a: [[list(map(str, k)), v] for k, v in t.most_common()] for a, t in table.items()},
                     "click_cells": cells, "secs": round(time.perf_counter() - t0, 1)}
        line = []
        for a, t in table.items():
            top = ", ".join(f"{'/'.join(map(str,k))}×{v}" for k, v in t.most_common(4))
            line.append(f"{a}: {top}")
        print(f"{g} objs={len(objs)} bg={bg} | " + " | ".join(line), flush=True)
    json.dump(census, open("arc3_s0.json", "w"), indent=1)
    print("wrote arc3_s0.json")


if __name__ == "__main__":
    main()
