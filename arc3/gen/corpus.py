"""Generate, solve and emit a set of games.

Usage: python corpus.py --out DIR --n 20 [--seed 0] [--k 2] [--must hazard,keydoor] [--max-nodes N] [--max-secs S]
                        [--goal keys|plates [--pair]]
Each kept game: <out>/<id>/<hash>/{<id>.py, metadata.json, cert.json}; <out>/manifest.json lists them.
A seed whose plan the oracle cannot solve (any level) is skipped and the next seed tried.

--goal makes goal-kind games (compose.compose_game; runtime_move GOAL_ITEMS): no G tile, the level completes when
its keys are collected or its plates covered. --pair emits each certified plan TWICE under two game ids, the goal
signalled on the HUD (`signalled`) and shown only by its consequence (`consequence`): one plan, one certification,
the same paths replayed on both, so the variants differ in what the HUD shows and nothing else (the goal-rung set,
2026-10-30). Their cert.json and metadata.json carry goal, variant and pair_seed; cert.json also names its pair.
"""
import argparse
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import certify  # noqa: E402
import compose  # noqa: E402
import emit  # noqa: E402
import runtime_click  # noqa: E402
import runtime_move  # noqa: E402


def load_emitted(path, cls):
    """Exec an emitted game file the way arc_agi's local wrapper does; returns the game class."""
    ns = {"__name__": "arc_agi_3.gen"}
    with open(path, encoding="utf-8") as f:
        exec(compile(f.read(), path, "exec"), ns)
    return ns[cls]


def pair_id(seed, gid, skip_ids=None):
    """A second game id for a pair: seeded, never the first's, never one the set holds."""
    r = compose.random.Random(seed * 104729 + 3)
    for _ in range(500):
        g2 = compose.game_id(r)
        if g2 != gid and not (skip_ids and g2 in skip_ids):
            return g2
    raise RuntimeError("no free game id for the pair of seed %d" % seed)


def make_one(seed, out, k=None, must=None, max_nodes=50000, max_secs=120.0, n_levels=6, tries=12, family="move",
             kind=None, skip_ids=None, goal=None, pair=False):
    if family == "click":
        plan = compose.compose_click_game(seed, kind=kind, n_levels=n_levels)
        Game, spec_fn = runtime_click.GenClickGame, lambda rng, mech, cfg, i, n: compose.click_level_spec(rng, cfg, i, n)
        cell = runtime_click.CELL
    else:
        plan = compose.compose_game(seed, k=k, must=must, n_levels=n_levels, goal=goal)
        Game, spec_fn, cell = runtime_move.GenGame, compose.level_spec, runtime_move.CELL
    gid = plan["game_id"]
    if skip_ids and gid in skip_ids:  # two games under one id would make arc_agi's lookup ambiguous
        return None, {"seed": seed, "game_id": gid, "mechanics": plan["config"]["mechanics"], "fail": [], "level": 0,
                      "collision": True}
    t0 = time.perf_counter()
    # level by level: each level gets up to `tries` grids; the first the oracle solves is kept if its optimal
    # is >= 3 (click games: >= 2 on level 1)
    rng = compose.random.Random(seed * 7919 + 17)
    mech = plan["config"]["mechanics"]
    kept_specs, levels = [], []
    for i in range(n_levels):
        found = None
        for t in range(tries):
            spec = dict(plan["levels"][i]) if t == 0 else spec_fn(rng, mech, plan["config"], i, n_levels)
            trial = kept_specs + [spec]
            game = Game(gid, plan["config"], trial)
            path, nodes, secs = certify.solve(game, i, max_nodes, max_secs)
            # a click level the oracle clears in one click is cleared by random clicking too (T1 smoke 10-27)
            if path and len(path) >= (3 if family == "move" or i > 0 else 2):
                found = (spec, {"level": i + 1, "optimal": len(path), "nodes": nodes, "secs": round(secs, 2),
                                "path": path, "tries": t + 1})
                break
        if found is None:
            return None, {"seed": seed, "game_id": gid, "mechanics": mech, "fail": levels, "level": i + 1}
        kept_specs.append(found[0])
        levels.append(found[1])
    plan["levels"] = kept_specs
    base = certify.baselines(levels)
    for spec, b in zip(plan["levels"], certify.budgets(levels)):
        spec["budget"] = b
        w = len(spec["rows"][0]) if "rows" in spec else len(spec["target"][0])
        spec["budget_unit"] = max(1, -(-b // max(4, w * cell - 8)))
    if pair and not goal:
        raise ValueError("--pair needs --goal: the variants differ only in how a goal kind is shown")
    variants = [("consequence", False, gid), ("signalled", True, pair_id(seed, gid, skip_ids))] if pair \
        else [(None, None, gid)]
    certs, dirs = [], []
    for variant, show, vid in variants:
        p = copy.deepcopy(plan)
        p["game_id"] = vid
        if show is not None:
            p["config"]["show_goal"] = show
        extra = {k2: v for k2, v in (("goal", goal), ("variant", variant), ("pair_seed", seed if pair else None))
                 if v is not None}
        d, full_id = emit.emit_game(p, out, baselines=base, family=family, extra=extra or None)
        cls = load_emitted(os.path.join(d, "%s.py" % vid), emit.class_name(vid))
        ok = certify.replay(cls(), [r["path"] for r in levels])
        cert = {"game_id": full_id, "seed": seed, "mechanics": p["config"]["mechanics"], "levels": levels,
                "baseline_actions": base, "replay_ok": ok, "secs": round(time.perf_counter() - t0, 1)}
        cert.update(extra)
        certs.append(cert)
        dirs.append(d)
    if pair:
        certs[0]["pair"], certs[1]["pair"] = certs[1]["game_id"], certs[0]["game_id"]
    for d, cert in zip(dirs, certs):
        with open(os.path.join(d, "cert.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(cert, f, indent=1)
    return (certs if pair else certs[0]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--must", default="")
    ap.add_argument("--max-nodes", type=int, default=50000)
    ap.add_argument("--max-secs", type=float, default=120.0)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--family", default="move", choices=["move", "click"])
    ap.add_argument("--kind", default=None, help="click family: toggle | move")
    ap.add_argument("--append", action="store_true", help="add to an existing set: keep its manifest, skip its ids")
    ap.add_argument("--budget-secs", type=float, default=0.0, help="start no new seed after this many seconds")
    ap.add_argument("--goal", default=None, choices=sorted(compose.GOAL_MECH), help="goal-kind games: no G tile")
    ap.add_argument("--pair", action="store_true", help="with --goal: emit each plan as a signalled/consequence pair")
    args = ap.parse_args()
    if args.pair and not args.goal:
        ap.error("--pair needs --goal")
    must = [m for m in args.must.split(",") if m]
    man_path = os.path.join(args.out, "manifest.json")
    old = {}
    if args.append and os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            old = json.load(f)
    runs = old.get("runs") or ([old["args"]] if "args" in old else [])
    skip_ids = set()
    if args.append and os.path.isdir(args.out):
        skip_ids = {d for d in os.listdir(args.out) if os.path.isdir(os.path.join(args.out, d))}
    kept, failed, seed = list(old.get("games", [])), list(old.get("failed", [])), args.seed
    new, t0 = 0, time.perf_counter()
    while new < args.n:
        if args.budget_secs and time.perf_counter() - t0 > args.budget_secs:
            print("budget of %.0f s reached after %d new games; next seed %d" % (args.budget_secs, new, seed))
            break
        cert, fail = make_one(seed, args.out, args.k, must, args.max_nodes, args.max_secs, args.levels,
                              family=args.family, kind=args.kind, skip_ids=skip_ids, goal=args.goal, pair=args.pair)
        if cert:
            for c in (cert if isinstance(cert, list) else [cert]):
                kept.append(c)
                skip_ids.add(c["game_id"].split("-")[0])
                print("%s seed=%d %s%s optimal=%s replay=%s %.1fs" % (
                    c["game_id"], seed, ",".join(c["mechanics"]), " %s/%s" % (c["goal"], c["variant"]) if c.get("variant") else "",
                    [r["optimal"] for r in c["levels"]], c["replay_ok"], c["secs"]), flush=True)
            new += 1
        else:
            failed.append(fail)
            done = "id collision" if fail.get("collision") else [r["optimal"] for r in fail["fail"]]
            print("skip seed=%d %s %s" % (seed, ",".join(fail["mechanics"]), done), flush=True)
        seed += 1
        # written after every seed, so a run that is killed keeps what it made
        man = {"games": kept, "failed": failed, "args": vars(args), "runs": runs + [vars(args)]}
        with open(man_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(man, f, indent=1)
    print("kept %d new (%d in the set), skipped %d in all, manifest %s" % (new, len(kept), len(failed), man_path))


if __name__ == "__main__":
    main()
