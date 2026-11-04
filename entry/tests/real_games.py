"""twin_tools on real public ARC-AGI-3 games, offline, with a perfect predictor.

The predictor here is the game itself (a deepcopy of the engine, held per board), so this checks
the tools' mechanics, not a model. plan() must find the level's optimum (the T2 ceiling probe of
2026-10-16 measured ls20 L1 = 13 and tu93 L1 = 18 by frame-hash BFS); run_plan() must complete
the level in exactly that many real actions with no mismatch; check_model() must reproduce every
logged transition with the perfect predictor and flag the first one a "nothing moves" predictor
gets wrong.

Run with the toolkit's venv (arc-agi 0.9.9, arcengine 0.9.3):
  <venv>/python -I tests/real_games.py --env-dir <environment_files> [game ...]
"""
import argparse
import copy
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import twin_tools as tw  # noqa: E402
from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

ENGINE = {"UP": GameAction.ACTION1, "DOWN": GameAction.ACTION2, "LEFT": GameAction.ACTION3,
          "RIGHT": GameAction.ACTION4, "SPACE": GameAction.ACTION5, "MOUSE": GameAction.ACTION6}
NAMES = {v: k for k, v in ENGINE.items()}
OPTIMUM = {"ls20": 13, "tu93": 18}  # T2, level 1


def rows_of(obs):
    frame = obs.frame[-1]
    return tuple("".join(tw.TWIN_CHARS[int(v)] for v in row) for row in frame)


class FV:
    def __init__(self, obs, step):
        self.ascii = "\n".join(rows_of(obs))
        self.level = obs.levels_completed + 1
        self.step = step
        self.shape = (len(obs.frame[-1]), len(obs.frame[-1][0]))


class HE:
    def __init__(self, action, frame, result):
        self.action, self.frame, self.result = action, frame, dict(result)


class TV:
    def __init__(self, action, before, after, result):
        self.action, self.before_frame, self.after_frame, self.frame = action, before, after, after
        self.result = dict(result)


def engine_step(env, spec):
    name, row, col = tw.twin_parse_action(spec)
    data = {"x": col, "y": row} if name == "MOUSE" else {}
    return env.step(ENGINE[name], data=data)


class RealRuntime:
    """The harness's history/transitions/action() over a live offline game; a death is followed
    by an automatic RESET entry, as the harness records it."""

    def __init__(self, env, obs):
        self.env, self.obs, self.steps = env, obs, 0
        self.history = [HE("", FV(obs, 0), {})]

    def frame(self):
        return self.history[-1].frame

    @property
    def transitions(self):
        return [TV(e.action, self.history[i - 1].frame, e.frame, e.result)
                for i, e in enumerate(self.history) if e.action]

    def act(self, actions):
        last = {}
        for spec in actions:
            before = self.obs
            obs = engine_step(self.env, spec)
            self.steps += 1
            result = {"level_completed": obs.levels_completed > before.levels_completed,
                      "game_over": obs.state == GameState.GAME_OVER,
                      "run_complete": obs.state == GameState.WIN}
            self.history.append(HE(tw.twin_action_text(spec), FV(obs, self.steps), result))
            self.obs = obs
            if result["game_over"]:
                self.obs = self.env.step(GameAction.RESET, data={})
                self.history.append(HE("RESET", FV(self.obs, self.steps), {"automatic": True}))
            last = result
            if any(result.values()):
                break
        return last


class Oracle:
    """predict(frame, action) by stepping a deepcopy of the engine held for that exact board.

    Callers ask about one board at a time and never come back to it (breadth-first search
    expands a board once; run_plan and check_model walk forward), so a board's snapshot is
    dropped as soon as the next board is asked about. ls20's snapshots are ~1.2 MB each.
    """

    def __init__(self, env, obs):
        self.snap = {rows_of(obs): copy.deepcopy(env)}
        self.calls = 0
        self.current = None
        self.peak = 1

    def predict(self, frame, action):
        if self.current is not None and frame.rows != self.current:
            self.snap.pop(self.current, None)
        self.current = frame.rows
        env0 = self.snap.get(frame.rows)
        if env0 is None:
            return None
        self.calls += 1
        env1 = copy.deepcopy(env0)
        obs = engine_step(env1, action)
        if obs is None or obs.state == GameState.GAME_OVER:
            return None  # a death is not a move worth planning through
        rows = rows_of(obs)
        self.snap.setdefault(rows, env1)
        self.peak = max(self.peak, len(self.snap))
        return tw.TwinFrame(rows, level=obs.levels_completed + 1)


def still(frame, action):
    return {}  # the null hypothesis: no action changes anything


def run_game(arc, game, time_limit):
    out = {"game": game}
    env = arc.make(game, seed=0)
    obs = env.reset()
    moves = [NAMES[a] for a in env.action_space if a in NAMES and NAMES[a] != "MOUSE"]
    out["moves"] = moves
    start = FV(obs, 0)
    oracle = Oracle(env, obs)
    level0 = start.level
    t0 = time.perf_counter()
    planned = tw.twin_plan(oracle.predict, lambda f: f.level > level0, start, moves=moves,
                           max_nodes=200000, time_limit=time_limit)
    out["plan"] = {k: planned[k] for k in ("status", "length", "nodes", "visited", "seconds")}
    out["plan"]["predict_calls"] = oracle.calls
    out["plan"]["peak_snapshots"] = oracle.peak
    out["optimum_t2"] = OPTIMUM.get(game)
    if planned["status"] != "found":
        return out
    env = arc.make(game, seed=0)
    obs = env.reset()
    rt = RealRuntime(env, obs)
    ran = tw.twin_run_plan(planned["actions"], Oracle(env, obs).predict, rt.act, rt.frame)
    out["run"] = {k: ran.get(k) for k in ("stopped", "executed")}
    out["run"]["levels_completed"] = rt.obs.levels_completed
    fresh = arc.make(game, seed=0)
    fresh_obs = fresh.reset()
    checked = tw.twin_check_model(Oracle(fresh, fresh_obs).predict, rt.transitions, level="all",
                                  ignore=None, include_terminal=True)
    out["check_oracle"] = {k: checked[k] for k in ("checked", "matched", "mismatched", "abstained")}
    null = tw.twin_check_model(still, rt.transitions, level="all", ignore=None)
    out["check_null"] = {k: null[k] for k in ("checked", "matched", "mismatched")}
    out["check_null"]["summary"] = null["summary"]
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("--time-limit", type=float, default=900.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("games", nargs="*", default=["tu93", "ls20"])
    args = ap.parse_args()
    tw.TWIN_MAX_SECONDS = args.time_limit  # the 20 s cap is the sandbox's; a probe may run longer
    rec_dir = tempfile.mkdtemp(prefix="twin_rec_")
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=args.env_dir, recordings_dir=rec_dir)
    results = []
    for game in args.games:
        res = run_game(arc, game, args.time_limit)
        print(json.dumps(res), flush=True)
        results.append(res)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    ok = all(r.get("run", {}).get("stopped") == "level_completed"
             and r["run"]["executed"] == r["plan"]["length"]
             and (r["optimum_t2"] is None or r["plan"]["length"] == r["optimum_t2"])
             and r["check_oracle"]["mismatched"] == 0
             for r in results)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
