"""T1 - the non-intelligent-strategy probe over every cached ARC-AGI-3 public game.

Policies (all memoryless or one-bit):
  blind_A1..blind_A5  one fixed simple action forever (only if available)
  click_center        ACTION6 at (32,32) forever
  click_sweep         ACTION6 rastering a coarse grid (step 4) forever
  one_probe           try each available action once; keep the first that changes the frame
  repeat_last_change  repeat whichever action last changed the frame, else try the next one
  random              uniform over available actions (clicks uniform over the grid)

Per (game, policy, seed): N actions, level resets on GAME_OVER (counted), record levels
completed, actions to first level, human baseline per level (from the local scorecard), and the
action list (for T3/T4). Output: arc3_t1.json + a one-line-per-game summary on stdout.

Usage: python t1_probe.py [--fetch] [--actions N] [--seeds K] [game ...]
"""
import argparse
import json
import random
import sys
import time

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

SIMPLE = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]


def frame_key(obs):
    f = obs.frame[-1] if obs.frame else []
    return hash(tuple(tuple(r) for r in f))


class Policy:
    name = "base"

    def __init__(self, rng):
        self.rng = rng

    def act(self, obs, acts, changed):
        raise NotImplementedError

    @staticmethod
    def click(x, y):
        return GameAction.ACTION6, {"x": x, "y": y}


class Blind(Policy):
    def __init__(self, rng, a):
        super().__init__(rng)
        self.a = a
        self.name = "blind_" + a.name.replace("ACTION", "A")

    def act(self, obs, acts, changed):
        return (self.a, {}) if self.a in acts else None


class ClickCenter(Policy):
    name = "click_center"

    def act(self, obs, acts, changed):
        return self.click(32, 32) if GameAction.ACTION6 in acts else None


class ClickSweep(Policy):
    name = "click_sweep"

    def __init__(self, rng):
        super().__init__(rng)
        self.i = 0

    def act(self, obs, acts, changed):
        if GameAction.ACTION6 not in acts:
            return None
        n = 16  # 16x16 grid of step 4, offset 2
        x = 2 + 4 * (self.i % n)
        y = 2 + 4 * ((self.i // n) % n)
        self.i += 1
        return self.click(x, y)


class OneProbe(Policy):
    name = "one_probe"

    def __init__(self, rng):
        super().__init__(rng)
        self.tried = []
        self.keep = None
        self.last = None

    def options(self, acts):
        out = [(a, {}) for a in acts if a != GameAction.ACTION6]
        if GameAction.ACTION6 in acts:
            out += [self.click(x, y) for y in (16, 32, 48) for x in (16, 32, 48)]
        return out

    def act(self, obs, acts, changed):
        if self.keep is None and self.last is not None and changed:
            self.keep = self.last
        if self.keep is not None:
            return self.keep
        opts = [o for o in self.options(acts) if o not in self.tried]
        if not opts:
            self.tried = []
            opts = self.options(acts)
        if not opts:
            return None
        self.last = opts[0]
        self.tried.append(opts[0])
        return self.last


class RepeatLastChange(OneProbe):
    name = "repeat_last_change"

    def act(self, obs, acts, changed):
        if self.last is not None and changed:
            self.keep = self.last
        elif self.last is not None and not changed:
            self.keep = None
        if self.keep is not None:
            return self.keep
        opts = [o for o in self.options(acts) if o not in self.tried]
        if not opts:
            self.tried = []
            opts = self.options(acts)
        if not opts:
            return None
        self.last = opts[0]
        self.tried.append(opts[0])
        return self.last


class Random(Policy):
    name = "random"

    def act(self, obs, acts, changed):
        if not acts:
            return None
        a = self.rng.choice(list(acts))
        if a == GameAction.ACTION6:
            return self.click(self.rng.randint(0, 63), self.rng.randint(0, 63))
        return a, {}


def policies(rng):
    return [Blind(rng, a) for a in SIMPLE] + [ClickCenter(rng), ClickSweep(rng), OneProbe(rng), RepeatLastChange(rng), Random(rng)]


def run(arc, game, pol, seed, n):
    env = arc.make(game, seed=seed)
    obs = env.reset()
    key = frame_key(obs)
    actions = []
    first_level_at = None
    resets = 0
    changed = False
    t0 = time.perf_counter()
    for i in range(n):
        acts = env.action_space
        choice = pol.act(obs, acts, changed)
        if choice is None:
            break
        a, data = choice
        nobs = env.step(a, data=data)
        if nobs is None:
            break
        actions.append([a.value, data.get("x"), data.get("y")] if data else [a.value])
        nkey = frame_key(nobs)
        changed = nkey != key or nobs.levels_completed != obs.levels_completed
        key = nkey
        if nobs.levels_completed > obs.levels_completed and first_level_at is None:
            first_level_at = i + 1
        obs = nobs
        if obs.state == GameState.WIN:
            break
        if obs.state == GameState.GAME_OVER:
            resets += 1
            obs = env.reset()
            key = frame_key(obs)
            changed = False
    dt = time.perf_counter() - t0
    return {
        "game": game, "policy": pol.name, "seed": seed, "steps": len(actions), "levels": obs.levels_completed,
        "win_levels": obs.win_levels, "state": obs.state.name, "first_level_at": first_level_at,
        "resets": resets, "secs": round(dt, 2), "actions": actions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--actions", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("games", nargs="*")
    args = ap.parse_args()

    if args.fetch:
        arc = Arcade(operation_mode=OperationMode.NORMAL)
        ids = [e.game_id for e in arc.get_environments()]
        for g in ids:
            env = arc.make(g.split("-")[0], seed=0)
            print("fetched", g, "ok" if env else "FAILED", flush=True)

    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    ids = sorted(e.game_id for e in arc.get_environments())
    games = args.games or [g.split("-")[0] for g in ids]
    rows = []
    baselines = {}
    for g in games:
        for seed in range(args.seeds):
            for pol in policies(random.Random(seed)):
                r = run(arc, g, pol, seed, args.actions)
                rows.append(r)
        sc = arc.get_scorecard()
        for e in sc.environments:
            if e.id.startswith(g):
                for ru in e.runs:
                    if ru.level_baseline_actions:
                        baselines[g] = ru.level_baseline_actions
        best = {}
        for r in rows:
            if r["game"] == g:
                b = best.get(r["policy"], (0, None))
                if (r["levels"], -(r["first_level_at"] or 10**9)) > (b[0], -(b[1] or 10**9)):
                    best[r["policy"]] = (r["levels"], r["first_level_at"])
        win = next((r["win_levels"] for r in rows if r["game"] == g), None)
        print(f"{g}: win_levels={win} baseline={baselines.get(g)} best_levels_by_policy="
              + " ".join(f"{k}={v[0]}@{v[1]}" for k, v in sorted(best.items())), flush=True)
    with open("arc3_t1.json", "w", encoding="utf-8") as f:
        json.dump({"actions_per_run": args.actions, "seeds": args.seeds, "baselines": baselines, "rows": rows}, f)
    print("wrote arc3_t1.json", len(rows), "rows")


if __name__ == "__main__":
    main()
