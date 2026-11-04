"""S4 click agent - survey and sequence hypotheses for click games (ACTION6).

Per game:
  survey   click the centre of every object once (largest first, then by position); record for
           each click: nothing / changed the clicked object / changed others / level-up / death.
  classes  objects whose click changed something are 'live'; hypotheses in order:
             (a) click every live object of the same colour once (toggle-all),
             (b) click live objects in pairs (A then B),
             (c) click the same live object repeatedly (cycle) up to 4 times,
             (d) re-survey the background on a coarse lattice.
  levels   a level-up records the sequence of object keys clicked since the last level-up; later
           levels replay it by key (then colour), falling back to the survey.
Budget: 3x the human baseline sum, cap 600. Usage: python s4c_click.py [--budget N] [--debug K] [game ...]
"""
import argparse
import json
import time
from collections import Counter

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcobj import deltas, segment

CLICK_GAMES = ["r11l", "ft09", "vc33", "s5i5", "tn36", "su15", "sb26", "lp85", "lf52", "cd82", "sc25", "sk48", "bp35", "cn04"]


def human_baselines(game):
    try:
        return json.load(open("arc3_t1.json", encoding="utf-8"))["baselines"].get(game)
    except Exception:  # noqa: BLE001
        return None


def okey(o):
    return (o.color, o.shape)


def centre(o):
    xs = [c[0] for c in o.cells]
    ys = [c[1] for c in o.cells]
    cx, cy = sum(xs) // len(xs), sum(ys) // len(ys)
    if (cx, cy) in set(o.cells):
        return cx, cy
    return min(o.cells, key=lambda c: abs(c[0] - cx) + abs(c[1] - cy))


class ClickAgent:
    def __init__(self, game, budget, debug=0):
        self.game, self.budget, self.debug = game, budget, debug
        self.diag = Counter()
        self.goal_seq = None

    def click(self, env, obs, x, y):
        return env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})

    def run(self, arc):
        env = arc.make(self.game, seed=0)
        obs = env.reset()
        bg, objs = segment(obs.frame[-1])
        actions = 0
        level_actions = []
        cur_level = 0
        last_level = obs.levels_completed
        since_level = []  # keys clicked since the last level-up
        live = {}  # key -> count of clicks that changed something
        dead = Counter()  # key -> clicks that changed nothing
        clicked_keys = set()
        clicked_pos = set()
        survey_cap = max(12, 2 * len(objs))
        phase = "survey"
        queue = []  # list of (x, y, key) to click
        pair_idx = 0
        cycle_count = Counter()
        lattice = [(x, y) for y in range(4, 64, 8) for x in range(4, 64, 8)]
        lat_i = 0
        while actions < self.budget:
            frame = obs.frame[-1]
            bg, objs = segment(frame, bg)
            objs = [o for o in objs if o.size <= 400]
            bykey = {}
            for o in objs:
                bykey.setdefault(okey(o), []).append(o)
            if not queue:
                # 0. replay a known sequence
                if self.goal_seq and phase != "replayed":
                    for k in self.goal_seq:
                        cands = bykey.get(k) or [o for kk in bykey if kk[0] == k[0] for o in bykey[kk]]
                        if cands:
                            queue.append(centre(cands[0]) + (okey(cands[0]),))
                    phase = "replayed"
                    self.diag["replays"] += 1
                # 1. survey: every unclicked object once, largest first
                if not queue and self.diag["survey"] < survey_cap:
                    todo = [o for o in objs if (o.color, o.x0, o.y0) not in clicked_pos and dead[okey(o)] < 2]
                    if todo:
                        o = max(todo, key=lambda o: (o.size, -o.y0, -o.x0))
                        queue.append(centre(o) + (okey(o),))
                        clicked_pos.add((o.color, o.x0, o.y0))
                        self.diag["survey"] += 1
                # 2. toggle-all: click every live object of one colour
                if not queue and live:
                    col = max(live, key=live.get)[0]
                    same = [o for o in objs if o.color == col and okey(o) in live]
                    if same and self.diag["toggle_all"] < 6:
                        for o in sorted(same, key=lambda o: (o.y0, o.x0)):
                            queue.append(centre(o) + (okey(o),))
                        self.diag["toggle_all"] += 1
                # 3. pairs of live objects
                if not queue and len(live) >= 2:
                    keys = sorted(live, key=live.get, reverse=True)
                    pairs = [(a, b) for a in keys for b in keys if a != b]
                    if pair_idx < len(pairs):
                        a, b = pairs[pair_idx]
                        pair_idx += 1
                        for k in (a, b):
                            if k in bykey:
                                queue.append(centre(bykey[k][0]) + (k,))
                        self.diag["pairs"] += 1
                # 4. cycle one live object
                if not queue and live:
                    for k in sorted(live, key=live.get, reverse=True):
                        if cycle_count[k] < 4 and k in bykey:
                            queue.append(centre(bykey[k][0]) + (k,))
                            cycle_count[k] += 1
                            self.diag["cycle"] += 1
                            break
                # 5. background lattice
                if not queue:
                    x, y = lattice[lat_i % len(lattice)]
                    lat_i += 1
                    queue.append((x, y, ("lat", lat_i)))
                    self.diag["lattice"] += 1
            x, y, key = queue.pop(0)
            prev_objs = objs
            nobs = self.click(env, obs, x, y)
            actions += 1
            cur_level += 1
            if nobs is None or not nobs.frame:
                if nobs is not None and nobs.state == GameState.GAME_OVER:
                    nobs = env.reset()
                    obs = nobs
                    continue
                break
            if self.debug and actions <= self.debug:
                print(f"  t={actions} click=({x},{y}) key={key[0] if isinstance(key, tuple) else key} lv={nobs.levels_completed} state={nobs.state.name} live={len(live)}", flush=True)
            if nobs.levels_completed > last_level:
                self.goal_seq = [k for k in since_level + [key] if not (isinstance(k, tuple) and k and k[0] == "lat")][-4:]
                level_actions.append(cur_level)
                cur_level = 0
                last_level = nobs.levels_completed
                since_level, queue, clicked_keys, clicked_pos = [], [], set(), set()
                self.diag["survey"] = 0
                live, dead, cycle_count, pair_idx = {}, Counter(), Counter(), 0
                phase = "survey"
                self.diag["level_ups"] += 1
                obs = nobs
                if obs.state == GameState.WIN:
                    break
                continue
            if nobs.state == GameState.GAME_OVER:
                self.diag["deaths"] += 1
                dead[key] += 2
                nobs = env.reset()
                obs = nobs
                continue
            ds, ncells, _, _ = deltas(frame, nobs.frame[-1], prev_objs=prev_objs, bg=bg)
            if key[0] != "lat":
                clicked_keys.add(key)
                since_level.append(key)
                if ncells > 0:
                    live[key] = live.get(key, 0) + 1
                    self.diag["live_clicks"] += 1
                else:
                    dead[key] += 1
            obs = nobs
        return obs.levels_completed, actions, level_actions, self.goal_seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("games", nargs="*")
    args = ap.parse_args()
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    games = args.games or CLICK_GAMES
    out = {}
    for g in games:
        t0 = time.perf_counter()
        base = human_baselines(g)
        budget = min(args.budget, 3 * sum(base)) if base else args.budget
        ag = ClickAgent(g, budget, args.debug)
        levels, actions, la, seq = ag.run(arc)
        scores = [round(min((base[i] / x) ** 2, 1.15), 3) if base and i < len(base) else None for i, x in enumerate(la)]
        n = len(base) if base else None
        gs = round(sum((s or 0) * (i + 1) for i, s in enumerate(scores)) / sum(range(1, n + 1)), 3) if n else None
        out[g] = {"levels": levels, "of": n, "actions": actions, "level_actions": la, "human": base, "scores": scores, "game_score": gs,
                  "seq": [list(map(str, k)) for k in seq] if seq else None, "diag": dict(ag.diag), "secs": round(time.perf_counter() - t0, 1)}
        print(f"{g}: levels={levels}/{n} actions={actions}/{budget} per-level={la} human={base[:len(la)] if base else None} scores={scores} game_score={gs} diag={dict(ag.diag)}", flush=True)
    json.dump(out, open("arc3_s4c.json", "w"), indent=1)
    done = [g for g in games if out[g]["levels"] >= 2]
    any1 = [g for g in games if out[g]["levels"] >= 1]
    mean = sum(out[g]["game_score"] or 0 for g in games) / len(games)
    print(f"GATE S4 (click): >=2 levels on {len(done)} {done}; >=1 level on {len(any1)} {any1}; mean game score {mean:.3f} over {len(games)} games")


if __name__ == "__main__":
    main()
