"""S2 gate: online rule induction on the T1 streams (seed 0: random + one_probe + repeat_last_change
+ all blind/click policies, concatenated as one play history per game).

Reports prequential accuracy of the full context tree vs two baselines (always-NONE; action-only
table) in windows after 10, 50 and 200 observations, overall and on CHANGE steps only (steps whose
true signature is not NONE), plus the induced rules with support.
"""
import json
import time
from collections import Counter

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

from arcobj import Tracker
from arcrules import NONE_SIG, ContextTree, Inducer

ACT = {a.value: a for a in GameAction}
WINDOWS = [(0, 10), (10, 50), (50, 200), (200, 10**9)]


def run_game(arc, game, rows):
    tr = Tracker()
    ind = Inducer(tr)
    act_only = ContextTree()
    log = []  # (sig, ok_full, ok_action_only)
    for r in rows:
        env = arc.make(game, seed=r["seed"])
        obs = env.reset()
        tr.start(obs.frame[-1])
        for a in r["actions"]:
            data = {"x": a[1], "y": a[2]} if len(a) == 3 and a[1] is not None else {}
            nobs = env.step(ACT[a[0]], data=data)
            ok, sig = ind.step(a[0], data, obs, nobs)
            ok1 = act_only.observe((a[0],), sig)
            log.append((sig, ok, ok1))
            if nobs is None:
                break
            obs = nobs
            if obs.state == GameState.GAME_OVER:
                obs = env.reset()
                tr.start(obs.frame[-1])
    return ind, log


def windowed(log):
    out = {}
    for lo, hi in WINDOWS:
        seg = log[lo:hi]
        if not seg:
            continue
        ch = [x for x in seg if x[0] != NONE_SIG]
        out[f"{lo}-{min(hi, len(log))}"] = {
            "n": len(seg), "full": round(sum(x[1] for x in seg) / len(seg), 3),
            "act_only": round(sum(x[2] for x in seg) / len(seg), 3),
            "none_base": round(sum(x[0] == NONE_SIG for x in seg) / len(seg), 3),
            "n_change": len(ch),
            "full_change": round(sum(x[1] for x in ch) / len(ch), 3) if ch else None,
            "act_only_change": round(sum(x[2] for x in ch) / len(ch), 3) if ch else None,
        }
    return out


def main():
    t1 = json.load(open("arc3_t1.json", encoding="utf-8"))
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    games = sorted({r["game"] for r in t1["rows"]})
    order = ["one_probe", "repeat_last_change", "random", "blind_A1", "blind_A2", "blind_A3", "blind_A4", "blind_A5", "click_center", "click_sweep"]
    results = {}
    for g in games:
        rows = [r for r in t1["rows"] if r["game"] == g and r["seed"] == 0 and r["steps"] > 0]
        rows.sort(key=lambda r: order.index(r["policy"]) if r["policy"] in order else 99)
        t0 = time.perf_counter()
        ind, log = run_game(arc, g, rows)
        w = windowed(log)
        rules = ind.tree.rules()
        simple = all(len(a) < 3 for r in rows for a in r["actions"])
        results[g] = {"simple": simple, "n": len(log), "windows": w, "codelen_per_step": round(ind.tree.codelen / max(len(log), 1), 3),
                      "alphabet": len(ind.tree.alphabet), "rules": [[list(map(str, p)), [list(s) for s in sig], c, t, pr] for p, sig, c, t, pr in rules[:12]],
                      "secs": round(time.perf_counter() - t0, 1)}
        k50 = w.get("50-200") or w.get("10-50")
        print(f"{g} simple={simple} n={len(log)} |sig|={len(ind.tree.alphabet)} bits/step={ind.tree.codelen/max(len(log),1):.2f} "
              f"w50-200: full={k50['full']} act={k50['act_only']} none={k50['none_base']} change(n={k50['n_change']}): full={k50['full_change']} act={k50['act_only_change']}", flush=True)
    json.dump(results, open("arc3_s2.json", "w"), indent=1)
    passed = [g for g in games if (results[g]["windows"].get("50-200") or {}).get("full", 0) >= 0.8]
    simple = [g for g in games if results[g]["simple"]]
    print(f"GATE: >=0.80 full accuracy in window 50-200 on {len(passed)}/25 games {passed}; simple games passing: {[g for g in simple if g in passed]}/{simple}")


if __name__ == "__main__":
    main()
