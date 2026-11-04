"""T3 - seal an ARC-AGI-3 play as a replay record and re-derive every digest.

Digest rule: d0 = sha256("replay:" + game_version + ":" + seed + ":" + frame0)
  d_i = sha256(d_{i-1} + ":" + action_text + ":" + frame_i)
  frame text = levels_completed, then each 64x64 frame as rows of hex digits joined by "/",
  multiple frames joined by "|". Click encoding: "6@x,y"; simple actions: "1".."5","7"; reset "0".
Usage: python t3_seal.py [--runs N]   (takes runs from arc3_t1.json; default 6 across games)
"""
import argparse
import hashlib
import json
import time

from arc_agi import Arcade, OperationMode
from arcengine import GameAction, GameState

ACT = {a.value: a for a in GameAction}


def frame_text(obs):
    frames = ["/".join("".join(format(c, "x") for c in row) for row in f) for f in obs.frame]
    return f"{obs.levels_completed}:" + "|".join(frames)


def action_text(a):
    if len(a) == 3 and a[1] is not None:
        return f"{a[0]}@{a[1]},{a[2]}"
    return str(a[0])


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def play(arc, game, seed, actions):
    env = arc.make(game, seed=seed)
    obs = env.reset()
    ver = env.info.get("version") if hasattr(env, "info") and isinstance(env.info, dict) else None
    d = sha(f"replay:{game}:{ver}:{seed}:{frame_text(obs)}")
    digests = [d]
    for a in actions:
        ga = ACT[a[0]]
        data = {"x": a[1], "y": a[2]} if len(a) == 3 and a[1] is not None else {}
        obs = env.step(ga, data=data)
        if obs is None:
            break
        d = sha(f"{d}:{action_text(a)}:{frame_text(obs)}")
        digests.append(d)
        if obs.state == GameState.GAME_OVER:
            obs = env.reset()
            d = sha(f"{d}:0:{frame_text(obs)}")
            digests.append(d)
    return digests, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=6)
    args = ap.parse_args()
    t1 = json.load(open("arc3_t1.json", encoding="utf-8"))
    rows = [r for r in t1["rows"] if r["policy"] == "random" and r["seed"] == 0 and r["steps"] >= 50]
    rows = rows[: args.runs]
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    records = []
    ok_all = True
    for r in rows:
        t0 = time.perf_counter()
        d1, obs1 = play(arc, r["game"], r["seed"], r["actions"])
        d2, obs2 = play(arc, r["game"], r["seed"], r["actions"])  # fresh make(): re-derivation
        same = d1 == d2
        ok_all &= same
        rec = {
            "v": "arc3-replay/1", "toolkit": "arc-agi-0.9.9", "game": r["game"], "seed": r["seed"],
            "actions": [action_text(a) for a in r["actions"]], "plies": len(r["actions"]),
            "outcome": "win" if obs1.state == GameState.WIN else "loss",
            "levels_completed": obs1.levels_completed, "start_digest": d1[0], "digest": d1[-1],
            "state_digests": d1[1:],
        }
        rec["record_sha256"] = sha(json.dumps(rec, sort_keys=True, separators=(",", ":")))
        records.append(rec)
        print(f"{r['game']} seed={r['seed']} plies={len(r['actions'])} digests={len(d1)} "
              f"re-derived={'OK' if same else 'MISMATCH'} {time.perf_counter()-t0:.1f}s record={rec['record_sha256'][:12]}")
    json.dump(records, open("arc3_t3_records.json", "w", encoding="utf-8"), indent=1)
    print("ALL RE-DERIVED" if ok_all else "SOME MISMATCH")


if __name__ == "__main__":
    main()
