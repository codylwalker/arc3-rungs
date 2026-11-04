"""Where does a demo run lose points? Per level: (human / ours)^2 capped at 1.15, against the human
baselines the offline toolkit ships, split into levels never finished and levels finished wastefully.

Usage (the toolkit's venv): python -I tools/loss_accounting.py --env-dir <environment_files> <run output dir> [...]
Reads each run's artifacts/*_viewer_data.json (game_id, levels_completed, total_levels, actions_per_level).
"""
import argparse
import glob
import json
import os
import sys
import tempfile


def baselines(env_dir, games):
    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir,
                 recordings_dir=tempfile.mkdtemp(prefix="loss_rec_"))
    out = {}
    for g in games:
        try:
            env = arc.make(g, seed=0)
            env.reset()
        except Exception as exc:  # noqa: BLE001
            print("no local game", g, exc, file=sys.stderr)
    sc = arc.get_scorecard()
    for e in sc.environments:
        for ru in e.runs:
            if ru.level_baseline_actions:
                out[e.id.split("-")[0]] = list(ru.level_baseline_actions)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    runs = {}
    for run in args.runs:
        rows = {}
        for path in glob.glob(os.path.join(run, "**", "*_viewer_data.json"), recursive=True):
            d = json.load(open(path, encoding="utf-8"))
            rows[d["game_id"].split("-")[0]] = d
        runs[run] = rows
    games = sorted(set(g for rows in runs.values() for g in rows))
    base = baselines(args.env_dir, games)
    for run, rows in runs.items():
        print("=" * 100)
        print(run)
        tot_unfinished = tot_waste = tot_got = tot_w = 0.0
        for g in games:
            d = rows.get(g)
            h = base.get(g)
            if d is None or h is None:
                print("  %-6s missing (run %s, baseline %s)" % (g, d is not None, h is not None))
                continue
            n = len(h)
            w = list(range(1, n + 1))
            apl = list(d.get("actions_per_level") or [])
            done = int(d.get("levels_completed") or 0)
            per = []
            for i in range(n):
                if i < done and i < len(apl) and apl[i] > 0:
                    per.append(min(1.15, (h[i] / apl[i]) ** 2))
                else:
                    per.append(0.0)
            got = sum(wi * si for wi, si in zip(w, per)) / sum(w)
            unfinished = sum(wi for i, wi in enumerate(w) if i >= done) / sum(w)
            waste = sum(wi * max(0.0, 1.0 - si) for i, (wi, si) in enumerate(zip(w, per)) if i < done) / sum(w)
            tot_unfinished += unfinished
            tot_waste += waste
            tot_got += got
            tot_w += 1
            eff = ", ".join("L%d %d/%d=%.2f" % (i + 1, apl[i], h[i], per[i]) for i in range(min(done, len(apl))))
            print("  %-6s %d/%d levels | score %5.1f | lost: unfinished %4.1f, waste %4.1f | %s" % (
                g, done, n, 100 * got, 100 * unfinished, 100 * waste, eff))
        if tot_w:
            print("  MEAN score %.1f | lost to unfinished levels %.1f | lost to wasted actions on finished levels %.1f" % (
                100 * tot_got / tot_w, 100 * tot_unfinished / tot_w, 100 * tot_waste / tot_w))


if __name__ == "__main__":
    main()
