"""Cost per COMPLETED level on clock-bound demo runs, so the clock's tail does not masquerade as efficiency.

python -I analysis/demo_cost.py <label=output dir> [...]

On the public demo every run is clock-bound, so total tokens ~ throughput x wallclock and "tokens per level
solved" collapses to 1/levels. Here, per game-pass, the generated characters (analyze_run's gen_chars_by_level:
reasoning + reply + code, by the level being played) are split into the levels the pass completed (1..L) and
the tail (level L+1, unfinished, cut by the clock). Reported per run: runs, levels, mean final score, chars per
completed level, the tail's share, and the per-level medians for levels 1-3 (the levels most runs complete).
"""
import json
import os
import statistics
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "entry", "tools"))
import analyze_run  # noqa: E402


def summary_field(out_dir, key):
    path = os.path.join(out_dir, "summary.txt")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].strip()
    return None


rows = []
per_level = {}
for arg in sys.argv[1:]:
    label, _, root = arg.partition("=")
    out = analyze_run.summarise(analyze_run.analyse(root))
    done_chars = tail_chars = 0
    levels = 0
    scores = []
    lv = {}
    for r in out["per_game"]:
        L = int(r["levels"] or 0)
        levels += L
        if r["score"] is not None:
            scores.append(float(r["score"]))
        for k, n in r["gen_chars_by_level"].items():
            try:
                k = int(k)
            except ValueError:
                continue
            if k <= L:
                done_chars += n
                lv.setdefault(k, []).append(n)
            else:
                tail_chars += n
    per_level[label] = lv
    rows.append((label, len(out["per_game"]), levels, statistics.mean(scores) if scores else None,
                 done_chars / max(1, levels), tail_chars / max(1, done_chars + tail_chars),
                 [round(statistics.median(lv[k]) / 1000.0, 1) if lv.get(k) else None for k in (1, 2, 3)],
                 summary_field(root, "started"), summary_field(root, "generated tokens/sec")))

print("run              runs levels  mean   chars/completed level  tail share  median k-chars at L1 L2 L3   started              tok/s")
for label, n, levels, mean, cpl, tail, med, started, tps in rows:
    print("%-16s %4d %6d  %5s  %21.0f  %9.1f%%  %-28s %-20s %s" % (
        label, n, levels, "%.2f" % mean if mean is not None else "-", cpl, 100 * tail, med, started, tps))
