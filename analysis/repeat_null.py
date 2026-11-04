"""Do the unfinished games concentrate across runs? Exact null for the number of DISTINCT games they fall in.

python -I analysis/repeat_null.py <label=summary.txt> [...]

Null: each run's unfinished games are a uniform random k_r-subset of the n games, independently across runs.
Exact distribution of the number of distinct games covered, by a DP over runs: with c games covered, a run of
size k adds j new ones with hypergeometric probability C(n-c, j) C(c, k-j) / C(n, k). Reports the observed
distinct count, its expectation, and P(distinct <= observed) (fewer distinct = more repeats than chance).
"""
import math
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
LINE = re.compile(r"^\s+(g\w{3})-\w+: score=[\d.]+, levels=([\d.]+)/(\d+)")
runs = {}
n_games = None
for arg in sys.argv[1:]:
    label, _, path = arg.partition("=")
    ids, total = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line)
            if m:
                total += 1
                if float(m.group(2)) < float(m.group(3)):
                    ids.append(m.group(1))
    runs[label] = ids
    n_games = total if n_games is None else n_games
    if total != n_games:
        raise SystemExit("game counts differ: %s has %d" % (label, total))
n = n_games
dist = {0: 1.0}
for ids in runs.values():
    k = len(ids)
    new = {}
    for c, p in dist.items():
        for j in range(0, k + 1):
            if j > n - c or k - j > c:
                continue
            q = math.comb(n - c, j) * math.comb(c, k - j) / math.comb(n, k)
            new[c + j] = new.get(c + j, 0.0) + p * q
    dist = new
counts = Counter(g for ids in runs.values() for g in ids)
observed = len(counts)
total = sum(len(v) for v in runs.values())
expect = sum(c * p for c, p in dist.items())
p_le = sum(p for c, p in dist.items() if c <= observed)
print("%d runs over %d games, %d unfinished game-runs (%s)" % (len(runs), n, total, ", ".join("%s %d" % (k, len(v)) for k, v in runs.items())))
print("distinct games: observed %d, expected %.2f at random; P(distinct <= %d) = %.4f" % (observed, expect, observed, p_le))
print("repeats: %s" % ", ".join("%s x%d" % (g, c) for g, c in counts.most_common() if c > 1))
