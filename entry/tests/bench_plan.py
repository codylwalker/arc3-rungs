"""How fast plan() searches a model-style predict on full 64x64 boards.

A seeded 64x64 maze (walls G, a budget bar in the top wall row, player R, goal Y); predict scans
the rows for R the way a model's code does. Reports nodes per second, and whether the shortest
path is found inside the default 8 s and the 20 s cap, for a few start-to-goal distances.

Run: python -I tests/bench_plan.py
"""
import os
import random
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import twin_tools as tw  # noqa: E402

MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
N = 64


def board(seed, density):
    rng = random.Random(seed)
    grid = [["B"] * N for _ in range(N)]
    for i in range(N):
        grid[0][i] = grid[N - 1][i] = grid[i][0] = grid[i][N - 1] = "G"
    for i in range(1, 40):
        grid[0][i] = "S"
    for r in range(1, N - 1):
        for c in range(1, N - 1):
            if rng.random() < density:
                grid[r][c] = "G"
    return grid


def predict(frame, action):
    name = tw.twin_parse_action(action)[0]
    if name not in MOVES:
        return None
    for r, row in enumerate(frame.rows):
        c = row.find("R")
        if c >= 0:
            break
    else:
        return None
    dr, dc = MOVES[name]
    if frame.rows[r + dr][c + dc] in "GS":
        return {}
    return {(r, c): "B", (r + dr, c + dc): "R"}


def distance(grid, start, goal):
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c), d = queue.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in MOVES.values():
            n = (r + dr, c + dc)
            if grid[n[0]][n[1]] not in "GS" and n not in seen:
                seen.add(n)
                queue.append((n, d + 1))
    return None


def main():
    print("%-8s %-8s %-6s %-12s %-8s %-8s %s" % ("density", "start", "dist", "status", "length", "nodes", "s / nodes per s"))
    for density, start, goal in ((0.2, (32, 32), (32, 40)), (0.2, (32, 32), (20, 50)),
                                 (0.25, (5, 5), (58, 58)), (0.0, (5, 5), (58, 58))):
        grid = board(7, density)
        for (r, c) in (start, goal):
            grid[r][c] = "B"
        grid[goal[0]][goal[1]] = "Y"
        grid[start[0]][start[1]] = "R"
        rows = ["".join(row) for row in grid]
        true = distance(grid, start, goal)
        for limit in (8.0, 20.0):
            t0 = time.perf_counter()
            out = tw.twin_plan(predict, lambda f, g=goal: f.rows[g[0]][g[1]] == "R", tw.TwinFrame(rows, 1, 0),
                               moves=list(MOVES), max_nodes=10 ** 7, time_limit=limit)
            dt = time.perf_counter() - t0
            print("%-8s %-8s %-6s %-12s %-8s %-8s %.1f / %.0f  (limit %ss)" % (
                density, start, true, out["status"], out["length"], out["nodes"], dt, out["nodes"] / max(dt, 1e-9),
                int(limit)))
            if out["status"] == "found":
                assert out["length"] == true, (out["length"], true)
                break


if __name__ == "__main__":
    main()
