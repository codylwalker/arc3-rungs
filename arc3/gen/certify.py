"""Oracle and certification for generated games.

oracle_solve(game, level, max_nodes, max_secs): shortest action list that completes `level` of a raw
GenGame (export_state/import_state snapshots; BFS over the game's movement actions and ACTION5).
baselines(plan, ...): per-level optimal lengths and the metadata baselines ceil(1.5 x optimal).
Phase B adds non-triviality, bands and novelty here.
"""
import math
import time
from collections import deque

from arcengine import ActionInput, GameAction, GameState

RATIO = 1.5
BUDGET_RATIO = 3


def _actions(game):
    if 6 in game._available_actions:
        return game._get_valid_clickable_actions()
    return [ActionInput(id=GameAction.from_id(a)) for a in game._available_actions if a != 6]


def _rec(a):
    """An action as a JSON-able record: [id] or [6, x, y]."""
    if a.id == GameAction.ACTION6:
        return [6, int(a.data["x"]), int(a.data["y"])]
    return [a.id.value]


def _inp(rec):
    if isinstance(rec, int):  # certificates written before 10-26's click family stored bare ids
        rec = [rec]
    if rec[0] == 6:
        return ActionInput(id=GameAction.ACTION6, data={"x": rec[1], "y": rec[2]})
    return ActionInput(id=GameAction.from_id(rec[0]))


def oracle_solve(game, level, max_nodes=50000, max_secs=120.0):
    """(path, nodes, secs) for `level` (0-based); path None if the cap hit or no solution."""
    game.full_reset()
    if level > 0:
        game.set_level(level)
        game._score = level
    game._state = GameState.NOT_FINISHED
    root = game.export_state()
    key = lambda s: s[:-1]  # the action count is not state
    seen = {key(root)}
    q = deque([(root, ())])
    click = 6 in game._available_actions
    acts = _actions(game)
    nodes = 0
    t0 = time.perf_counter()
    while q:
        snap, path = q.popleft()
        if click:
            game.import_state(snap)
            acts = _actions(game)
        for a in acts:
            game.import_state(snap)
            fd = game.perform_action(a, raw=True)
            nodes += 1
            if fd.levels_completed > level or fd.state == GameState.WIN:
                return list(path) + [_rec(a)], nodes, time.perf_counter() - t0
            if fd.state == GameState.GAME_OVER:
                continue
            s2 = game.export_state()
            k = key(s2)
            if k in seen:
                continue
            seen.add(k)
            q.append((s2, path + (_rec(a),)))
            if nodes >= max_nodes or time.perf_counter() - t0 > max_secs:
                return None, nodes, time.perf_counter() - t0
    return None, nodes, time.perf_counter() - t0


def _toggle_masks(w, h, rule):
    """masks[j]: the tiles (bit y*w+x) that a click on tile j flips, from compose's rule (the runtime's twin;
    a mismatch fails the replay of the emitted file)."""
    import compose
    masks = []
    for cy in range(h):
        for cx in range(w):
            s = [[0] * w for _ in range(h)]
            compose._apply_rule(s, cx, cy, rule)
            masks.append(sum(1 << (y * w + x) for y in range(h) for x in range(w) if s[y][x]))
    return masks


def gf2_min_clicks(masks, b, n, max_free=20):
    """The fewest-clicks set x (bit j = click tile j) with XOR of masks[j] over x == b, as (x, candidates);
    x None if no set works, or if the null space has more than max_free dimensions (then search instead).
    Clicks commute and a double click cancels, so a toggle optimum is a set: Gaussian elimination over GF(2)
    gives one solution, and the 2^k null-space translates hold every other."""
    piv = []  # (pivot column, row bits over the clicks, right-hand side); kept fully reduced
    for i in range(n):
        r = sum(1 << j for j in range(len(masks)) if masks[j] >> i & 1)
        v = b >> i & 1
        for c, pr, pv in piv:
            if r >> c & 1:
                r, v = r ^ pr, v ^ pv
        if r == 0:
            if v:
                return None, 0
            continue
        col = (r & -r).bit_length() - 1
        piv = [(c, pr ^ r, pv ^ v) if pr >> col & 1 else (c, pr, pv) for c, pr, pv in piv]
        piv.append((col, r, v))
    pcols = {c for c, _, _ in piv}
    free = [j for j in range(len(masks)) if j not in pcols]
    if len(free) > max_free:
        return None, 0
    x0 = sum(1 << c for c, _, pv in piv if pv)
    basis = [(1 << f) | sum(1 << c for c, pr, _ in piv if pr >> f & 1) for f in free]
    best, cur = x0, x0
    for i in range(1, 1 << len(basis)):  # Gray code: one basis vector flips per step
        cur ^= basis[(i & -i).bit_length() - 1]
        if bin(cur).count("1") < bin(best).count("1"):
            best = cur
    return best, 1 << len(basis)


def toggle_solve(game, level):
    """(path, candidates, secs) for a toggle level: the exact optimum, as the game's own click actions."""
    import runtime_click
    t0 = time.perf_counter()
    game.full_reset()
    if level > 0:
        game.set_level(level)
        game._score = level
    game._state = GameState.NOT_FINISHED
    w, h = game._w, game._h
    st, tgt = game._st["state"], game.current_level.get_data("target")
    b = sum(1 << (y * w + x) for y in range(h) for x in range(w) if st[y][x] != tgt[y][x])
    x, cands = gf2_min_clicks(_toggle_masks(w, h, game._cfg.get("rule", "self")), b, w * h)
    if x is None:
        return None, cands, time.perf_counter() - t0
    by_tile = {}
    for a in game._get_valid_clickable_actions():
        g = game.camera.display_to_grid(int(a.data["x"]), int(a.data["y"]))
        if g is not None:
            by_tile.setdefault((g[0] // runtime_click.CELL, (g[1] - runtime_click.HUD) // runtime_click.CELL), a)
    path = []
    for j in range(w * h):
        if x >> j & 1:
            if (j % w, j // w) not in by_tile:
                return None, cands, time.perf_counter() - t0
            path.append(_rec(by_tile[(j % w, j // w)]))
    return path, cands, time.perf_counter() - t0


def solve(game, level, max_nodes=50000, max_secs=120.0):
    """The level's optimum: exact algebra for toggle boards, else (or if the algebra declines) the BFS."""
    if getattr(game, "_cfg", {}).get("kind") == "toggle":
        path, cands, secs = toggle_solve(game, level)
        if path is not None:
            return path, cands, secs
    return oracle_solve(game, level, max_nodes, max_secs)


def solve_plan(make_game, plan, max_nodes=50000, max_secs=120.0):
    """Solve every level of a plan; returns [{level, optimal, nodes, secs, path}] (optimal None on failure)."""
    out = []
    game = make_game()
    for i in range(len(plan["levels"])):
        path, nodes, secs = solve(game, i, max_nodes, max_secs)
        out.append({"level": i + 1, "optimal": len(path) if path else None, "nodes": nodes, "secs": round(secs, 2),
                    "path": path})
        if path is None:
            break
    return out


def baselines(levels):
    return [int(math.ceil(RATIO * r["optimal"])) for r in levels]


def budgets(levels):
    return [BUDGET_RATIO * r["optimal"] for r in levels]


def replay(game, paths):
    """Replay oracle paths level by level on a fresh game; True if every level completes in order."""
    game.full_reset()
    for i, path in enumerate(paths):
        for a in path:
            fd = game.perform_action(_inp(a), raw=True)
            if fd.state == GameState.GAME_OVER:
                return False
        if not (fd.levels_completed == i + 1 or fd.state == GameState.WIN):
            return False
    return True
