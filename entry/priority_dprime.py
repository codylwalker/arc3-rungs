"""D' slot priority for Daniel Franzen's Milestone-2 scheduler, written from its published definition.

The formula is the Osaka Metropolitan University team Affectify's (shiiin9), stated in full in their
public Kaggle notebook "ARC-AGI-3 Milestone 2 Solution with a new slot priority (D')" (2026-10-04,
"31.54 in a Single Sub"). This module implements that statement; it does not copy their code.

    priority = A * M * C + B * phi

    A    depth and action efficiency: (1 + 0.5 (l - 1)) * norm(N) * (300 / (300 + a)) ** 2.5,
         norm(N) = 55 / (N (N + 1) / 2), N clamped to 6..10 as in Franzen's code
    M    pace: clip((30000 / p) ** 0.4, 0.25, 4); 1 before the game's first clear
    C    patience: 0.1 * max(0.1, 1 - a / 115) + 0.9 * max(0.1, 1 - t / T),
         T = 225000 * clip(p / 30000, 0.5, 2) ** 0.5
    B    room to grow: 16, 14, 10 or 0 when 3 or more, 2, 1 or 0 levels remain after the current one
    phi  the harness's tail fade, 1 -> 0 over the last 40% of the run (None = no fade)

l is the current level (from 1), N the game's level count, a and t the actions and generated tokens
spent on the current level, p the mean generated tokens per level cleared so far in this game.
Returns the harness's integer scale (value * 100, at least 1), as Franzen's priority_value does.
"""

N_MIN, N_MAX = 6, 10
B_BY_REMAINING = (0.0, 10.0, 14.0, 16.0)  # 0, 1, 2, 3+ levels left after the current one


def _clip(x, lo, hi):
    return min(hi, max(lo, x))


def dprime_value(level, actions, tokens, total_levels=None, tokens_per_level=None, tail_fraction=None):
    n = int(_clip(int(total_levels) if total_levels else N_MAX, N_MIN, N_MAX))
    level = max(1, int(level))
    a = max(0.0, float(actions))
    t = max(0.0, float(tokens))
    p = float(tokens_per_level) if tokens_per_level and tokens_per_level > 0 else None
    norm = 55.0 / (n * (n + 1) / 2.0)
    depth = (1.0 + 0.5 * (level - 1)) * norm * (300.0 / (300.0 + a)) ** 2.5
    pace = _clip((30000.0 / p) ** 0.4, 0.25, 4.0) if p else 1.0
    patience_tokens = 225000.0 * _clip((p or 30000.0) / 30000.0, 0.5, 2.0) ** 0.5
    patience = 0.1 * max(0.1, 1.0 - a / 115.0) + 0.9 * max(0.1, 1.0 - t / patience_tokens)
    room = B_BY_REMAINING[min(3, max(0, n - level))]
    phi = 1.0 if tail_fraction is None else _clip(float(tail_fraction), 0.0, 1.0)
    return depth * pace * patience + room * phi


def dprime_priority(snapshot, tail_fraction=None):
    """The harness's PrioritySnapshot (level, actions, tokens, cost_multiplier, total_levels,
    tokens_per_level) scored by D'. cost_multiplier, Franzen's own pace, is not used."""
    value = dprime_value(snapshot.level, snapshot.actions, snapshot.tokens,
                         getattr(snapshot, "total_levels", None),
                         getattr(snapshot, "tokens_per_level", None), tail_fraction)
    return max(1, int(round(100.0 * value)))
