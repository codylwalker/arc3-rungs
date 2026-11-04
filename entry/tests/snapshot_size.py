"""Bytes per engine snapshot (pickled deepcopy) for a few public games, to size the oracle probe.

Run with the toolkit's venv: <venv>/python -I tests/snapshot_size.py --env-dir <environment_files> [game ...]
"""
import argparse
import copy
import tempfile
import time
import tracemalloc

from arc_agi import Arcade, OperationMode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", required=True)
    ap.add_argument("games", nargs="*", default=["tu93", "ls20"])
    args = ap.parse_args()
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=args.env_dir,
                 recordings_dir=tempfile.mkdtemp(prefix="twin_rec_"))
    for game in args.games:
        env = arc.make(game, seed=0)
        env.reset()
        t0 = time.perf_counter()
        for _ in range(5):
            copy.deepcopy(env)
        per_copy = (time.perf_counter() - t0) / 5
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        kept = [copy.deepcopy(env) for _ in range(10)]
        size = (tracemalloc.get_traced_memory()[0] - base) / len(kept)
        tracemalloc.stop()
        print(game, "deepcopy %.1f ms" % (per_copy * 1000), "~%.2f MB per snapshot" % (size / 1e6), flush=True)


if __name__ == "__main__":
    main()
