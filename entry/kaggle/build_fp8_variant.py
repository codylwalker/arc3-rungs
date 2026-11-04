"""Build an FP8-PLE kernel from a built arm (run kaggle/build_kernels.py first).

Usage: python -I kaggle/build_fp8_variant.py <kaggle-username> [a|b] [--dprime] [--passes N]
Writes kaggle/kernels/arc3-twin-<arm>-fp8[-dprime]/: the arm unchanged except
- the target checkpoint, Albucino's W4A16 with an FP8 PLE table (124.8 GB in 25 shards against the Intel
  BF16-PLE 181.2 GB in 38; the launcher accepts either, reading the PLE dtype from config.json);
- --dprime: PRIORITY_FORM = "dprime" in the twin cell (Affectify's D' slot priority, priority_dprime.py);
- --passes N: the Save Version's demo plays each game N times (N x 10 games for the gate's 10 slots, so
  the slot priority is exercised; the competition rerun keeps its own n_passes = 1).
A separate slug, so a scheduled submission's READY check of another kernel's latest version is not disturbed.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {
    "a": ("arc3-twin-a-control", "arc3-twin-a-fp8", "arc3 twin a fp8", "**Arm A (control):**",
          "**Arm A (control), FP8-PLE checkpoint:**"),
    "b": ("arc3-twin-b-on", "arc3-twin-b-fp8", "arc3 twin b fp8", "**Arm B:**",
          "**Arm B, FP8-PLE checkpoint test:**"),
}
OLD_DIR = "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
NEW_DIR = "/kaggle/input/models/dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple/transformers/default/1"
OLD_SRC = "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1"
NEW_SRC = "dfranzen/albucino-qwen3.8-flash-next-w4a16-fp8ple/Transformers/default/1"
DEMO_PASSES = "\nbm.n_passes = 1\nbm.game_weights = None\n"
CONTEST_PASSES = "    bm.n_passes = 1\n"


def replace_once(cells, old, new, what):
    hits = [c for c in cells if old in "".join(c.get("source", []))]
    if len(hits) != 1 or "".join(hits[0]["source"]).count(old) != 1:
        raise SystemExit("expected %s exactly once in the notebook" % what)
    hits[0]["source"] = "".join(hits[0]["source"]).replace(old, new).splitlines(keepends=True)
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("arm", nargs="?", default="b", choices=sorted(ARMS))
    ap.add_argument("--dprime", action="store_true")
    ap.add_argument("--passes", type=int, default=1)
    args = ap.parse_args()
    src_slug, dst_slug, title, old_head, new_head = ARMS[args.arm]
    if args.dprime:
        dst_slug, title, new_head = dst_slug + "-dprime", title + " dprime", new_head[:-3] + ", D' slot priority:**"
    src = os.path.join(HERE, "kernels", src_slug)
    dst = os.path.join(HERE, "kernels", dst_slug)
    with open(os.path.join(src, src_slug + ".ipynb"), encoding="utf-8") as f:
        nb = json.load(f)
    with open(os.path.join(src, "kernel-metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cells = nb["cells"]
    replace_once(cells, OLD_DIR, NEW_DIR, "the target MODEL_DIR")
    note = "\nThis variant loads Albucino's W4A16-FP8PLE checkpoint (124.8 GB) instead of Intel's BF16-PLE one (181.2 GB), for a shorter model load.\n"
    if args.dprime:
        replace_once(cells, 'PRIORITY_FORM = ""\n', 'PRIORITY_FORM = "dprime"\n', "the twin cell's PRIORITY_FORM")
        note += ("\nThe admission gate's slot priority is D', the formula published by the Osaka Metropolitan University "
                 "team Affectify (shiiin9) in their notebook \"ARC-AGI-3 Milestone 2 Solution with a new slot priority "
                 "(D')\", implemented from its stated definition (priority_dprime.py).\n")
    if args.passes != 1:
        settings = replace_once(cells, DEMO_PASSES, "\nbm.n_passes = %d\nbm.game_weights = None\n" % args.passes,
                                "the demo's n_passes")
        if CONTEST_PASSES not in "".join(settings["source"]):
            raise SystemExit("the competition-rerun branch lost its n_passes = 1")
        note += "\nThe Save Version's demo plays each game %d times, so more games wait than the gate has slots.\n" % args.passes
    head = "".join(cells[0]["source"])
    if old_head not in head:
        raise SystemExit("header cell does not name %s" % old_head)
    cells[0]["source"] = (head.replace(old_head, new_head, 1) + note).splitlines(keepends=True)
    if meta["model_sources"].count(OLD_SRC) != 1:
        raise SystemExit("expected the Intel model source exactly once in the metadata")
    meta["model_sources"] = [NEW_SRC if s == OLD_SRC else s for s in meta["model_sources"]]
    meta.update(id="%s/%s" % (args.user, dst_slug), title=title, code_file=dst_slug + ".ipynb")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, dst_slug + ".ipynb"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    with open(os.path.join(dst, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
    print("wrote", dst, "| model_sources:", meta["model_sources"])


if __name__ == "__main__":
    main()
