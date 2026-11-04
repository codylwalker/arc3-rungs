"""Build the D' + M contest candidate from the D' kernel (2026-10-29).

Usage: python -I kaggle/build_m_variant.py <kaggle-username> [a|b]
From kernels/arc3-twin-<arm>-fp8-dprime (build_fp8_variant.py <arm> --dprime first) writes
kernels/arc3-twin-<arm>-fp8-dprime-m/: the notebook unchanged except, in the twin cell right after the
ARC3_TWIN_TOOLS line, the two M flags
    ARC3_AUTO_FRAME_DIFF=1   Franzen's own text diff of each action in the prompt
    ARC3_METERS=1            meters.py: a `meter:` line per budget bar in the opener, a `far:` note per action
so they apply to the Save Version's demo and to the competition rerun alike (the harness reads them at
import). Arm M on the generated games (2026-10-28): tokens per level -7%, steps -10%, levels a tie; the
hidden run is token-bound per game (every D' draw runs to the 532-min clock), so fewer tokens a level is
the lever. The dataset arc3-twin-tools must carry the fixed meters.py (version of 2026-10-28 22:46 UTC).
A separate slug: the scheduled submission keeps submitting its own kernel and version.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = {"a": "arc3-twin-a-fp8-dprime", "b": "arc3-twin-b-fp8-dprime"}
TWIN_FLAG_LINE = 'os.environ["ARC3_TWIN_TOOLS"] = TWIN_TOOLS\n'
M_FLAGS = (("ARC3_AUTO_FRAME_DIFF", "1"), ("ARC3_METERS", "1"))


def find_cell(cells, marker, what):
    hits = [i for i, c in enumerate(cells) if marker in "".join(c.get("source", []))]
    if len(hits) != 1:
        raise SystemExit("expected %s exactly once in the notebook, found %d" % (what, len(hits)))
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("user")
    ap.add_argument("arm", nargs="?", default="a", choices=sorted(ARMS))
    args = ap.parse_args()
    src_slug = ARMS[args.arm]
    dst_slug = src_slug + "-m"
    src, dst = os.path.join(HERE, "kernels", src_slug), os.path.join(HERE, "kernels", dst_slug)
    if not os.path.isdir(src):
        raise SystemExit("%s missing: build it first (build_fp8_variant.py %s --dprime)" % (src, args.arm))
    with open(os.path.join(src, src_slug + ".ipynb"), encoding="utf-8") as f:
        nb = json.load(f)
    with open(os.path.join(src, "kernel-metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    cells = nb["cells"]
    i = find_cell(cells, TWIN_FLAG_LINE, "the twin cell's ARC3_TWIN_TOOLS line")
    text = "".join(cells[i]["source"])
    if text.count(TWIN_FLAG_LINE) != 1:
        raise SystemExit("expected the ARC3_TWIN_TOOLS line once in its cell")
    for key, _ in M_FLAGS:
        if key in text:
            raise SystemExit("%s already set in the twin cell" % key)
    extra = "".join("os.environ[%r] = %r  # build_m_variant: the M flags (diff and meters)\n" % kv for kv in M_FLAGS)
    cells[i]["source"] = text.replace(TWIN_FLAG_LINE, TWIN_FLAG_LINE + extra).splitlines(keepends=True)
    head = "".join(cells[0]["source"])
    note = ("\n**D' + M candidate (kaggle/build_m_variant.py, arm %s): the D' kernel with Franzen's text diff and the "
            "meters lines on (%s), for the demo and the competition rerun alike. Built 2026-10-29 after arm M on the "
            "generated games (tokens per level -7%%, levels a tie).**\n" % (args.arm, ", ".join("%s=%s" % kv for kv in M_FLAGS)))
    cells[0]["source"] = (head + note).splitlines(keepends=True)
    meta.update(id="%s/%s" % (args.user, dst_slug), title=meta["title"] + " m", code_file=dst_slug + ".ipynb")
    if "%s/arc3-twin-tools" % args.user not in meta["dataset_sources"]:
        raise SystemExit("the source kernel does not attach arc3-twin-tools")
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, dst_slug + ".ipynb"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    with open(os.path.join(dst, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
    print("wrote", dst, "| title", meta["title"], "| flags", M_FLAGS, "| twin cell", i)


if __name__ == "__main__":
    main()
