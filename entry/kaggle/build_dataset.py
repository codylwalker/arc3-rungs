"""Build the Kaggle dataset folder for twin_tools from the current sources.

Usage: python -I kaggle/build_dataset.py <kaggle-username> [out-dir]
Writes <out-dir> (default kaggle/arc3-twin-tools/) with twin_tools.py, twin_prompt.py, apply_twin.py
(flat: apply_twin.py finds the modules beside itself) and dataset-metadata.json for the Kaggle CLI.
Upload it PRIVATE: `kaggle datasets create -p <out-dir>` (new) or
`kaggle datasets version -p <out-dir> -m "<note>"` (update).
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FILES = [("twin_tools.py", ROOT), ("twin_prompt.py", ROOT), ("twin_keepalive.py", ROOT),
         ("priority_dprime.py", ROOT), ("meters.py", ROOT), ("apply_twin.py", os.path.join(ROOT, "harness")),
         # the data-session kernel's last cell turns the demo's trajectories into SFT records
         ("traj_to_sft.py", os.path.join(ROOT, "tools"))]


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    user = sys.argv[1]
    out = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "arc3-twin-tools"))
    os.makedirs(out, exist_ok=True)
    for name, src in FILES:
        shutil.copyfile(os.path.join(src, name), os.path.join(out, name))
    meta = {
        "title": "arc3-twin-tools",
        "id": "%s/arc3-twin-tools" % user,
        # our own code, MIT-0 (see LICENSE)
        "licenses": [{"name": "CC0-1.0"}],
    }
    with open(os.path.join(out, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("wrote", out, sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
