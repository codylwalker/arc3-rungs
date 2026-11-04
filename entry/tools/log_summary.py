"""Recover a run's summary.txt from its execution log when the output has none.

Usage: python -I tools/log_summary.py <run.log> <out summary.txt>
The log is `kaggle kernels logs` output: either a JSON list of {stream_name, time, data} entries or plain
text. The harness prints its summary block ("benchmark: … per-game (mean across passes): …") after every
checkpoint and at the end; the LAST block is written, which ab_compare.load_arm and harness_report read.
"""
import json
import re
import sys


def log_text(raw):
    # the CLI can print an "outdated kaggle version" warning ahead of the JSON: parse from the first bracket
    start = raw.find("[")
    try:
        entries = json.loads(raw[start:] if start >= 0 else raw)
    except ValueError:
        return raw
    if isinstance(entries, list):
        return "".join(str(e.get("data", "")) for e in entries if isinstance(e, dict))
    return raw


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8", errors="replace") as f:
        text = log_text(f.read())
    starts = [m.start() for m in re.finditer(r"^benchmark:\s", text, re.M)]
    if not starts:
        raise SystemExit("no summary block in the log")
    block = text[starts[-1]:]
    end = re.search(r"\n\n", block[block.find("per-game"):])
    block = block[: block.find("per-game") + end.start()] if end else block
    with open(dst, "w", encoding="utf-8") as f:
        f.write(block.rstrip() + "\n")
    print("wrote", dst, "| %d per-game lines" % len(re.findall(r"^\s+\S+: score=", block, re.M)))


if __name__ == "__main__":
    main()
