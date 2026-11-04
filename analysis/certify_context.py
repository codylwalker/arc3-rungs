"""Arm C read: per game steps/actions/tokens against b, and every [certify] line with what the model did next.

python -I analysis/certify_context.py <analyze-c.json> <analysis-b.json> <c output dir>
"""
import glob
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
c_an, b_an, out_dir = sys.argv[1:4]
C = {r["game"]: r for r in json.load(open(c_an, encoding="utf-8"))["per_game"]}
B = {r["game"]: r for r in json.load(open(b_an, encoding="utf-8"))["per_game"]}
print("game           lv C/b  steps C/b  actions C/b  pycalls C/b  certify(all)")
tc = tb = 0
for g in sorted(C):
    c, b = C[g], B.get(g, {})
    tc += c["steps"]; tb += b.get("steps", 0)
    flag = " <-- C unfinished" if (c["levels"] or 0) < 6 else ""
    print("%-14s %s/%s   %3d/%-3d    %3d/%-3d     %3d/%-3d      %d(%d)%s" % (
        g[:13], c["levels"], b.get("levels"), c["steps"], b.get("steps", 0), c["actions"], b.get("actions", 0),
        c["python_calls"], b.get("python_calls", 0), c["certify_lines"], c["certify_full"], flag))
print("total analysis steps C %d, b %d" % (tc, tb))

CERT = re.compile(r"\[certify\][^\n]*")
HEAD = re.compile(r"^\[(THINKING|ASSISTANT|TOOL CALL: python)\]\s*$", re.M)
print("\n== certify lines and the model's next turn")
for path in sorted(glob.glob(os.path.join(out_dir, "**", "*_events.jsonl"), recursive=True)):
    game = os.path.basename(path).split("_")[0]
    events = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    analyses = [e for e in events if e.get("type") == "analysis"]
    seen = set()
    for i, ev in enumerate(analyses):
        tr = str(ev.get("transcript") or "")
        for m in CERT.finditer(tr):
            text = m.group(0)
            if text in seen:
                continue
            seen.add(text)
            # the next model turn in the following event, if any: its THINKING's first 240 chars
            nxt = ""
            if i + 1 < len(analyses):
                t2 = str(analyses[i + 1].get("transcript") or "")
                pos = t2.rfind(text)
                after = t2[pos + len(text):] if pos >= 0 else ""
                k = after.find("[THINKING]")
                if k >= 0:
                    nxt = " ".join(after[k + 10:k + 400].split())
            print("%s L%s step %d: %s\n    next: %s" % (game, ev.get("level"), i + 1, text[:220], nxt[:300]))
