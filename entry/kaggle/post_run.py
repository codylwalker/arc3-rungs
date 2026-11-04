"""After a Save Version run: fetch what the checks need, check it, summarise. Never submits.

Usage: python -I kaggle/post_run.py --out <dir> [--cap-mb 600] [--version N] <user/slug> [...]
Needs KAGGLE_CONFIG_DIR (the token) and, if the CLI is not on PATH, --kaggle <path to kaggle.exe>.

Per kernel, into <out>/<slug>/: the run status, the execution log (`kaggle kernels logs`), the output
listing (`kaggle kernels files`), and only the small outputs the checks need (`*_viewer_data.json`,
`*_events.jsonl`, `diagnostics.html`; never `requests.jsonl`), under --cap-mb, so the download
stays small. Then the checks: the run completed; our cell printed `twin_tools wired` with the arm's flag;
Franzen's patch applied; `submission.parquet` exists in the output; Tracebacks counted; the model server
never crashed (`serve.log`); and, when events came down, tools/analyze_run.py. Writes
<out>/<slug>/summary.json and prints one verdict per kernel: READY (submit this version) or NOT_READY
(reasons). A generated-games kernel (`-gen` in its slug) is never READY: it is a Save Version only.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import analyze_run  # noqa: E402

WANT = re.compile(r"(_viewer_data\.json|_events\.jsonl|diagnostics\.html|serve\.log|summary\.txt)$")
# a data session's SFT records (tools/traj_to_sft.py, run by the notebook's last cell), with --sft
WANT_SFT = re.compile(r"(^|/)sft/[^/]+\.(jsonl\.gz|json)$")
EXPECTED_FLAG = {"arc3-twin-a-control": "0", "arc3-twin-b-on": "1", "arc3-twin-a-fp8": "0", "arc3-twin-b-fp8": "1",
                 "arc3-twin-a-fp8-dprime": "0", "arc3-twin-data-dprime": "0", "arc3-twin-b-fp8-dprime": "1",
                 "arc3-twin-a-fp8-dprime-m": "0",  # the D' + M candidate (build_m_variant.py, 2026-10-29)
                 # the generated-games evaluation kernels (build_gen_variant.py), Save Version only
                 "arc3-twin-a-fp8-dprime-gen": "0", "arc3-twin-b-fp8-dprime-gen": "1",
                 "arc3-twin-a-fp8-dprime-gen-goal": "0",  # the goal-rung pairs, five passes (2026-10-30, tools off)
                 # the structural arms of 2026-10-28 (build_gen_variant.py --suffix): C certifies the model's
                 # own predict unasked (tools on), M adds the text diff and the meters (tools off)
                 "arc3-twin-b-fp8-dprime-gen-c": "1", "arc3-twin-a-fp8-dprime-gen-m": "0"}


def cli(kaggle, *args, timeout=600):
    env = dict(os.environ)
    # Under Task Scheduler `~` is the real home, where the CLI finds no access_token: hand it over.
    token_file = os.path.join(env.get("KAGGLE_CONFIG_DIR", ""), "access_token")
    if "KAGGLE_API_TOKEN" not in env and os.path.isfile(token_file):
        with open(token_file, encoding="utf-8") as f:
            env["KAGGLE_API_TOKEN"] = f.read().strip()
    p = subprocess.run([kaggle] + list(args), capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, env=env,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return p.returncode, p.stdout, p.stderr


def log_text(raw):
    """`kaggle kernels logs` prints a JSON list of {stream_name, time, data}; fall back to raw text."""
    try:
        entries = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(entries, list):
        return "".join(str(e.get("data", "")) if isinstance(e, dict) else str(e) for e in entries)
    return raw


def server_crash(serve):
    """The time of the model server's first fatal line in serve.log, or None. On 2026-10-30 SGLang's scheduler
    raised an IndexError 2.3 minutes into the goal-rung run, and the harness played on to its budget against a
    dead port (11 actions in 8 h 40 min)."""
    # [0-9] rather than the digit class around the colons: the release scanner reads a backslash-d before a colon
    # and a backslash as a drive path
    m = re.search(r"^\[(\d{4}-\d{2}-\d{2} [0-9]{2}:[0-9]{2}:[0-9]{2})\] (?:Scheduler hit an exception|SIGQUIT received)",
                  serve, re.M)
    return m.group(1) if m else None


def listing(kaggle, kernel):
    # A page holds 200 entries and a generated-games run lists 340-450 files: page 1 alone dropped
    # serve.log, summary.txt and submission.parquet from every one (found 2026-11-04). The CLI prints
    # "Next Page Token = <token>" ahead of the JSON while more pages follow.
    files, token = [], None
    for _ in range(50):
        args = ["kernels", "files", kernel, "--format", "json", "--page-size", "200"]
        if token:
            args += ["--page-token", token]
        code, out, err = cli(kaggle, *args)
        page, error = parse_listing(out, err)
        if error:
            return files, error
        files.extend(page)
        token = next_page_token(out)
        if not token:
            break
    return files, None


def next_page_token(out):
    m = re.search(r"^Next Page Token = (\S+)\s*$", out, re.M)
    return m.group(1) if m else None


def parse_listing(out, err=""):
    # The CLI can print an "outdated kaggle version" warning ahead of the JSON (seen under Task
    # Scheduler, 2026-10-23): parse from the first bracket.
    starts = [i for i in (out.find("["), out.find("{")) if i >= 0]
    try:
        data = json.loads(out[min(starts):]) if starts else json.loads(out)
    except ValueError:
        return [], (out + err)[-2000:]
    items = data if isinstance(data, list) else data.get("files") or data.get("items") or []
    files = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name") or it.get("fileName") or it.get("ref") or ""
        size = it.get("size") or it.get("totalBytes") or it.get("fileSize") or 0
        try:
            size = int(size)
        except (TypeError, ValueError):
            m = re.match(r"([\d.]+)\s*([KMG]?B)", str(size))
            size = int(float(m.group(1)) * {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9}[m.group(2)]) if m else 0
        files.append({"name": name, "size": size})
    return files, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-mb", type=float, default=600.0)
    ap.add_argument("--version", default=None, help="kernel version to read (default: latest)")
    ap.add_argument("--kaggle", default="kaggle")
    ap.add_argument("--sft", action="store_true", help="also fetch sft/*.jsonl.gz and sft/index.json (a data session)")
    ap.add_argument("kernels", nargs="+")
    args = ap.parse_args()
    want = (lambda n: WANT.search(n) or WANT_SFT.search(n)) if args.sft else WANT.search
    verdicts = []
    for kernel in args.kernels:
        slug = kernel.split("/")[-1]
        ref = kernel if args.version is None else "%s/%s" % (kernel, args.version)
        d = os.path.join(args.out, slug)
        os.makedirs(d, exist_ok=True)
        summary = {"kernel": ref}
        code, out, err = cli(args.kaggle, "kernels", "status", kernel)
        status = re.search(r'status "([^"]+)"', out)
        summary["status"] = status.group(1) if status else (out + err).strip()[-300:]
        code, out, err = cli(args.kaggle, "kernels", "logs", kernel, timeout=900)
        text = log_text(out)
        with open(os.path.join(d, "run.log"), "w", encoding="utf-8") as f:
            f.write(text)
        wired = re.findall(r"twin_tools wired from .*?ARC3_TWIN_TOOLS = (\d)", text)
        summary.update({
            "keepalive_resets": len(re.findall(r"keep-alive RESET \d+/\d+", text)),
            "keepalive_stop": (re.findall(r"keep-alive stops: ([^\n]*)", text) or [None])[-1],
            "log_bytes": len(text),
            "twin_wired_flag": wired[-1] if wired else None,
            "harness_patch_applied": "harness patch applied successfully" in text,
            # the log keeps escaped line ends (\n"}), so stop at a backslash or quote too
            "slot_priority": (re.findall(r"\| slot priority ([A-Za-z']+)", text) or [None])[-1],
            "dprime_active": (re.findall(r"priority_dprime: D' slot priority active; ([^\n\\\"]*)", text) or [None])[0],
            "tracebacks": text.count("Traceback (most recent call last)"),
            "log_tail": text[-1500:],
        })
        files, list_error = listing(args.kaggle, kernel)
        with open(os.path.join(d, "files.json"), "w", encoding="utf-8") as f:
            json.dump(files, f, indent=1)
        summary["output_files"] = len(files)
        summary["output_mb"] = round(sum(x["size"] for x in files) / 1e6, 1)
        summary["has_submission_parquet"] = any(x["name"].endswith("submission.parquet") for x in files)
        if list_error:
            summary["listing_error"] = list_error
        chosen, total = [], 0
        for x in sorted((x for x in files if want(x["name"])), key=lambda x: x["size"]):
            if total + x["size"] > args.cap_mb * 1e6:
                continue
            chosen.append(x["name"])
            total += x["size"]
        summary["downloaded"] = len(chosen)
        summary["downloaded_mb"] = round(total / 1e6, 1)
        if chosen:
            pattern = "|".join(re.escape(os.path.basename(n)) + "$" for n in chosen)
            cli(args.kaggle, "kernels", "output", kernel, "-p", os.path.join(d, "output"), "-o", "-q",
                "--file-pattern", pattern, timeout=1800)
            games = analyze_run.analyse(os.path.join(d, "output"))
            summary["analysis"] = analyze_run.summarise(games)
            for dirpath, _, names in os.walk(os.path.join(d, "output")):
                if "summary.txt" in names:
                    with open(os.path.join(dirpath, "summary.txt"), encoding="utf-8", errors="replace") as f:
                        stext = f.read()
                    m = re.search(r"total tokens:\s*(\d+)", stext)
                    summary["total_tokens"] = int(m.group(1)) if m else None
                    m = re.search(r"mean score:\s*([\d.]+)", stext)
                    summary["demo_summary_mean"] = float(m.group(1)) if m else None
                if "serve.log" in names:
                    with open(os.path.join(dirpath, "serve.log"), encoding="utf-8", errors="replace") as f:
                        serve = f.read().replace("\r", "\n")
                    loads = re.findall(r"Loading safetensors checkpoint shards:\s*\d+% Completed \| (\d+)/(\d+) \[([^\]<]+)", serve)
                    # The main model's shards, not the 2-shard MTP drafter that loads after it.
                    main = [l for l in loads if int(l[1]) == max(int(t) for _, t, _ in loads)]
                    summary["weights_loaded"] = "%s/%s in %s" % main[-1] if main else None
                    summary["server_ready"] = bool(re.search(r"fired up|Application startup complete|Uvicorn running", serve))
                    summary["server_crash"] = server_crash(serve)
            done = [g for g in summary["analysis"]["per_game"] if g.get("levels") is not None]
            summary["demo_games"] = len(done)
            summary["demo_levels"] = sum(int(g["levels"] or 0) for g in done)
            scores = [float(g["score"]) for g in done if g.get("score") is not None]
            summary["demo_mean_score"] = round(sum(scores) / len(scores), 3) if scores else None
        reasons = []
        if "COMPLETE" not in str(summary["status"]).upper():
            reasons.append("status %s" % summary["status"])
        # not `want`: that name is the file filter, still needed for the next kernel
        expected = EXPECTED_FLAG.get(slug)
        if expected is not None and summary["twin_wired_flag"] != expected:
            reasons.append("twin cell flag %s, expected %s" % (summary["twin_wired_flag"], expected))
        if not summary["harness_patch_applied"]:
            reasons.append("Franzen's harness patch line missing from the log")
        if not summary["has_submission_parquet"]:
            reasons.append("no submission.parquet in the output")
        if not summary.get("total_tokens"):
            reasons.append("the model generated no tokens (weights loaded %s; server ready %s)" % (
                summary.get("weights_loaded"), summary.get("server_ready")))
        if summary.get("slot_priority") == "dprime" and not summary.get("dprime_active"):
            reasons.append("D' slot priority requested but it never scored a snapshot")
        if summary.get("server_crash"):
            reasons.append("the model server crashed at %s UTC (serve.log)" % summary["server_crash"])
        if "-gen" in slug:
            reasons.append("a generated-games kernel: a Save Version only, never submitted")
        summary["verdict"] = "READY" if not reasons else "NOT_READY"
        summary["reasons"] = reasons
        with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        line = "%s %s | status %s | flag %s | parquet %s | tokens %s | weights %s | tracebacks %d | demo games %s levels %s mean %s" % (
            summary["verdict"], slug, summary["status"], summary["twin_wired_flag"],
            summary["has_submission_parquet"], summary.get("total_tokens"), summary.get("weights_loaded"),
            summary["tracebacks"], summary.get("demo_games"), summary.get("demo_levels"), summary.get("demo_mean_score"))
        if reasons:
            line += " | " + "; ".join(reasons)
        if summary.get("analysis"):
            line += " | tool calls %s" % summary["analysis"]["calls"]
        print(line, flush=True)
        verdicts.append(summary["verdict"])
    return 0 if all(v == "READY" for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
