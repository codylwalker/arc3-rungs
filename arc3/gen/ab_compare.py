"""Paired A/B over harness runs (the plan's Phase A metric): per arm, levels solved, tokens per level
solved and the local score; per game, the B - A differences with a bootstrap 95% interval over games and
a sign test; per mechanic for generated games; and, when both arms also ran public games, whether the
public anchor moves the same way as the generated set.

Usage: python -I gen/ab_compare.py --a RUN_DIR [RUN_DIR ...] --b RUN_DIR [...] [--set gen/out/set-a ...]
       [--json out.json]
A RUN_DIR is a harness output dir holding summary.txt (per-game means across passes) and, optionally,
artifacts/*_viewer_data.json (per pass). Several dirs per arm are merged (games weighted by passes).
Standard library only; reads the run dirs as data.
"""
import argparse
import glob
import json
import math
import os
import random
import re

PER_GAME = re.compile(r"^\s+(\S+): score=([\d.]+), levels=([\d.]+)/(\d+), actions=(\d+), tokens=(\d+)", re.M)
PASSES = re.compile(r"^passes:\s+(\d+)", re.M)


def load_arm(dirs):
    """{game_id: {score, levels, total_levels, actions, tokens, passes, pass_levels}} merged over dirs."""
    games = {}
    for d in dirs:
        summ = os.path.join(d, "summary.txt")
        if not os.path.isfile(summ):
            hits = glob.glob(os.path.join(d, "**", "summary.txt"), recursive=True)
            if len(hits) != 1:
                raise SystemExit("%s: expected one summary.txt, found %d" % (d, len(hits)))
            summ = hits[0]
        with open(summ, encoding="utf-8", errors="replace") as f:
            text = f.read()
        m = PASSES.search(text)
        passes = int(m.group(1)) if m else 1
        per_pass = {}
        for vd in glob.glob(os.path.join(os.path.dirname(summ), "artifacts", "*_viewer_data.json")):
            with open(vd, encoding="utf-8", errors="replace") as f:
                v = json.load(f)
            per_pass.setdefault(v["game_id"], []).append(v.get("levels_completed", 0))
        for gid, score, lv, tot, acts, toks in PER_GAME.findall(text):
            rec = {"score": float(score), "levels": float(lv), "total_levels": int(tot), "actions": float(acts),
                   "tokens": float(toks), "passes": passes, "pass_levels": per_pass.get(gid, [])}
            if gid in games:  # the same game in two runs of one arm: weight by passes
                old = games[gid]
                w0, w1 = old["passes"], passes
                for k in ("score", "levels", "actions", "tokens"):
                    old[k] = (old[k] * w0 + rec[k] * w1) / (w0 + w1)
                old["passes"] = w0 + w1
                old["pass_levels"] += rec["pass_levels"]
            else:
                games[gid] = rec
    return games


def set_meta(set_dirs):
    """{game_id: mechanics} from generated sets' metadata.json files."""
    out = {}
    for s in set_dirs:
        for mp in glob.glob(os.path.join(s, "*", "*", "metadata.json")):
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
            out[m["game_id"]] = m.get("mechanics", [])
    return out


def pair(a, b):
    """Games present in both arms: full ids first, then the 4-char base id (a re-emitted game's hash moves)."""
    out = []
    b_by_base = {}
    for g in b:
        b_by_base.setdefault(g.split("-")[0], []).append(g)
    for g in sorted(a):
        if g in b:
            out.append((g, g))
        elif len(b_by_base.get(g.split("-")[0], [])) == 1:
            out.append((g, b_by_base[g.split("-")[0]][0]))
    return out


def boot_ci(xs, n=4000, seed=0):
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def sign_test(xs):
    """(positives, negatives, two-sided exact binomial p over the non-zero differences)."""
    pos, neg = sum(x > 0 for x in xs), sum(x < 0 for x in xs)
    n, k = pos + neg, min(pos, neg)
    if n == 0:
        return pos, neg, 1.0
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return pos, neg, min(1.0, 2 * p)


def arm_totals(arm, ids):
    lv = sum(arm[g]["levels"] for g in ids)
    tok = sum(arm[g]["tokens"] for g in ids)
    return {"games": len(ids), "levels_solved": round(lv, 2), "levels_total": sum(arm[g]["total_levels"] for g in ids),
            "mean_score": round(sum(arm[g]["score"] for g in ids) / max(1, len(ids)), 2), "tokens": int(tok),
            "tokens_per_level_solved": round(tok / lv) if lv else None}


def compare(a, b, pairs):
    d = {k: [b[gb][k] - a[ga][k] for ga, gb in pairs] for k in ("levels", "score", "tokens")}
    out = {"games": len(pairs), "arm_a": arm_totals(a, [ga for ga, _ in pairs]),
           "arm_b": arm_totals(b, [gb for _, gb in pairs])}
    for k, xs in d.items():
        lo, hi = boot_ci(xs)
        pos, neg, p = sign_test(xs)
        out["delta_" + k] = {"mean": round(sum(xs) / max(1, len(xs)), 3), "ci95": [round(lo, 3), round(hi, 3)],
                             "b_better" if k != "tokens" else "b_more": pos, "a_better" if k != "tokens" else "a_more": neg,
                             "sign_p": round(p, 4)}
    return out


def report(name, r):
    print("== %s: %d paired games" % (name, r["games"]))
    for arm in ("arm_a", "arm_b"):
        t = r[arm]
        print("  %s  levels %s/%d  mean score %.2f  tokens %d  tokens per level solved %s" % (
            arm[-1].upper(), t["levels_solved"], t["levels_total"], t["mean_score"], t["tokens"], t["tokens_per_level_solved"]))
    for k in ("levels", "score", "tokens"):
        x = r["delta_" + k]
        keys = ("b_better", "a_better") if k != "tokens" else ("b_more", "a_more")
        print("  B-A %-6s mean %+.3f  95%% CI [%+.3f, %+.3f]  per game: B %d, A %d  sign p %.4f" % (
            k, x["mean"], x["ci95"][0], x["ci95"][1], x[keys[0]], x[keys[1]], x["sign_p"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="generated set dirs, for mechanics and the gen/public split")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    a, b = load_arm(args.a), load_arm(args.b)
    meta = set_meta(args.set)
    gen_base = {g.split("-")[0] for g in meta}
    pairs = pair(a, b)
    if not pairs:
        raise SystemExit("no game is in both arms")
    unpaired = sorted(set(a) - {x for x, _ in pairs}) + sorted(set(b) - {y for _, y in pairs})
    is_gen = lambda g: g in meta or g.split("-")[0] in gen_base  # noqa: E731
    groups = {"all": pairs}
    if meta:
        groups["generated"] = [p for p in pairs if is_gen(p[0])]
        groups["public"] = [p for p in pairs if not is_gen(p[0])]
    out = {"unpaired": unpaired}
    for name, ps in groups.items():
        if ps:
            out[name] = compare(a, b, ps)
            report(name, out[name])
    if unpaired:
        print("unpaired (in one arm only): %s" % ", ".join(unpaired))
    if meta and groups.get("generated"):
        mech = {}
        for ga, gb in groups["generated"]:
            ms = meta.get(ga) or next((v for k, v in meta.items() if k.split("-")[0] == ga.split("-")[0]), [])
            for m in ms:
                mech.setdefault(m, []).append(b[gb]["levels"] - a[ga]["levels"])
        out["by_mechanic"] = {m: {"games": len(xs), "delta_levels_mean": round(sum(xs) / len(xs), 3)}
                              for m, xs in sorted(mech.items())}
        print("  B-A levels by mechanic: " + "; ".join("%s %+.2f (%d)" % (m, v["delta_levels_mean"], v["games"])
                                                     for m, v in out["by_mechanic"].items()))
        if groups.get("public") and "public" in out:
            g, p = out["generated"]["delta_levels"]["mean"], out["public"]["delta_levels"]["mean"]
            agree = (g > 0) == (p > 0) if g and p else None
            out["direction_agreement"] = agree
            print("  direction agreement, levels (generated %+.3f vs public %+.3f): %s" % (g, p, agree))
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
