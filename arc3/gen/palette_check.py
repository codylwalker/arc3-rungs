"""Legibility check over a generated set: does a game draw its agent (or another glyph it uses) in the
colour of an object it also draws? Reads CONFIG and LEVELS from each emitted file.

Usage (from tools/arc3): python gen/palette_check.py gen/out/set-a
"""
import ast
import os
import sys

# level characters -> the palette role that draws them (movement family)
CHAR_ROLE = {"#": "wall", "G": "goal", "X": "hazard", "K": "key", "D": "door", "B": "block", "P": "plate",
             "^": "oneway", ">": "oneway", "v": "oneway", "<": "oneway", "I": "ice", "C": "cross"}


def literal(src, name):
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    return None


def clashes(cfg, levels):
    """Colour clashes a player would see in a movement game: a glyph in use drawn in the colour of a role
    the game draws (or of another glyph in use), and the HUD band against the glyph and budget bar on it."""
    pal = cfg["palette"]
    drawn = {"bg", "band"}
    for lv in levels:
        for row in lv["rows"]:
            drawn.update(CHAR_ROLE[c] for c in row if c in CHAR_ROLE)
    if cfg.get("snake"):
        drawn.add("trail")
    n_glyph = cfg["state_flip"] if cfg.get("state_flip") else 1
    out = []
    for k in range(n_glyph):
        for role in sorted(drawn):
            if pal[role] == pal["glyph"][k]:
                out.append("glyph%d=%s" % (k, role))
        for j in range(k):
            if pal["glyph"][k] == pal["glyph"][j]:
                out.append("glyph%d=glyph%d" % (k, j))
    if any(lv.get("budget") for lv in levels) and pal["budget"] == pal["band"]:
        out.append("budget=band")
    if pal["wall"] == pal["bg"]:
        out.append("wall=bg")
    return out


def check(path):
    with open(path, encoding="utf-8") as f:
        src = f.read()
    cfg, levels = literal(src, "CONFIG"), literal(src, "LEVELS")
    if "rows" not in levels[0] or "kind" in cfg:
        return None  # click family: no glyphs, every role its own colour
    return cfg["mechanics"], clashes(cfg, levels)


def main():
    n = bad = agent_bad = 0
    for root, _dirs, files in os.walk(sys.argv[1]):
        for fn in files:
            if fn.endswith(".py") and fn[:-3] == os.path.basename(os.path.dirname(root)):
                r = check(os.path.join(root, fn))
                if r is None:
                    continue
                n += 1
                mech, found = r
                if found:
                    bad += 1
                    agent_bad += any(c.startswith("glyph0=") for c in found)
                    print(fn[:-3], ",".join(mech), " ".join(found))
    print("games %d, with a clash %d, agent colour = a drawn role %d" % (n, bad, agent_bad))


if __name__ == "__main__":
    main()
