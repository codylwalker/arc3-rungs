# ## 1b. twin_tools: CPU world-model checks and planning for the model's Python tool
# Insert this cell right after section 1 ("Environment and submission mode"), which applies the
# harness patch and sets the ARC3_* flags. It wires twin_tools into the patched harness and turns
# it on or off; the control arm keeps the wiring and sets TWIN_TOOLS = "0".
TWIN_TOOLS = "1"
# "" keeps Franzen's slot priority; "dprime" swaps in Affectify's D' (priority_dprime.py, their 40% fade).
PRIORITY_FORM = ""

_twin_apply = next(Path("/kaggle/input").rglob("apply_twin.py"), None)
if _twin_apply is None:
    raise RuntimeError("attach the arc3-twin-tools dataset (Add Input -> Your Work -> Datasets)")
subprocess.run([sys.executable, "-I", str(_twin_apply), str(BUNDLE_DIR / "src" / "ARC3-Inference")], check=True)
os.environ["ARC3_TWIN_TOOLS"] = TWIN_TOOLS

# Both arms: tolerate a slow model load. On 2026-10-22 the weights came off the model mount at
# 50-92 s a shard (38 shards), so the server was still loading 30 minutes in, while the harness gave
# each game's first request 900 s of grace and dropped a game after 10 failed requests: every game was
# abandoned with zero tokens. 7200 s of grace covers a load of ~2 h (the forum reported loads of over
# an hour on 2026-10-24); 30 failures is the spin allowance if the server dies mid-run. A fast load
# (~10 min, as in Franzen's runs) is unaffected.
os.environ["ARC3_HTTP_RETRY_INITIAL_SECONDS"] = "7200"
os.environ["ARC3_MAX_ANALYZER_FAILURES"] = "30"
# Both arms: keep the competition run alive during that load. "If you don't do any move for 15
# minutes, the run will end with the score so far" (the competition forum, 2026-10-17); our model
# took 35-40 min in some sessions (others report 35-63), so both 2026-10-22/23 submissions ended
# ~27 min in with 0.00. The warmup game repeats its RESET every 5 min until the server's /health
# answers (twin_keepalive.py): the interval another team validated on a scored run, 2026-10-22.
os.environ["ARC3_KEEPALIVE_SECONDS"] = "300"
# 24 RESETs = 2 h, matching the grace (the default 12 covered a load of 75 min at most).
os.environ["ARC3_KEEPALIVE_MAX"] = "24"
# The admission gate (10 streams for 110 games) hands a freed slot to the waiting game with the highest
# priority. D' (Affectify, "31.54 in a Single Sub", 2026-10-04) prices it as A*M*C + B*phi and also
# prices unstarted games, which Franzen's band lets jump the queue; read at call time by the harness.
if PRIORITY_FORM:
    os.environ["ARC3_PRIORITY_FORM"] = PRIORITY_FORM
    os.environ["ARC3_PRIORITY_TAIL_FADE_FRACTION"] = "0.4"
print("twin_tools wired from", _twin_apply.parent, "| ARC3_TWIN_TOOLS =", TWIN_TOOLS,
      "| first-request grace 7200 s, 30 failures, keep-alive every 300 s up to 24",
      "| slot priority", PRIORITY_FORM or "Franzen's")
