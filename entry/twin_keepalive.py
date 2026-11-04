# twin_keepalive: keep a competition scorecard alive while the model server is still loading.
#
# The ARC-AGI-3 gateway closes a scorecard that sees no action for 15 minutes. The harness sends one
# warmup RESET when the first game starts (~12 min into a run) so the run's first action lands in
# time, but on 2026-10-22/23 our model took 35-40 min to load, so after that RESET the scorecard sat
# idle and the run ended at ~27 min with nothing played (two submissions at 0.00). This repeats the
# warmup RESET every `period` seconds on the warmup game until the server's health endpoint answers,
# the game acts on its own, or the game stops; a RESET at a level's start costs one action on that
# game's level 1, the lowest weight.
#
# Clean-room code for an ARC Prize 2026 entry. Licence: MIT-0 (see LICENSE).

import os
import threading
import time


def _health_url():
    base = os.environ.get("LOCAL_ANALYZER_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8001/v1"
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base + "/health"


def _endpoint_up(url, timeout=5.0):
    try:
        import requests
        return requests.get(url, timeout=timeout).status_code == 200
    except Exception:
        return False


def start_keepalive(game, log=None, period=None, max_resets=None, endpoint_up=None, sleep=time.sleep):
    """Start the keep-alive thread for one warmup game; returns the thread, or None when disabled.

    game needs `_execute_auto_reset()`, `should_stop()` and `action_count`. period and max_resets
    default to ARC3_KEEPALIVE_SECONDS (0 = off) and ARC3_KEEPALIVE_MAX (12).
    """
    if period is None:
        period = float(os.environ.get("ARC3_KEEPALIVE_SECONDS", "0") or 0)
    if max_resets is None:
        max_resets = int(os.environ.get("ARC3_KEEPALIVE_MAX", "12") or 12)
    if period <= 0 or max_resets <= 0:
        return None
    url = _health_url()
    up = endpoint_up or (lambda: _endpoint_up(url))
    game_id = getattr(getattr(getattr(game, "game", None), "game_run", None), "game_id", "?")

    def say(msg):
        if log is not None:
            log.warning("%s: %s", game_id, msg)

    def loop():
        expected = getattr(game, "action_count", None)
        done = 0
        while done < max_resets:
            sleep(period)
            if up():
                say("keep-alive stops: the analyzer endpoint answers (%d keep-alive RESETs)" % done)
                return
            try:
                if game.should_stop():
                    return
            except Exception:
                return
            now = getattr(game, "action_count", None)
            if expected is not None and now is not None and now != expected:
                say("keep-alive stops: the game has acted")
                return
            try:
                game._execute_auto_reset()
            except Exception as exc:  # never take the run down for a keep-alive
                say("keep-alive RESET failed (%s); stopping" % exc)
                return
            done += 1
            expected = getattr(game, "action_count", None)
            say("keep-alive RESET %d/%d while the model server is still loading" % (done, max_resets))
        say("keep-alive stops after %d RESETs" % done)

    thread = threading.Thread(target=loop, name="arc3-keepalive", daemon=True)
    thread.start()
    return thread
