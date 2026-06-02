"""
Shared backtest harness for app.py.

Why this file exists
--------------------
app.py is a Streamlit script: importing it normally would fire `st.title()`,
`st.selectbox(...)`, etc. at import time. It also uses module-level date globals
(`today`, `today_str`, `season_start`, `season_end`, `recent_start`,
`recent_end`) that the prediction functions read directly — so to run those
functions for a historical date we need to override those globals per-iteration.

This module does three things:
  1. Installs a fake `streamlit` module so app.py imports cleanly with no UI.
  2. Imports app.py and exposes it as `app`.
  3. Provides `as_of(date)` — a context manager that swaps the date globals
     (and disables future-looking caches) for the duration of one prediction.

It also exposes a couple of small MLB Stats API helpers used by both
backtests (final scores, probable pitchers).

Honest caveats — please read before quoting numbers
---------------------------------------------------
Some of the data sources app.py uses are season-cumulative snapshots fetched
at run time (FanGraphs pitching/batting tables, MLB Stats API team stats,
Baseball-Reference schedules). They don't expose an "as-of-date" filter, so
when you backtest a game from April using today's API, those calls return
end-of-season-to-date numbers — i.e. mild look-ahead. The components that DO
respect as-of-date (Statcast pulls, recent-form windows, weather, schedule
matchup lookups) are the dominant signal, but the FanGraphs/team-stats
leakage is real and should be disclosed alongside any reported accuracy.
"""

from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Iterator


# ---------------------------------------------------------------------------
# Step 1: install a fake streamlit before app.py is imported
# ---------------------------------------------------------------------------


def _install_fake_streamlit() -> None:
    if "streamlit" in sys.modules and not getattr(sys.modules["streamlit"], "_is_backtest_fake", False):
        return

    fake = types.ModuleType("streamlit")
    fake._is_backtest_fake = True  # type: ignore[attr-defined]

    def _noop(*_args, **_kwargs):
        return None

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __call__(self, *_a, **_kw):
            return self

    def _ctx(*_a, **_kw):
        return _Ctx()

    def _cache_data(*args, **kwargs):
        # Support both `@st.cache_data` and `@st.cache_data(ttl=...)`.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            fn = args[0]

            @wraps(fn)
            def passthrough(*a, **kw):
                return fn(*a, **kw)

            return passthrough

        def deco(fn):
            @wraps(fn)
            def passthrough(*a, **kw):
                return fn(*a, **kw)

            return passthrough

        return deco

    fake.cache_data = _cache_data       # type: ignore[attr-defined]
    fake.cache_resource = _cache_data   # type: ignore[attr-defined]
    fake.spinner = _ctx                 # type: ignore[attr-defined]
    fake.expander = _ctx                # type: ignore[attr-defined]
    fake.container = _ctx               # type: ignore[attr-defined]
    fake.columns = lambda n=1, *a, **kw: tuple(_Ctx() for _ in range(n if isinstance(n, int) else len(n)))  # type: ignore[attr-defined]
    fake.tabs = lambda labels, *a, **kw: tuple(_Ctx() for _ in labels)  # type: ignore[attr-defined]
    fake.progress = lambda *a, **kw: types.SimpleNamespace(progress=_noop, empty=_noop)  # type: ignore[attr-defined]
    fake.session_state = {}             # type: ignore[attr-defined]
    fake.button = lambda *a, **kw: False  # type: ignore[attr-defined]

    # Anything else (title, write, warning, info, dataframe, ...) is a no-op.
    def __getattr__(name: str):
        return _noop

    fake.__getattr__ = __getattr__       # type: ignore[attr-defined]

    sys.modules["streamlit"] = fake


_install_fake_streamlit()

# Make app.py importable regardless of the caller's CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import app  # noqa: E402  — imports must follow the fake streamlit install


# ---------------------------------------------------------------------------
# Step 2: date-global patching context manager
# ---------------------------------------------------------------------------


@contextmanager
def as_of(date_str: str) -> Iterator[None]:
    """
    Temporarily make app.py believe "today" is `date_str` (YYYY-MM-DD).

    Sets:
      today        = date_str
      today_str    = date_str
      season_start = {year}-03-27
      season_end   = (date - 1 day)   <- avoids look-ahead on the prediction day itself
      recent_start = (date - 30 days)
      recent_end   = (date - 1 day)
    """
    d = datetime.strptime(date_str, "%Y-%m-%d")
    prev_day = (d - timedelta(days=1)).strftime("%Y-%m-%d")
    rec_start = (d - timedelta(days=30)).strftime("%Y-%m-%d")

    saved = {
        "today":        app.today,
        "today_str":    app.today_str,
        "season_start": app.season_start,
        "season_end":   app.season_end,
        "recent_start": app.recent_start,
        "recent_end":   app.recent_end,
    }

    app.today        = d
    app.today_str    = date_str
    app.season_start = f"{d.year}-03-27"
    app.season_end   = prev_day
    app.recent_start = rec_start
    app.recent_end   = prev_day

    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(app, k, v)


# ---------------------------------------------------------------------------
# Step 3: small MLB Stats API helpers shared by both backtests
# ---------------------------------------------------------------------------

import requests  # noqa: E402


def fetch_schedule(date_str: str) -> list[dict]:
    """Return the list of games on `date_str` with probable pitchers + final scores."""
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": date_str,
        "hydrate": "probablePitcher,team,linescore,officials",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    out = []
    for block in data.get("dates", []):
        out.extend(block.get("games", []))
    return out


def game_is_final(game: dict) -> bool:
    state = game.get("status", {}).get("abstractGameState", "")
    return state == "Final"


def winner_abbrev(game: dict) -> str | None:
    """Return the abbrev of the winning team, or None if not final / tie / unknown."""
    if not game_is_final(game):
        return None
    home = game["teams"]["home"]
    away = game["teams"]["away"]
    hs = home.get("score")
    as_ = away.get("score")
    if hs is None or as_ is None or hs == as_:
        return None
    home_abbrev = app.ID_TEAM_MAP.get(home["team"]["id"])
    away_abbrev = app.ID_TEAM_MAP.get(away["team"]["id"])
    return home_abbrev if hs > as_ else away_abbrev


def daterange(start: str, end: str) -> list[str]:
    """Inclusive YYYY-MM-DD list from start to end."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    out = []
    cur = s
    while cur <= e:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def default_date_ranges() -> list[tuple[str, str]]:
    """
    Default ranges: full 2025 season + 2026 season-to-date (relative to real today).
    Caller can override via CLI flags.
    """
    real_today = datetime.today()
    ranges = [("2025-03-27", "2025-09-28")]
    if real_today.year >= 2026:
        end = (real_today - timedelta(days=1)).strftime("%Y-%m-%d")
        ranges.append(("2026-03-27", end))
    return ranges
