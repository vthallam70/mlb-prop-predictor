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

    # Real in-memory cache. Key includes current app.py date globals so results
    # invalidate correctly when `as_of(date)` shifts the date window. Functions
    # that don't use date globals (e.g. get_team_schedule) still benefit because
    # the date-state tuple is identical across calls with the same as_of.
    _cache_store: dict = {}
    fake._cache_store = _cache_store  # type: ignore[attr-defined]

    def _date_state():
        app_mod = sys.modules.get("app")
        if app_mod is None:
            return None
        return (
            getattr(app_mod, "season_start", None),
            getattr(app_mod, "season_end", None),
            getattr(app_mod, "recent_start", None),
            getattr(app_mod, "recent_end", None),
        )

    # Skip in-memory caching for functions that return giant DataFrames.
    # pybaseball already disk-caches their raw output, and re-caching them
    # in our dict would explode RAM (statcast for a full season ≈ 500MB).
    _BLOCKED = {"get_statcast_data"}

    def _make_cached(fn):
        if fn.__name__ in _BLOCKED:
            return fn

        @wraps(fn)
        def wrapper(*a, **kw):
            try:
                key = (id(fn), a, tuple(sorted(kw.items())), _date_state())
                hash(key)  # raise TypeError if any arg is unhashable
            except TypeError:
                return fn(*a, **kw)
            if key in _cache_store:
                return _cache_store[key]
            result = fn(*a, **kw)
            _cache_store[key] = result
            return result
        return wrapper

    def _cache_data(*args, **kwargs):
        # Support both `@st.cache_data` and `@st.cache_data(ttl=...)`.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _make_cached(args[0])
        return _make_cached

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
# Step 2b: preload statcast for the whole season, then slice in-memory.
#
# app.py uses a sliding 30-day window relative to `today`, so each backtest
# day has a different date range — pybaseball's disk cache never hits and we'd
# re-download the full season per game. Instead: pull statcast ONCE for the
# whole year, then replace `app.get_statcast_data` with an in-memory slice.
# ---------------------------------------------------------------------------

_PRELOADED_STATCAST = {}  # year -> DataFrame
_FANGRAPHS_CACHE: dict = {}  # (fn_name, args, kwargs) -> DataFrame
_SCHEDULE_CACHE: dict = {}  # (year, team) -> DataFrame
_PITCHER_STATCAST_CACHE: dict = {}  # (player_id, year) -> DataFrame
_ALL_TEAMS_HEAT_CACHE: dict = {}  # date_state -> {team: heat_result}
_ALL_TEAMS_OFFENSE_CACHE: dict = {}  # date_state -> {team: float}


def _wrap_fangraphs_calls() -> None:
    """
    Memoize FanGraphs pulls (batting_stats, pitching_stats) in-memory.
    FanGraphs frequently returns 403 to scrapers; we cache failures too so we
    don't keep retrying the same blocked endpoint hundreds of times per run.
    """
    import pybaseball

    for name in ("batting_stats", "pitching_stats"):
        orig = getattr(pybaseball, name)

        def make_cached(fn, fn_name=name):
            def cached(*args, **kwargs):
                key = (fn_name, args, tuple(sorted(kwargs.items())))
                if key not in _FANGRAPHS_CACHE:
                    try:
                        _FANGRAPHS_CACHE[key] = ("ok", fn(*args, **kwargs))
                    except Exception as e:
                        _FANGRAPHS_CACHE[key] = ("err", e)
                kind, val = _FANGRAPHS_CACHE[key]
                if kind == "err":
                    raise val
                return val
            return cached

        wrapped = make_cached(orig)
        if hasattr(app, name):
            setattr(app, name, wrapped)


_wrap_fangraphs_calls()


def _wrap_team_schedule() -> None:
    """
    Memoize get_team_schedule (Baseball-Reference pd.read_html scrape) by
    (year, team). It's called by many features per game; a real cache turns
    ~5s per call into instant.
    """
    orig = app.get_team_schedule

    def cached(year, team_abbrev):
        key = (year, team_abbrev.upper())
        if key not in _SCHEDULE_CACHE:
            try:
                _SCHEDULE_CACHE[key] = ("ok", orig(year, team_abbrev))
            except Exception as e:
                _SCHEDULE_CACHE[key] = ("err", e)
        kind, val = _SCHEDULE_CACHE[key]
        if kind == "err":
            raise val
        return val

    app.get_team_schedule = cached


_wrap_team_schedule()


def _wrap_statcast_pitcher() -> None:
    """
    Replace pybaseball.statcast_pitcher with an in-memory slice from the
    preloaded full-season DataFrame, keyed by player_id. Eliminates per-pitcher
    network pulls (the dominant bottleneck after FanGraphs and B-Ref).
    """
    import pybaseball
    orig = pybaseball.statcast_pitcher

    def sliced(start_dt, end_dt, player_id):
        # If we have preloaded data for the year, slice from it.
        year = int(start_dt[:4]) if isinstance(start_dt, str) else start_dt.year
        full = _PRELOADED_STATCAST.get(year)
        if full is None:
            return orig(start_dt, end_dt, player_id)
        sd = start_dt if isinstance(start_dt, str) else start_dt.strftime("%Y-%m-%d")
        ed = end_dt if isinstance(end_dt, str) else end_dt.strftime("%Y-%m-%d")
        return full[
            (full["pitcher"] == player_id)
            & (full["game_date"] >= sd)
            & (full["game_date"] <= ed)
        ]

    # app.py imports statcast_pitcher by name, so patch its module namespace.
    if hasattr(app, "statcast_pitcher"):
        app.statcast_pitcher = sliced
    # Also patch the same name in pybaseball, defensively.
    pybaseball.statcast_pitcher = sliced


def _wrap_statcast_batter() -> None:
    """Same trick for statcast_batter (used by player-prop modes)."""
    import pybaseball
    orig = pybaseball.statcast_batter

    def sliced(start_dt, end_dt, player_id):
        year = int(start_dt[:4]) if isinstance(start_dt, str) else start_dt.year
        full = _PRELOADED_STATCAST.get(year)
        if full is None:
            return orig(start_dt, end_dt, player_id)
        sd = start_dt if isinstance(start_dt, str) else start_dt.strftime("%Y-%m-%d")
        ed = end_dt if isinstance(end_dt, str) else end_dt.strftime("%Y-%m-%d")
        return full[
            (full["batter"] == player_id)
            & (full["game_date"] >= sd)
            & (full["game_date"] <= ed)
        ]

    if hasattr(app, "statcast_batter"):
        app.statcast_batter = sliced
    pybaseball.statcast_batter = sliced


_wrap_statcast_pitcher()
_wrap_statcast_batter()


# ---------------------------------------------------------------------------
# All-teams heat precompute. Replaces per-team team_player_heat / team_offense_heat
# with one big groupby that produces results for every team in a single pass.
# Within a day's slate (same date_state) all 30 teams share the same precompute.
# ---------------------------------------------------------------------------


_WOBA_WEIGHTS = {"walk": 0.69, "hit_by_pitch": 0.72, "single": 0.89,
                 "double": 1.27, "triple": 1.62, "home_run": 2.10}


def _compute_all_teams_offense_heat():
    """Return {team_abbrev: float in [-1.5, 1.5]} for the current date_state."""
    import pandas as pd
    key = (app.season_start, app.season_end, app.recent_start, app.recent_end)
    if key in _ALL_TEAMS_OFFENSE_CACHE:
        return _ALL_TEAMS_OFFENSE_CACHE[key]

    season = app.get_statcast_data(app.season_start, app.season_end)
    recent = app.get_statcast_data(app.recent_start, app.recent_end)

    def team_woba(df):
        if df.empty:
            return {}
        ev = df["events"]
        mask = ev.notna()
        sub = df[mask]
        # Per-team event sums
        bb = sub["events"].eq("walk")
        hbp = sub["events"].eq("hit_by_pitch")
        s1 = sub["events"].eq("single")
        s2 = sub["events"].eq("double")
        s3 = sub["events"].eq("triple")
        hr = sub["events"].eq("home_run")
        sf = sub["events"].eq("sac_fly")
        not_ab = sub["events"].isin(
            ["walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"]
        )
        ab = ~not_ab
        df2 = pd.DataFrame({
            "batting_team": sub["batting_team"].values,
            "bb": bb.values, "hbp": hbp.values, "s1": s1.values,
            "s2": s2.values, "s3": s3.values, "hr": hr.values,
            "sf": sf.values, "ab": ab.values,
        })
        agg = df2.groupby("batting_team").sum()
        num = (_WOBA_WEIGHTS["walk"] * agg["bb"]
               + _WOBA_WEIGHTS["hit_by_pitch"] * agg["hbp"]
               + _WOBA_WEIGHTS["single"] * agg["s1"]
               + _WOBA_WEIGHTS["double"] * agg["s2"]
               + _WOBA_WEIGHTS["triple"] * agg["s3"]
               + _WOBA_WEIGHTS["home_run"] * agg["hr"])
        denom = agg["ab"] + agg["bb"] + agg["hbp"] + agg["sf"]
        woba = num / denom.replace(0, 1)
        return woba.to_dict()

    s_w = team_woba(season)
    r_w = team_woba(recent)
    out = {}
    for team in set(s_w) | set(r_w):
        s = s_w.get(team)
        r = r_w.get(team)
        if s is None or r is None or s <= 0:
            out[team] = 0.0
            continue
        delta = (r - s) / 0.050
        out[team] = float(max(-1.5, min(1.5, delta)))
    _ALL_TEAMS_OFFENSE_CACHE[key] = out
    return out


def _compute_all_teams_player_heat():
    """Return {team_abbrev: {score, hot, cold}} for the current date_state."""
    import pandas as pd
    key = (app.season_start, app.season_end, app.recent_start, app.recent_end)
    if key in _ALL_TEAMS_HEAT_CACHE:
        return _ALL_TEAMS_HEAT_CACHE[key]

    season = app.get_statcast_data(app.season_start, app.season_end)
    recent = app.get_statcast_data(app.recent_start, app.recent_end)

    def player_woba_table(df):
        if df.empty:
            return pd.DataFrame()
        mask = df["events"].notna()
        sub = df[mask]
        ev = sub["events"]
        # Vectorized event indicator columns
        cols = {
            "bb": ev.eq("walk").astype(int),
            "hbp": ev.eq("hit_by_pitch").astype(int),
            "s1": ev.eq("single").astype(int),
            "s2": ev.eq("double").astype(int),
            "s3": ev.eq("triple").astype(int),
            "hr": ev.eq("home_run").astype(int),
            "sf": ev.eq("sac_fly").astype(int),
            "ab": (~ev.isin(
                ["walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"]
            )).astype(int),
        }
        name_series = sub["player_name"] if "player_name" in sub.columns else sub["batter"].astype(str)
        df2 = pd.DataFrame({
            "batting_team": sub["batting_team"].values,
            "batter": sub["batter"].values,
            "name": name_series.values,
            **{k: v.values for k, v in cols.items()},
        })
        df2["pa"] = 1
        agg = df2.groupby(["batting_team", "batter"]).agg(
            bb=("bb", "sum"), hbp=("hbp", "sum"), s1=("s1", "sum"),
            s2=("s2", "sum"), s3=("s3", "sum"), hr=("hr", "sum"),
            sf=("sf", "sum"), ab=("ab", "sum"), pa=("pa", "sum"),
            name=("name", "first"),
        )
        num = (_WOBA_WEIGHTS["walk"] * agg["bb"]
               + _WOBA_WEIGHTS["hit_by_pitch"] * agg["hbp"]
               + _WOBA_WEIGHTS["single"] * agg["s1"]
               + _WOBA_WEIGHTS["double"] * agg["s2"]
               + _WOBA_WEIGHTS["triple"] * agg["s3"]
               + _WOBA_WEIGHTS["home_run"] * agg["hr"])
        denom = (agg["ab"] + agg["bb"] + agg["hbp"] + agg["sf"]).replace(0, 1)
        agg["woba"] = num / denom
        return agg.reset_index()

    season_t = player_woba_table(season)
    recent_t = player_woba_table(recent)
    if season_t.empty or recent_t.empty:
        _ALL_TEAMS_HEAT_CACHE[key] = {}
        return {}

    season_t = season_t[season_t["pa"] >= 30]
    recent_t = recent_t[recent_t["pa"] >= 15]
    joined = recent_t.merge(season_t, on="batter", suffixes=("_r", "_s"))
    if joined.empty:
        _ALL_TEAMS_HEAT_CACHE[key] = {}
        return {}
    joined["delta"] = joined["woba_r"] - joined["woba_s"]

    out = {}
    for team, grp in joined.groupby("batting_team_r"):
        total_pa = grp["pa_r"].sum()
        if total_pa <= 0:
            out[team] = {"score": 0.0, "hot": [], "cold": []}
            continue
        weighted_delta = (grp["delta"] * grp["pa_r"]).sum() / total_pa
        score = float(max(-1.5, min(1.5, weighted_delta / 0.050)))

        top = grp.sort_values("delta", ascending=False)
        bot = grp.sort_values("delta", ascending=True)
        hot = []
        cold = []
        for _, row in top.head(3).iterrows():
            if row["delta"] <= 0:
                break
            nm = row.get("name_r") or str(row["batter"])
            hot.append((nm, float(row["delta"])))
        for _, row in bot.head(2).iterrows():
            if row["delta"] >= 0:
                break
            nm = row.get("name_r") or str(row["batter"])
            cold.append((nm, float(row["delta"])))
        out[team] = {"score": score, "hot": hot, "cold": cold}

    _ALL_TEAMS_HEAT_CACHE[key] = out
    return out


def _patch_heat_functions():
    """Replace app.team_offense_heat and app.team_player_heat with batch-precompute versions."""
    def patched_offense_heat(team_abbrev):
        return _compute_all_teams_offense_heat().get(team_abbrev.upper(), 0.0)

    def patched_player_heat(team_abbrev):
        d = _compute_all_teams_player_heat()
        return d.get(team_abbrev.upper(), {"score": 0.0, "hot": [], "cold": []})

    app.team_offense_heat = patched_offense_heat
    app.team_player_heat = patched_player_heat


_patch_heat_functions()


def preload_statcast(year: int, end_date: str | None = None) -> None:
    """
    Pull every statcast pitch for `year` (from 03-27 to end_date or 11-01)
    once and stash it. Subsequent `app.get_statcast_data(start, end)` calls
    return a slice from the stash instead of hitting pybaseball.
    """
    import pandas as pd
    from pybaseball import statcast

    if year in _PRELOADED_STATCAST:
        return

    start = f"{year}-03-27"
    end = end_date or f"{year}-11-01"
    print(f"  [harness] Preloading statcast {start} → {end} (one-time pull)…")
    df = statcast(start, end)
    df = df.copy()
    # Normalize game_date to string for cheap comparison against the YYYY-MM-DD
    # strings that app.py passes around.
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    # Precompute batting_team / fielding_team using vectorized ops — app.py
    # functions otherwise do a per-row .apply() on 218k rows each call, which
    # was the dominant CPU cost.
    is_top = df["inning_topbot"] == "Top"
    df["batting_team"] = df["away_team"].where(is_top, df["home_team"])
    df["fielding_team"] = df["home_team"].where(is_top, df["away_team"])
    _PRELOADED_STATCAST[year] = df
    print(f"  [harness] Statcast preload done: {len(df):,} pitches cached")

    # Install the in-memory slicer in place of app.get_statcast_data.
    def sliced(start_date: str, end_date: str):
        ydf = _PRELOADED_STATCAST.get(year)
        if ydf is None:
            return df.iloc[0:0]
        return ydf[(ydf["game_date"] >= start_date) & (ydf["game_date"] <= end_date)]

    app.get_statcast_data = sliced


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
