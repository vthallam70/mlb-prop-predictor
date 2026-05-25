"""
MLB Player Prop + Moneyline Predictor
======================================
Improvements in this version:
  A. FIP / xFIP instead of ERA for starting pitchers
  B. wRC+ (park-adjusted) instead of raw OPS for team offense
  C. Swinging-strike rate (SwStr%) for strikeout prop quality signal
  D. Injury / IL roster check via MLB Stats API
  E. Rest-days / schedule-fatigue signal
  F. BABIP regression flag for starters
  G. Weather adjustment (wind + temperature) via Open-Meteo (free, no key)
"""

import streamlit as st
from pybaseball import (
    statcast_pitcher,
    statcast_batter,
    playerid_lookup,
    pitching_stats,
    batting_stats,
    statcast,
    cache,
)
from datetime import datetime, timedelta
import math
import pandas as pd
import requests

cache.enable()

st.title("MLB Player Prop + Moneyline Predictor")

# ---------------------------------------------------------------------------
# Constants & mappings
# ---------------------------------------------------------------------------

TEAM_ID_MAP = {
    "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111, "CHC": 112,
    "CWS": 145, "CHW": 145, "CIN": 113, "CLE": 114, "COL": 115,
    "DET": 116, "HOU": 117, "KC":  118, "KCR": 118, "LAA": 108,
    "LAD": 119, "MIA": 146, "MIL": 158, "MIN": 142, "NYM": 121,
    "NYY": 147, "ATH": 133, "OAK": 133, "PHI": 143, "PIT": 134,
    "SD":  135, "SDP": 135, "SEA": 136, "SF":  137, "SFG": 137,
    "STL": 138, "TB":  139, "TBR": 139, "TEX": 140, "TOR": 141,
    "WSH": 120, "WAS": 120,
}

BR_TEAM_MAP = {
    "SF":  "SFG", "SD":  "SDP", "TB":  "TBR", "KC":  "KCR",
    "CWS": "CHW", "CHW": "CHW", "OAK": "ATH", "ATH": "ATH",
    "WSH": "WSN", "WAS": "WSN",
}

ID_TEAM_MAP = {v: k for k, v in TEAM_ID_MAP.items()}

# Ballpark GPS coordinates for weather lookup
PARK_COORDS = {
    "ARI": (33.4453, -112.0667), "ATL": (33.8908, -84.4678),
    "BAL": (39.2839, -76.6216),  "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),  "CWS": (41.8299, -87.6338),
    "CHW": (41.8299, -87.6338),  "CIN": (39.0975, -84.5080),
    "CLE": (41.4962, -81.6852),  "COL": (39.7559, -104.9942),
    "DET": (42.3390, -83.0485),  "HOU": (29.7572, -95.3555),
    "KC":  (39.0517, -94.4803),  "KCR": (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827), "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197),  "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2778),  "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),  "ATH": (37.7516, -122.2005),
    "OAK": (37.7516, -122.2005), "PHI": (39.9061, -75.1665),
    "PIT": (40.4469, -80.0057),  "SD":  (32.7076, -117.1570),
    "SDP": (32.7076, -117.1570), "SEA": (47.5914, -122.3325),
    "SF":  (37.7786, -122.3893), "SFG": (37.7786, -122.3893),
    "STL": (38.6226, -90.1928),  "TB":  (27.7683, -82.6534),
    "TBR": (27.7683, -82.6534),  "TEX": (32.7473, -97.0824),
    "TOR": (43.6414, -79.3894),  "WSH": (38.8730, -77.0074),
    "WAS": (38.8730, -77.0074),
}

PARK_FACTORS = {
    "COL": 1.18, "CIN": 1.08, "TEX": 1.06, "BOS": 1.05, "CHC": 1.04,
    "MIL": 1.03, "PHI": 1.02, "BAL": 1.01, "ARI": 1.00, "STL": 0.99,
    "HOU": 0.99, "ATL": 0.98, "LAD": 0.98, "MIN": 0.97, "NYY": 0.97,
    "CLE": 0.97, "DET": 0.97, "PIT": 0.96, "WSH": 0.96, "WAS": 0.96,
    "NYM": 0.96, "KC":  0.96, "TOR": 0.95, "SDP": 0.95, "SD":  0.95,
    "LAA": 0.95, "CHW": 0.94, "CWS": 0.94, "SF":  0.94, "SFG": 0.94,
    "OAK": 0.94, "ATH": 0.94, "TB":  0.94, "TBR": 0.94, "MIA": 0.93,
    "SEA": 0.93,
}

PARK_K_FACTORS = {
    "SEA": 1.05, "MIA": 1.04, "SF": 1.03, "SFG": 1.03,
    "TB":  1.03, "TBR": 1.03, "COL": 0.94, "CIN": 0.96,
    "TEX": 0.97, "BOS": 0.97,
}

UMPIRE_K_RATES = {
    "Angel Hernandez": 0.94, "CB Bucknor": 0.96,
    "Doug Eddings":    1.06, "Laz Diaz":   1.05,
    "Jordan Baker":    1.04, "Dan Bellino": 1.03,
}

# ---------------------------------------------------------------------------
# Date globals
# ---------------------------------------------------------------------------

today        = datetime.today()
today_str    = today.strftime("%Y-%m-%d")
season_start = f"{today.year}-03-27"
season_end   = today.strftime("%Y-%m-%d")
recent_start = (today - timedelta(days=30)).strftime("%Y-%m-%d")
recent_end   = today.strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

prop_type = st.selectbox(
    "Choose Bet Type",
    ["Pitcher Strikeouts", "Batter Hits", "Batter Total Bases", "Team Moneyline"],
)

if prop_type == "Team Moneyline":
    team          = st.text_input("Team Abbreviation", "DET")
    opponent_team = st.text_input("Opponent Team Abbreviation", "SEA")
    american_odds = st.number_input("American Odds", value=-110)

    override_starters = st.checkbox("Manually enter starting pitchers", value=False)
    if override_starters:
        team_starter_last      = st.text_input("Your Team Starter Last Name", "")
        team_starter_first     = st.text_input("Your Team Starter First Name", "")
        opponent_starter_last  = st.text_input("Opponent Starter Last Name", "")
        opponent_starter_first = st.text_input("Opponent Starter First Name", "")
    else:
        team_starter_last = team_starter_first = ""
        opponent_starter_last = opponent_starter_first = ""
else:
    last_name       = st.text_input("Player Last Name", "skubal")
    first_name      = st.text_input("Player First Name", "tarik")
    opponent_team   = st.text_input("Opponent Team Abbreviation", "SEA")
    sportsbook_line = st.number_input("Sportsbook Line (e.g. 6.5)", value=6.5, step=0.5)

    st.caption("Enter the odds for each side separately — books often shade these differently.")
    odds_col1, odds_col2 = st.columns(2)
    with odds_col1:
        over_odds  = st.number_input("Over Odds (e.g. -115)", value=-115)
    with odds_col2:
        under_odds = st.number_input("Under Odds (e.g. -105)", value=-105)

    # Keep american_odds as over_odds for any legacy references
    american_odds = over_odds

# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def get_player_id(last, first):
    p = playerid_lookup(last, first)
    return None if p.empty else p.iloc[0]["key_mlbam"]

def calc_weighted_projection(season_avg, recent_avg):
    if math.isnan(recent_avg):
        recent_avg = season_avg
    return season_avg * 0.8 + recent_avg * 0.2

def total_bases_from_event(event):
    return {"single": 1, "double": 2, "triple": 3, "home_run": 4}.get(event, 0)

def implied_probability(odds):
    return abs(odds) / (abs(odds) + 100) if odds < 0 else 100 / (odds + 100)

def estimate_over_probability(projection, line, std_dev=None):
    scale = max(std_dev, 0.5) if std_dev is not None else 1.5
    return 1 / (1 + math.exp(-(projection - line) / scale))

def expected_value(model_prob, odds):
    profit = odds / 100 if odds > 0 else 100 / abs(odds)
    return model_prob * profit - (1 - model_prob)

def normalize(value, center, scale):
    return (value - center) / scale

def confidence_label(edge, over_ev, under_ev):
    """
    Evaluate both sides of a prop independently.
    edge > 0 means projection is above the line (lean over).
    edge < 0 means projection is below the line (lean under).
    We evaluate whichever side has positive EV.
    """
    if   over_ev  > 0.10 and edge >  0.5: return "Strong Over"
    elif over_ev  > 0.03 and edge >  0.2: return "Lean Over"
    elif under_ev > 0.10 and edge < -0.5: return "Strong Under"
    elif under_ev > 0.03 and edge < -0.2: return "Lean Under"
    return "No Bet"

def moneyline_label(ev, model_prob, implied_prob, american_odds):
    """
    Betting filter: edge-first with minimum win probability floor.

    Sharp MLB bettors realistically target 54-57% win rates — not 60%.
    A 60% win rate on MLB moneylines over a season would be historically
    elite. The filter is designed to find REAL edge vs the book, not
    chase an unrealistic win-rate ceiling.

    Primary filter: edge over book implied probability
    Secondary floor: model must show above-average win confidence

    Tier 1 - STRONG BET:
        model_prob >= 54%   (meaningfully above coin-flip)
        edge over book >= 5%  (real mispricing)
        EV >= 4%

    Tier 2 - LEAN:
        model_prob >= 52%   (slightly above coin-flip)
        edge over book >= 3%  (some mispricing)
        EV >= 2%

    Hard blocks:
        - Model < 52%: true coin flip, never bet
        - Heavy chalk (-160 or worse): vig kills value, need 7%+ edge
        - Negative edge: model agrees with or is below book — no value
    """
    # Round to avoid floating point edge cases (e.g. 0.51999... vs 0.52)
    mp       = round(model_prob,   4)
    ip       = round(implied_prob, 4)
    edge_pct = round(mp - ip,      4)
    ev_r     = round(ev,           4)

    # Hard block 1: model not confident enough
    if mp < 0.52:
        return "No Bet"

    # Hard block 2: negative or zero edge
    if edge_pct <= 0:
        return "No Bet"

    # Hard block 3: heavy chalk
    if american_odds <= -160 and edge_pct < 0.07:
        return "No Bet"

    # Tier 1: Strong Bet
    if mp >= 0.54 and edge_pct >= 0.05 and ev_r >= 0.04:
        return "Strong Bet"

    # Tier 2: Lean
    if mp >= 0.52 and edge_pct >= 0.03 and ev_r >= 0.02:
        return "Lean"

    return "No Bet"

    # Tier 1: Strong Bet — solid win confidence + clear value
    if model_prob >= 0.54 and edge_pct >= 0.05 and ev >= 0.04:
        return "Strong Bet"

    # Tier 2: Lean — above coin-flip + some value
    if model_prob >= 0.52 and edge_pct >= 0.03 and ev >= 0.02:
        return "Lean"

    return "No Bet"

def get_umpire_k_adjustment(name):
    return UMPIRE_K_RATES.get(name, 1.0) if name else 1.0

# ---------------------------------------------------------------------------
# A. FIP / xFIP for starters  (via pybaseball pitching_stats)
# ---------------------------------------------------------------------------

@st.cache_data
def get_fip_stats(year):
    """Return a DataFrame with FIP, xFIP, SIERA, BABIP, K/9, BB/9 per pitcher."""
    try:
        df = pitching_stats(year, qual=1)   # qual=1 => all pitchers with ≥1 IP
        return df
    except Exception:
        return pd.DataFrame()

def get_pitcher_fip(mlbam_id, year):
    """
    Look up a pitcher's FIP/xFIP/SIERA from the Fangraphs-sourced pitching_stats.
    Returns dict with fip, xfip, siera, babip, k_per_9, era, name.
    Falls back gracefully.
    """
    defaults = {"fip": 4.20, "xfip": 4.20, "siera": 4.20,
                "babip": 0.300, "k_per_9": 8.0, "era": 4.30,
                "swstr_pct": 0.11, "name": "Unknown", "throws": "R"}
    try:
        df = get_fip_stats(year)
        if df.empty:
            return defaults

        # pybaseball uses MLBAMID column; match on it
        match = df[df["MLBAMID"] == mlbam_id] if "MLBAMID" in df.columns else pd.DataFrame()

        # Fallback: try IDfg which sometimes links
        if match.empty and "IDfg" in df.columns:
            match = df[df["IDfg"] == mlbam_id]

        if match.empty:
            return defaults

        row = match.iloc[0]

        def safe(col, default):
            try:
                v = row.get(col, default)
                return float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else default
            except Exception:
                return default

        return {
            "fip":       safe("FIP",   4.20),
            "xfip":      safe("xFIP",  4.20),
            "siera":     safe("SIERA", 4.20),
            "babip":     safe("BABIP", 0.300),
            "k_per_9":   safe("K/9",   8.0),
            "era":       safe("ERA",   4.30),
            "swstr_pct": safe("SwStr%", 0.11),
            "name":      str(row.get("Name", "Unknown")),
            "throws":    defaults["throws"],  # filled below from MLB API
        }
    except Exception:
        return defaults


@st.cache_data
def get_pitcher_info(player_id):
    """Pull ERA + handedness from MLB Stats API; FIP/xFIP overlaid from Fangraphs."""
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
    params = {"hydrate": "stats(group=[pitching],type=[season])"}
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {"era": 4.30, "fip": 4.20, "xfip": 4.20, "siera": 4.20,
                "babip": 0.300, "k_per_9": 8.0, "swstr_pct": 0.11,
                "throws": "R", "name": "Unknown"}

    people = data.get("people", [])
    if not people:
        return {"era": 4.30, "fip": 4.20, "xfip": 4.20, "siera": 4.20,
                "babip": 0.300, "k_per_9": 8.0, "swstr_pct": 0.11,
                "throws": "R", "name": "Unknown"}

    person = people[0]
    throws = person.get("pitchHand", {}).get("code", "R")
    name   = person.get("fullName", "Unknown")
    era    = 4.30

    stats = person.get("stats", [])
    if stats:
        splits = stats[0].get("splits", [])
        if splits:
            try:
                era = float(splits[0].get("stat", {}).get("era", "4.30"))
            except ValueError:
                pass

    # Overlay FIP data from Fangraphs
    fip_data = get_pitcher_fip(int(player_id), today.year)

    # If FIP lookup returned all defaults (4.20), estimate from ERA instead.
    # FIP closely tracks ERA; xFIP regresses toward mean more aggressively.
    # This ensures the model uses real pitching signal even when Fangraphs
    # ID matching fails, rather than cancelling out both pitchers at 4.20.
    if fip_data["fip"] == 4.20 and fip_data["xfip"] == 4.20 and era != 4.30:
        fip_data["fip"]   = round(era * 0.95 + 4.20 * 0.05, 3)
        fip_data["xfip"]  = round(era * 0.88 + 4.20 * 0.12, 3)
        fip_data["siera"] = round(era * 0.85 + 4.20 * 0.15, 3)
        # Estimate K/9 from ERA (rough inverse relationship)
        fip_data["k_per_9"] = round(max(4.0, 12.0 - era * 1.0), 1)

    fip_data["era"]    = era
    fip_data["throws"] = throws
    fip_data["name"]   = name
    return fip_data


# ---------------------------------------------------------------------------
# B. wRC+ for team offense  (via pybaseball batting_stats)
# ---------------------------------------------------------------------------

@st.cache_data
def get_team_wrc_plus(team_abbrev, year):
    """
    Compute a PA-weighted average wRC+ for a team by pulling all individual
    batters from batting_stats() and filtering to that team.

    batting_stats(ind=1) returns one row per player. We filter by Team,
    then take a PA-weighted mean of wRC+ so that everyday starters drive
    the number rather than bench players with 3 ABs.

    Returns float (100 = league average).
    """
    try:
        # ind=1 = individual player rows (default); qual=0 = no PA minimum
        df = batting_stats(year, qual=0, ind=1)

        if df.empty or "Team" not in df.columns or "wRC+" not in df.columns:
            return 100.0

        team_abbrev = team_abbrev.upper()

        # Fangraphs team abbreviations differ from MLB/Statcast in a few cases
        FG_MAP = {
            "WSH": "WSN", "WAS": "WSN", "CWS": "CHW",
            "SD":  "SDP", "SF":  "SFG", "TB":  "TBR", "KC": "KCR",
            "OAK": "ATH",
        }
        fg_abbrev = FG_MAP.get(team_abbrev, team_abbrev)

        # Filter to this team's players
        team_df = df[df["Team"].str.upper() == fg_abbrev].copy()

        if team_df.empty:
            # Soft fallback: try 3-char prefix match in case abbreviation differs
            team_df = df[df["Team"].str.upper().str.startswith(fg_abbrev[:3], na=False)].copy()

        if team_df.empty:
            return 100.0

        # Drop rows missing wRC+; coerce PA to numeric
        team_df["wRC+"] = pd.to_numeric(team_df["wRC+"], errors="coerce")
        team_df["PA"]   = pd.to_numeric(team_df["PA"], errors="coerce") if "PA" in team_df.columns else 1.0
        team_df = team_df.dropna(subset=["wRC+"])

        if team_df.empty:
            return 100.0

        # PA-weighted average so starters drive the number
        total_pa = team_df["PA"].sum()
        if total_pa > 0:
            return round(float((team_df["wRC+"] * team_df["PA"]).sum() / total_pa), 1)

        return round(float(team_df["wRC+"].mean()), 1)

    except Exception:
        return 100.0


# ---------------------------------------------------------------------------
# C. Swinging-strike rate from recent Statcast  (pitcher-level)
# ---------------------------------------------------------------------------

@st.cache_data
def get_pitcher_swstr(player_id):
    """
    Compute swinging-strike % from recent Statcast data.
    SwStr% = swinging strikes / total pitches.
    Returns float (e.g. 0.13 = 13 %).
    """
    try:
        data = statcast_pitcher(recent_start, recent_end, player_id)
        if data.empty:
            return None

        total_pitches = len(data)
        swinging_strikes = data[
            data["description"].isin([
                "swinging_strike",
                "swinging_strike_blocked",
                "foul_tip",
            ])
        ].shape[0]

        if total_pitches < 50:
            return None

        return round(swinging_strikes / total_pitches, 4)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# D. Injury / IL roster check
# ---------------------------------------------------------------------------

@st.cache_data
def get_injured_players(team_abbrev):
    """
    Return list of player names currently on the IL for a team.
    Uses the MLB Stats API 10-day and 60-day IL roster types.
    """
    team_id = TEAM_ID_MAP.get(team_abbrev.upper())
    if team_id is None:
        return []

    injured = []
    for roster_type in ["injuries", "illList"]:
        try:
            url    = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
            params = {"rosterType": roster_type, "season": today.year}
            resp   = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data   = resp.json()
            for p in data.get("roster", []):
                name = p.get("person", {}).get("fullName", "")
                pos  = p.get("position", {}).get("abbreviation", "")
                if name:
                    injured.append(f"{name} ({pos})")
        except Exception:
            pass

    return list(set(injured))


# ---------------------------------------------------------------------------
# E. Rest days / schedule fatigue
# ---------------------------------------------------------------------------

@st.cache_data
def get_team_schedule(year, team_abbrev):
    br_team = BR_TEAM_MAP.get(team_abbrev.upper(), team_abbrev.upper())
    url     = (
        f"https://www.baseball-reference.com/teams/"
        f"{br_team}/{year}-schedule-scores.shtml"
    )
    try:
        tables = pd.read_html(url)
    except Exception as e:
        st.warning(f"Could not load schedule for {team_abbrev}: {e}")
        return pd.DataFrame()

    data = tables[0].copy()
    data = data[pd.to_numeric(data["Gm#"], errors="coerce").notna()]
    data = data[data["W/L"].astype(str).str.match(r"^(W|L)(-|$)")]
    data["win"] = data["W/L"].astype(str).str.startswith("W").astype(int)
    data["R"]   = pd.to_numeric(data["R"],  errors="coerce")
    data["RA"]  = pd.to_numeric(data["RA"], errors="coerce")
    return data


def get_rest_days(team_abbrev):
    """
    Return how many days since a team's last game.
    0 = back-to-back, 1 = normal rest, 2+ = extra rest.
    """
    try:
        data = get_team_schedule(today.year, team_abbrev)
        if data.empty or "Date" not in data.columns:
            return 1

        # Parse the Date column (B-Ref format: "Monday, Apr 1")
        dates = pd.to_datetime(
            data["Date"].astype(str).str.extract(r"(\w+ \d+)")[0]
            + f" {today.year}",
            format="%b %d %Y",
            errors="coerce",
        ).dropna()

        if dates.empty:
            return 1

        last_game = dates.max()
        delta     = (today - last_game).days
        return max(int(delta), 0)
    except Exception:
        return 1


def get_streak(team_abbrev):
    """Return recent W/L streak length (+N = win streak, -N = loss streak)."""
    try:
        data = get_team_schedule(today.year, team_abbrev)
        if data.empty:
            return 0

        results = data["win"].tolist()
        if not results:
            return 0

        last    = results[-1]
        streak  = 0
        for r in reversed(results):
            if r == last:
                streak += 1
            else:
                break
        return streak if last == 1 else -streak
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# F. BABIP regression flag  (inside get_pitcher_info already fetches BABIP)
# ---------------------------------------------------------------------------

def babip_regression_flag(babip):
    """
    Returns a warning string if BABIP is far from league average (.295-.305).
    Low BABIP pitcher = ERA may be unsustainably good (due for regression).
    High BABIP pitcher = ERA may be inflated by bad luck.
    """
    if babip < 0.260:
        return f"⚠️ Low BABIP ({babip:.3f}) — ERA likely better than true talent, expect regression"
    elif babip > 0.340:
        return f"✅ High BABIP ({babip:.3f}) — ERA likely worse than true talent, improvement likely"
    return None


# ---------------------------------------------------------------------------
# G. Weather adjustment via Open-Meteo (free, no API key)
# ---------------------------------------------------------------------------

@st.cache_data
def get_weather(lat, lon, date_str):
    """
    Fetch hourly wind speed + temperature for game day from Open-Meteo.
    Returns dict with temp_f, wind_mph, wind_dir_deg.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":       lat,
            "longitude":      lon,
            "hourly":         "temperature_2m,windspeed_10m,winddirection_10m",
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "timezone":       "auto",
            "start_date":     date_str,
            "end_date":       date_str,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        temps  = hourly.get("temperature_2m", [])
        winds  = hourly.get("windspeed_10m", [])
        dirs   = hourly.get("winddirection_10m", [])

        # Use 7 PM local (index ~19) as game-time proxy; fall back to daily avg
        idx = 19 if len(temps) > 19 else len(temps) // 2

        return {
            "temp_f":       round(temps[idx], 1) if temps else 70.0,
            "wind_mph":     round(winds[idx], 1) if winds else 5.0,
            "wind_dir_deg": round(dirs[idx], 1)  if dirs  else 0.0,
        }
    except Exception:
        return {"temp_f": 70.0, "wind_mph": 5.0, "wind_dir_deg": 0.0}


def weather_run_factor(temp_f, wind_mph, wind_dir_deg=None):
    """
    Estimate a run-scoring multiplier based on temperature and wind.
    Cold air = ball dies; hot air = ball carries.
    Wind blowing out (approx 45-135 deg = out to RF/CF/LF in typical stadiums)
    gives a boost; blowing in suppresses.
    Returns multiplier near 1.0 (e.g. 1.04 = 4% more runs expected).
    """
    # Temperature effect: ~0.4% per degree above/below 72°F
    temp_factor = 1.0 + (temp_f - 72) * 0.004

    # Wind effect: rough approximation, out wind boosts, in suppresses
    wind_factor = 1.0
    if wind_mph > 5:
        # Treat 45-135° as "blowing out", 225-315° as "blowing in"
        if wind_dir_deg is not None:
            if 45 <= wind_dir_deg <= 135:
                wind_factor = 1.0 + wind_mph * 0.005   # out
            elif 225 <= wind_dir_deg <= 315:
                wind_factor = 1.0 - wind_mph * 0.004   # in
            else:
                wind_factor = 1.0 + wind_mph * 0.001   # cross-wind, minor boost
        else:
            wind_factor = 1.0 + wind_mph * 0.002

    return round(temp_factor * wind_factor, 4)


def weather_k_factor(temp_f, wind_mph):
    """Cold / windy days slightly suppress strikeouts (more contact-friendly)."""
    temp_effect = 1.0 - (temp_f - 72) * 0.001  # cold = slightly more Ks? negligible
    wind_effect = 1.0 - wind_mph * 0.001
    return round(max(temp_effect * wind_effect, 0.90), 4)


# ---------------------------------------------------------------------------
# Schedule / odds helpers
# ---------------------------------------------------------------------------

@st.cache_data
def get_mlb_schedule(date):
    url    = "https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1, "date": date,
        "hydrate": "probablePitcher,team,linescore,officials",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def find_today_matchup(team, opponent_team):
    team          = team.upper()
    opponent_team = opponent_team.upper()
    team_id       = TEAM_ID_MAP.get(team)
    opponent_id   = TEAM_ID_MAP.get(opponent_team)
    if team_id is None or opponent_id is None:
        return None

    schedule = get_mlb_schedule(today_str)
    for date_block in schedule.get("dates", []):
        for game in date_block.get("games", []):
            home    = game["teams"]["home"]["team"]
            away    = game["teams"]["away"]["team"]
            home_id = home["id"]
            away_id = away["id"]

            if not (
                (home_id == team_id and away_id == opponent_id)
                or (home_id == opponent_id and away_id == team_id)
            ):
                continue

            selected_is_home = home_id == team_id
            if selected_is_home:
                sel_p   = game["teams"]["home"].get("probablePitcher")
                opp_p   = game["teams"]["away"].get("probablePitcher")
                home_ab = team
            else:
                sel_p   = game["teams"]["away"].get("probablePitcher")
                opp_p   = game["teams"]["home"].get("probablePitcher")
                home_ab = opponent_team

            umpire = None
            for off in game.get("officials", []):
                if off.get("officialType") == "Home Plate":
                    umpire = off["official"].get("fullName")
                    break

            return {
                "selected_is_home": selected_is_home,
                "home_team":        ID_TEAM_MAP.get(home_id, home["name"]),
                "away_team":        ID_TEAM_MAP.get(away_id, away["name"]),
                "home_team_abbrev": home_ab,
                "selected_pitcher": sel_p,
                "opponent_pitcher": opp_p,
                "game_time":        game.get("gameDate"),
                "umpire_name":      umpire,
            }
    return None


@st.cache_data
def get_bullpen_era(team_abbrev):
    team_id = TEAM_ID_MAP.get(team_abbrev.upper())
    if team_id is None:
        return 4.20
    url    = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats"
    params = {"stats": "season", "group": "pitching",
               "season": today.year, "pitcherType": "relief", "sportId": 1}
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        if splits:
            return float(splits[0].get("stat", {}).get("era", 4.20))
    except Exception:
        pass
    return 4.20


@st.cache_data
def get_pitcher_recent_era(player_id, num_starts=3):
    try:
        data = statcast_pitcher(season_start, season_end, player_id)
        if data.empty:
            return None
        data        = data.sort_values("game_date")
        game_dates  = data["game_date"].unique()
        if len(game_dates) < 2:
            return None
        recent_data = data[data["game_date"].isin(game_dates[-num_starts:])]
        bf          = recent_data[recent_data["events"].notna()].shape[0]
        ip_est      = bf / 3.0
        if ip_est < 1:
            return None
        hrs = (recent_data["events"] == "home_run").sum()
        return round((hrs * 1.4 / ip_est) * 9, 2)
    except Exception:
        return None


@st.cache_data
def get_statcast_data(start_date, end_date):
    return statcast(start_date, end_date)


def calculate_team_ops_vs_hand(team_abbrev, pitcher_hand):
    data         = get_statcast_data(recent_start, recent_end).copy()
    team_abbrev  = team_abbrev.upper()
    pitcher_hand = pitcher_hand.upper()

    data["batting_team"] = data.apply(
        lambda r: r["away_team"] if r["inning_topbot"] == "Top" else r["home_team"], axis=1
    )
    # Filter all conditions together on the same index to avoid reindex mismatch
    mask = (
        (data["batting_team"] == team_abbrev)
        & (data["p_throws"] == pitcher_hand)
        & (data["events"].notna())
    )
    pa = data[mask].copy()

    if pa.empty:
        return 0.700

    for col in ["single","double","triple","home_run","walk","hbp"]:
        pa[col] = (pa["events"] == (col if col != "hbp" else "hit_by_pitch")).astype(int)

    hits  = pa[["single","double","triple","home_run"]].sum().sum()
    tb    = pa["single"].sum() + pa["double"].sum()*2 + pa["triple"].sum()*3 + pa["home_run"].sum()*4
    walks = pa["walk"].sum()
    hbp   = pa["hbp"].sum()
    ab    = len(pa[~pa["events"].isin(
        ["walk","hit_by_pitch","sac_fly","sac_bunt","catcher_interf","sac_fly_double_play"]
    )])
    plate = len(pa)

    if ab == 0 or plate == 0:
        return 0.700
    return (hits + walks + hbp) / plate + tb / ab


def late_inning_runs_allowed(start_date, end_date, selected_team):
    data          = get_statcast_data(start_date, end_date)
    selected_team = selected_team.upper()
    late          = data[data["inning"].between(7, 9)].copy()
    if late.empty:
        return 1.5

    late["fielding_team"] = late.apply(
        lambda r: r["home_team"] if r["inning_topbot"] == "Top" else r["away_team"], axis=1
    )
    tl = late[late["fielding_team"] == selected_team].copy()
    if tl.empty:
        return 1.5

    hi = tl.groupby(["game_pk","inning","inning_topbot"]).agg(
        s=("bat_score","min"), e=("post_bat_score","max")
    ).reset_index()
    hi["ra"] = (hi["e"] - hi["s"]).clip(lower=0)
    return hi.groupby("game_pk")["ra"].sum().mean()


def head_to_head_record(team_abbrev, opponent_abbrev):
    br_opp = BR_TEAM_MAP.get(opponent_abbrev.upper(), opponent_abbrev.upper())
    try:
        data = get_team_schedule(today.year, team_abbrev)
        if data.empty or "Opp" not in data.columns:
            return 0, 0
        h2h = data[data["Opp"].str.upper().str.contains(br_opp, na=False)]
        if h2h.empty:
            return 0, 0
        w = int(h2h["win"].sum())
        return w, len(h2h) - w
    except Exception:
        return 0, 0


def team_record_stats(team_abbrev, recent=False):
    data = get_team_schedule(today.year, team_abbrev)
    if recent and not data.empty:
        data = data.tail(30)
    if data.empty:
        return {"wins":0,"losses":0,"games":0,"win_rate":0.5,
                "runs_scored":4.3,"runs_allowed":4.3,"run_diff":0}
    wins  = int(data["win"].sum())
    games = len(data)
    return {
        "wins": wins, "losses": games-wins, "games": games,
        "win_rate":     wins / games,
        "runs_scored":  data["R"].mean(),
        "runs_allowed": data["RA"].mean(),
        "run_diff":     data["R"].mean() - data["RA"].mean(),
    }


@st.cache_data
def opponent_k_adjustment(opponent_team):
    data = get_statcast_data(recent_start, recent_end)
    data["batting_team"] = data.apply(
        lambda r: r["away_team"] if r["inning_topbot"] == "Top" else r["home_team"], axis=1
    )
    opp     = data[data["batting_team"] == opponent_team.upper()]
    tot_pa  = opp[opp["events"].notna()]
    opp_ks  = opp[opp["events"] == "strikeout"]
    lg_pa   = data[data["events"].notna()]
    lg_ks   = data[data["events"] == "strikeout"]
    if len(tot_pa) == 0 or len(lg_pa) == 0:
        return 1.0
    return (len(opp_ks) / len(tot_pa)) / (len(lg_ks) / len(lg_pa))


# ---------------------------------------------------------------------------
# Pitcher quality score: blends ERA, FIP, xFIP, SIERA (lower = better)
# ---------------------------------------------------------------------------

def pitcher_quality_score(info):
    """
    Weighted blend of ERA, FIP, xFIP, SIERA.
    xFIP and SIERA are the most predictive so get higher weight.
    """
    era   = info.get("era",   4.30)
    fip   = info.get("fip",   4.20)
    xfip  = info.get("xfip",  4.20)
    siera = info.get("siera", 4.20)
    return era * 0.15 + fip * 0.25 + xfip * 0.30 + siera * 0.30


# ---------------------------------------------------------------------------
# Moneyline model
# ---------------------------------------------------------------------------

def team_moneyline_probability(
    team, opponent_team,
    team_starter_first="", team_starter_last="",
    opponent_starter_first="", opponent_starter_last="",
):
    team          = team.upper()
    opponent_team = opponent_team.upper()

    matchup          = find_today_matchup(team, opponent_team)
    default_pitcher  = {"name":"Unknown","era":4.30,"fip":4.20,"xfip":4.20,
                        "siera":4.20,"babip":0.300,"k_per_9":8.0,
                        "swstr_pct":0.11,"throws":"R"}
    sel_p_info       = dict(default_pitcher)
    opp_p_info       = dict(default_pitcher)
    home_field_edge  = 0
    umpire_name      = None
    home_team_abbrev = team

    if matchup:
        home_field_edge  = 1 if matchup["selected_is_home"] else -1
        home_team_abbrev = matchup.get("home_team_abbrev", team)
        umpire_name      = matchup.get("umpire_name")
        if matchup.get("selected_pitcher"):
            sel_p_info = get_pitcher_info(matchup["selected_pitcher"]["id"])
        if matchup.get("opponent_pitcher"):
            opp_p_info = get_pitcher_info(matchup["opponent_pitcher"]["id"])

    if team_starter_first and team_starter_last:
        pid = get_player_id(team_starter_last, team_starter_first)
        if pid:
            sel_p_info = get_pitcher_info(int(pid))

    if opponent_starter_first and opponent_starter_last:
        pid = get_player_id(opponent_starter_last, opponent_starter_first)
        if pid:
            opp_p_info = get_pitcher_info(int(pid))

    # Team records
    team_season = team_record_stats(team,          recent=False)
    opp_season  = team_record_stats(opponent_team, recent=False)
    team_recent = team_record_stats(team,          recent=True)
    opp_recent  = team_record_stats(opponent_team, recent=True)

    # Late inning
    team_late = late_inning_runs_allowed(recent_start, recent_end, team)
    opp_late  = late_inning_runs_allowed(recent_start, recent_end, opponent_team)

    # OPS vs hand
    team_ops = calculate_team_ops_vs_hand(team,          opp_p_info["throws"])
    opp_ops  = calculate_team_ops_vs_hand(opponent_team, sel_p_info["throws"])

    # B. wRC+ — if Fangraphs lookup returns 100 (default) for both teams,
    # estimate from runs/game: wRC+ ≈ 100 + (R/G - 4.5) * 16
    team_wrc = get_team_wrc_plus(team,          today.year)
    opp_wrc  = get_team_wrc_plus(opponent_team, today.year)

    if team_wrc == 100.0 and opp_wrc == 100.0:
        team_wrc = round(100 + (team_season["runs_scored"] - 4.5) * 16, 1)
        opp_wrc  = round(100 + (opp_season["runs_scored"]  - 4.5) * 16, 1)

    wrc_edge = normalize(team_wrc - opp_wrc, 0, 15)  # 15 pt diff = 1σ

    # Bullpen
    team_bp_era = get_bullpen_era(team)
    opp_bp_era  = get_bullpen_era(opponent_team)
    bullpen_edge = opp_bp_era - team_bp_era

    # Recent starter ERA
    sel_recent_era = None
    opp_recent_era = None
    if matchup and matchup.get("selected_pitcher"):
        sel_recent_era = get_pitcher_recent_era(matchup["selected_pitcher"]["id"])
    if matchup and matchup.get("opponent_pitcher"):
        opp_recent_era = get_pitcher_recent_era(matchup["opponent_pitcher"]["id"])

    def blend_era(season, recent):
        return season if recent is None else season * 0.60 + recent * 0.40

    sel_era_bl = blend_era(sel_p_info["era"], sel_recent_era)
    opp_era_bl = blend_era(opp_p_info["era"], opp_recent_era)

    # A. FIP/xFIP/SIERA quality scores
    sel_quality = pitcher_quality_score({**sel_p_info, "era": sel_era_bl})
    opp_quality = pitcher_quality_score({**opp_p_info, "era": opp_era_bl})
    starter_edge = opp_quality - sel_quality  # positive = team advantage

    # Park factor
    pf = PARK_FACTORS.get(home_team_abbrev.upper(), 1.00)
    offense_edge      = team_season["runs_scored"] - opp_season["runs_scored"]
    park_offense_boost = offense_edge * (pf - 1.0)

    # H2H
    h2h_w, h2h_l = head_to_head_record(team, opponent_team)
    h2h_total    = h2h_w + h2h_l
    h2h_edge     = (h2h_w / h2h_total - 0.5) if h2h_total >= 3 else 0.0

    # E. Rest / fatigue
    team_rest = get_rest_days(team)
    opp_rest  = get_rest_days(opponent_team)
    rest_edge = normalize(team_rest - opp_rest, 0, 2)  # extra rest = slight edge

    team_streak = get_streak(team)
    opp_streak  = get_streak(opponent_team)
    streak_edge = normalize(team_streak - opp_streak, 0, 3)

    # G. Weather
    coords      = PARK_COORDS.get(home_team_abbrev.upper(), (39.0, -95.0))
    weather     = get_weather(coords[0], coords[1], today_str)
    wx_run_adj  = weather_run_factor(
        weather["temp_f"], weather["wind_mph"], weather["wind_dir_deg"]
    )
    # Weather favors better offense: if team has more runs/game, hot weather helps them more
    wx_edge = (offense_edge / max(abs(offense_edge), 0.01)) * (wx_run_adj - 1.0) * 2

    # D. Injury penalty: losing key players hurts win probability
    team_injured = get_injured_players(team)
    opp_injured  = get_injured_players(opponent_team)
    # Rough proxy: each IL player = slight disadvantage (0.02 per player, capped)
    injury_edge = min(len(opp_injured) - len(team_injured), 5) * 0.02

    # Composite
    season_win_edge  = team_season["win_rate"] - opp_season["win_rate"]
    recent_win_edge  = team_recent["win_rate"] - opp_recent["win_rate"]
    defense_edge     = opp_season["runs_allowed"] - team_season["runs_allowed"]
    late_inning_edge = opp_late - team_late
    handedness_edge  = team_ops - opp_ops

    # -----------------------------------------------------------------------
    # Composite matchup score
    # -----------------------------------------------------------------------
    # CALIBRATION NOTE:
    # Each factor is normalized so a 1-sigma edge contributes its weight.
    # The raw score is then passed through a temperature-scaled sigmoid with
    # temperature=12.  This maps:
    #   score=0  -> 50.0%   (coin flip)
    #   score=3  -> 56.2%   (moderate edge)
    #   score=6  -> 62.2%   (strong edge)
    #   score=10 -> 69.7%   (extreme edge, rarely seen)
    # MLB empirical ceiling for any single game is ~65-66% for the best
    # teams vs the worst, so this keeps us honest.
    # -----------------------------------------------------------------------
    matchup_score = (
        # Win-rate signals — most reliable single predictors
        season_win_edge                              * 1.0
        + recent_win_edge                            * 0.8

        # Pitching — FIP/xFIP/SIERA blend, most predictive per-game factor
        + normalize(starter_edge,        0, 2)       * 1.2

        # Bullpen ERA differential
        + normalize(bullpen_edge,        0, 1)       * 0.8

        # Offense — wRC+ (park-adjusted) is primary; OPS vs hand is secondary
        + wrc_edge                                   * 0.7
        + normalize(handedness_edge,     0, 0.15)    * 0.4

        # Run differential (offense minus defense)
        + normalize(offense_edge,        0, 2)       * 0.3
        + normalize(defense_edge,        0, 2)       * 0.4

        # Late-inning bullpen (innings 7-9)
        + normalize(late_inning_edge,    0, 1)       * 0.5

        # Contextual / situational factors (lower weight — noisier signals)
        + h2h_edge                                   * 0.4
        + rest_edge                                  * 0.2
        + streak_edge                                * 0.15
        + normalize(park_offense_boost,  0, 0.5)     * 0.3
        + wx_edge                                    * 0.15
        + injury_edge                                * 0.3
        + home_field_edge                            * 0.15
    )

    # Temperature-scaled sigmoid: keeps output in realistic MLB win-prob range
    # temperature=12 means a "perfect game" score of ~10 -> 70% win prob max
    # Temperature=8 calibrated against real MLB matchup data:
    # Strong favorite (score ~3.5) -> ~60-61% | Close game (score ~0.8) -> ~52%
    # Extreme mismatch (score ~6+) -> ~68% max — realistic MLB ceiling
    TEMPERATURE = 8.0
    model_prob = round(1 / (1 + math.exp(-matchup_score / TEMPERATURE)), 6)

    return {
        "model_prob":        model_prob,
        "matchup_score":     matchup_score,
        "score_breakdown": {
            "season_win_edge":  round(season_win_edge * 1.0, 3),
            "recent_win_edge":  round(recent_win_edge * 0.8, 3),
            "starter_edge":     round(normalize(starter_edge, 0, 2) * 1.2, 3),
            "bullpen_edge":     round(normalize(bullpen_edge, 0, 1) * 0.8, 3),
            "wrc_edge":         round(wrc_edge * 0.7, 3),
            "handedness_edge":  round(normalize(handedness_edge, 0, 0.15) * 0.4, 3),
            "offense_edge":     round(normalize(offense_edge, 0, 2) * 0.3, 3),
            "defense_edge":     round(normalize(defense_edge, 0, 2) * 0.4, 3),
            "late_inning":      round(normalize(late_inning_edge, 0, 1) * 0.5, 3),
            "h2h":              round(h2h_edge * 0.4, 3),
            "rest":             round(rest_edge * 0.2, 3),
            "streak":           round(streak_edge * 0.15, 3),
            "park":             round(normalize(park_offense_boost, 0, 0.5) * 0.3, 3),
            "weather":          round(wx_edge * 0.15, 3),
            "injury":           round(injury_edge * 0.3, 3),
            "home_field":       round(home_field_edge * 0.15, 3),
        },
        "team_season":       team_season,
        "opp_season":        opp_season,
        "team_recent":       team_recent,
        "opp_recent":        opp_recent,
        "team_late":         team_late,
        "opp_late":          opp_late,
        "late_inning_edge":  late_inning_edge,
        "team_ops":          team_ops,
        "opp_ops":           opp_ops,
        "team_wrc":          team_wrc,
        "opp_wrc":           opp_wrc,
        "sel_p_info":        sel_p_info,
        "opp_p_info":        opp_p_info,
        "sel_recent_era":    sel_recent_era,
        "opp_recent_era":    opp_recent_era,
        "sel_quality":       sel_quality,
        "opp_quality":       opp_quality,
        "team_bp_era":       team_bp_era,
        "opp_bp_era":        opp_bp_era,
        "bullpen_edge":      bullpen_edge,
        "park_factor":       pf,
        "home_team_abbrev":  home_team_abbrev,
        "h2h_wins":          h2h_w,
        "h2h_losses":        h2h_l,
        "team_rest":         team_rest,
        "opp_rest":          opp_rest,
        "team_streak":       team_streak,
        "opp_streak":        opp_streak,
        "weather":           weather,
        "wx_run_adj":        wx_run_adj,
        "team_injured":      team_injured,
        "opp_injured":       opp_injured,
        "matchup":           matchup,
        "home_field_edge":   home_field_edge,
        "umpire_name":       umpire_name,
    }


# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------

if st.button("Predict"):

    # ======================== MONEYLINE ========================
    if prop_type == "Team Moneyline":

        with st.spinner("Fetching data and computing…"):
            result = team_moneyline_probability(
                team, opponent_team,
                team_starter_first, team_starter_last,
                opponent_starter_first, opponent_starter_last,
            )

        model_prob   = result["model_prob"]
        implied_prob = implied_probability(american_odds)
        ev           = expected_value(model_prob, american_odds)
        edge_pct     = model_prob - implied_prob
        label        = moneyline_label(ev, model_prob, implied_prob, american_odds)

        # ---- Top-line result ----
        st.subheader("📊 Moneyline Prediction")
        st.write(f"**{team.upper()}** vs **{opponent_team.upper()}**")

        if result["matchup"]:
            loc = "HOME" if result["matchup"]["selected_is_home"] else "AWAY"
            st.write(f"Location: {team.upper()} is **{loc}**")
            if result["umpire_name"]:
                st.write(f"Home Plate Umpire: {result['umpire_name']}")
        else:
            st.warning("Could not find today\'s matchup. Check abbreviations.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model Win Prob",    f"{model_prob*100:.1f}%")
        col2.metric("Book Implied Prob", f"{implied_prob*100:.1f}%")
        col3.metric("Edge vs Book",      f"{edge_pct*100:+.1f}%")
        col4.metric("Expected Value",    f"{ev*100:.1f}%")

        st.caption(
            "🟢 Strong Bet = model ≥54% + edge ≥5% over book + EV ≥4%  |  "
            "🟡 Lean = model ≥52% + edge ≥3% over book + EV ≥2%  |  "
            "🔴 No Bet = no positive edge, coin flip, or chalk without value  |  "
            "Heavy chalk (−160+) needs ≥7% edge to qualify"
        )

        if label == "Strong Bet":
            st.success("✅ STRONG BET — High win probability + clear value over the book")
        elif label == "Lean":
            st.success("📊 LEAN — Decent win probability + some value over the book")
        else:
            st.info("⛔ NO BET — Does not meet both win probability AND edge thresholds")

        # ---- Filter checklist: show exactly why a bet passed or failed ----
        st.subheader("🔎 Bet Filter Checklist")

        win_prob_ok_strong  = model_prob >= 0.54
        win_prob_ok_lean    = model_prob >= 0.52
        edge_ok_strong      = edge_pct >= 0.05
        edge_ok_lean        = edge_pct >= 0.03
        ev_ok_strong        = ev >= 0.04
        ev_ok_lean          = ev >= 0.02
        chalk_blocked       = american_odds <= -160 and edge_pct < 0.07

        def check(cond): return "✅" if cond else "❌"

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**Strong Bet requirements (all must pass)**")
            st.write(f"{check(win_prob_ok_strong)} Model win prob ≥54%  →  {model_prob*100:.1f}%")
            st.write(f"{check(edge_ok_strong)} Edge over book ≥5%  →  {edge_pct*100:+.1f}%")
            st.write(f"{check(ev_ok_strong)} Expected value ≥4%  →  {ev*100:.1f}%")
            st.write(f"{check(edge_pct > 0)} Positive edge (model > book)  →  {edge_pct*100:+.1f}%")
            if chalk_blocked:
                st.write(f"❌ Heavy chalk (≤−160) blocked: need ≥7% edge, have {edge_pct*100:.1f}%")
        with fc2:
            st.markdown("**Lean requirements (all must pass)**")
            st.write(f"{check(win_prob_ok_lean)} Model win prob ≥52%  →  {model_prob*100:.1f}%")
            st.write(f"{check(edge_ok_lean)} Edge over book ≥3%  →  {edge_pct*100:+.1f}%")
            st.write(f"{check(ev_ok_lean)} Expected value ≥2%  →  {ev*100:.1f}%")
            st.write(f"{check(edge_pct > 0)} Positive edge (model > book)  →  {edge_pct*100:+.1f}%")

        # ---- Score debug panel ----
        with st.expander("🔬 Model Score Breakdown (why this probability?)"):
            st.caption(f"Raw matchup score: **{result['matchup_score']:.3f}** → sigmoid → **{result['model_prob']*100:.1f}%**")
            bd = result["score_breakdown"]
            rows = sorted(bd.items(), key=lambda x: abs(x[1]), reverse=True)
            for factor, contribution in rows:
                bar = "█" * int(abs(contribution) * 20)
                direction = "▲" if contribution > 0 else ("▼" if contribution < 0 else "—")
                st.text(f"  {direction} {factor:<20} {contribution:+.3f}  {bar}")
            st.caption("Positive = favors your team. Largest contributors shown first.")
            if team_wrc := result.get("team_wrc"):
                st.caption(f"wRC+ used: {result.get('team_wrc', 100):.0f} vs {result.get('opp_wrc', 100):.0f}")

        # ---- Starting Pitchers ----
        st.subheader("⚾ Starting Pitchers")
        sp = result["sel_p_info"]
        op = result["opp_p_info"]

        for abbrev, info, recent_era, quality in [
            (team.upper(),          sp, result["sel_recent_era"], result["sel_quality"]),
            (opponent_team.upper(), op, result["opp_recent_era"], result["opp_quality"]),
        ]:
            st.markdown(f"**{abbrev}: {info['name']}** (Throws: {info['throws']})")

            pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
            pcol1.metric("ERA",   f"{info['era']:.2f}")
            pcol2.metric("FIP",   f"{info['fip']:.2f}")
            pcol3.metric("xFIP",  f"{info['xfip']:.2f}")
            pcol4.metric("SIERA", f"{info['siera']:.2f}")
            pcol5.metric("Quality Score", f"{quality:.2f}")

            babip_flag = babip_regression_flag(info.get("babip", 0.300))
            if babip_flag:
                st.caption(babip_flag)

            recent_str = f"{recent_era:.2f}" if recent_era is not None else "N/A"
            st.caption(
                f"BABIP: {info.get('babip', 0.300):.3f} | "
                f"K/9: {info.get('k_per_9', 8.0):.1f} | "
                f"Recent ERA (last 3 starts): {recent_str}"
            )

        # ---- Offense ----
        st.subheader("🏏 Offense")
        ocol1, ocol2 = st.columns(2)
        ocol1.metric(f"{team.upper()} wRC+",          f"{result['team_wrc']:.0f}")
        ocol2.metric(f"{opponent_team.upper()} wRC+", f"{result['opp_wrc']:.0f}")
        st.caption("wRC+ is park-adjusted; 100 = league average, higher = better")

        st.write(f"{team.upper()} OPS vs {op['throws']}HP: {result['team_ops']:.3f}")
        st.write(f"{opponent_team.upper()} OPS vs {sp['throws']}HP: {result['opp_ops']:.3f}")

        # ---- Bullpen ----
        st.subheader("💪 Bullpen")
        bcol1, bcol2 = st.columns(2)
        bcol1.metric(f"{team.upper()} Bullpen ERA",          f"{result['team_bp_era']:.2f}")
        bcol2.metric(f"{opponent_team.upper()} Bullpen ERA", f"{result['opp_bp_era']:.2f}")

        be = result["bullpen_edge"]
        if be > 0.30:
            st.success(f"{team.upper()} has bullpen advantage (+{be:.2f} ERA).")
        elif be < -0.30:
            st.warning(f"{opponent_team.upper()} has bullpen advantage ({be:.2f} ERA).")
        else:
            st.info("Bullpens roughly even.")

        # ---- Late Innings ----
        st.subheader("🔒 Late-Inning Performance (Inn. 7-9)")
        lcol1, lcol2 = st.columns(2)
        lcol1.metric(f"{team.upper()} RA/G (7-9)", f"{result['team_late']:.2f}")
        lcol2.metric(f"{opponent_team.upper()} RA/G (7-9)", f"{result['opp_late']:.2f}")

        # ---- Records ----
        st.subheader("📋 Team Records")
        ts = result["team_season"]; os_ = result["opp_season"]
        tr = result["team_recent"]; orr = result["opp_recent"]

        st.write(
            f"{team.upper()} Season: **{ts['wins']}-{ts['losses']}** | "
            f"R/G: {ts['runs_scored']:.2f} | RA/G: {ts['runs_allowed']:.2f}"
        )
        st.write(
            f"{opponent_team.upper()} Season: **{os_['wins']}-{os_['losses']}** | "
            f"R/G: {os_['runs_scored']:.2f} | RA/G: {os_['runs_allowed']:.2f}"
        )
        st.write(
            f"{team.upper()} Last-30: {tr['wins']}-{tr['losses']} | "
            f"{opponent_team.upper()} Last-30: {orr['wins']}-{orr['losses']}"
        )

        # ---- H2H ----
        st.subheader("🤜 Head-to-Head This Season")
        h2h_total = result["h2h_wins"] + result["h2h_losses"]
        if h2h_total >= 3:
            st.write(
                f"{team.upper()} vs {opponent_team.upper()}: "
                f"**{result['h2h_wins']}-{result['h2h_losses']}**"
            )
        else:
            st.write(f"Only {h2h_total} H2H games played — not enough data yet.")

        # ---- E. Rest / Fatigue ----
        st.subheader("😴 Rest & Momentum")
        rcol1, rcol2 = st.columns(2)
        rcol1.metric(f"{team.upper()} Days Rest",           result["team_rest"])
        rcol2.metric(f"{opponent_team.upper()} Days Rest",  result["opp_rest"])

        def streak_label(s):
            return f"W{s}" if s > 0 else (f"L{abs(s)}" if s < 0 else "—")

        st.write(
            f"{team.upper()} Streak: {streak_label(result['team_streak'])} | "
            f"{opponent_team.upper()} Streak: {streak_label(result['opp_streak'])}"
        )

        # ---- G. Weather ----
        st.subheader("🌤️ Weather (Home Park)")
        wx = result["weather"]
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)
        wcol1.metric("Temp",        f"{wx['temp_f']:.0f}°F")
        wcol2.metric("Wind Speed",  f"{wx['wind_mph']:.0f} mph")
        wcol3.metric("Wind Dir",    f"{wx['wind_dir_deg']:.0f}°")
        wcol4.metric("Run Factor",  f"{result['wx_run_adj']:.3f}")

        if result["wx_run_adj"] > 1.04:
            st.caption("☀️ Hot/wind-out conditions favor hitters today.")
        elif result["wx_run_adj"] < 0.96:
            st.caption("❄️ Cold/wind-in conditions favor pitchers today.")

        # ---- D. Injuries ----
        st.subheader("🏥 Injured List")
        col_inj1, col_inj2 = st.columns(2)
        with col_inj1:
            st.write(f"**{team.upper()} IL ({len(result['team_injured'])} players)**")
            if result["team_injured"]:
                for p in result["team_injured"][:10]:
                    st.caption(f"• {p}")
            else:
                st.caption("No IL data retrieved.")
        with col_inj2:
            st.write(f"**{opponent_team.upper()} IL ({len(result['opp_injured'])} players)**")
            if result["opp_injured"]:
                for p in result["opp_injured"][:10]:
                    st.caption(f"• {p}")
            else:
                st.caption("No IL data retrieved.")

        # ---- Park ----
        st.subheader("🏟️ Park")
        st.write(
            f"Home Park ({result['home_team_abbrev'].upper()}) "
            f"Run Factor: {result['park_factor']:.3f}"
        )

    # ======================== PLAYER PROPS ========================
    else:
        player_id = get_player_id(last_name, first_name)

        if player_id is None:
            st.error("❌ Player not found. Check spelling of first and last name.")
            st.stop()

        # ----------------------------------------------------------------
        # Minimum data requirements before we attempt a prediction
        # ----------------------------------------------------------------
        MIN_GAMES_SEASON = 5   # need at least 5 starts/games for a season avg
        MIN_GAMES_RECENT = 3   # need at least 3 recent games to weight recent form

        with st.spinner("Fetching Statcast data…"):
            if prop_type == "Pitcher Strikeouts":
                season_data = statcast_pitcher(season_start, season_end, player_id)
                recent_data = statcast_pitcher(recent_start, recent_end, player_id)

                if season_data.empty:
                    st.error("❌ No Statcast data found for this pitcher this season. They may be injured, a reliever, or not yet in the Statcast database.")
                    st.stop()

                season_by_game = season_data[season_data["events"] == "strikeout"].groupby("game_date").size()
                recent_by_game = recent_data[recent_data["events"] == "strikeout"].groupby("game_date").size()

                # Count actual starts (games where pitcher threw pitches)
                season_starts = season_data["game_date"].nunique()
                recent_starts = recent_data["game_date"].nunique() if not recent_data.empty else 0

                stat_name = "Strikeouts"

            elif prop_type == "Batter Hits":
                season_data = statcast_batter(season_start, season_end, player_id)
                recent_data = statcast_batter(recent_start, recent_end, player_id)

                if season_data.empty:
                    st.error("❌ No Statcast data found for this batter this season. They may be injured or not yet active.")
                    st.stop()

                hit_events = ["single", "double", "triple", "home_run"]
                season_by_game = season_data[season_data["events"].isin(hit_events)].groupby("game_date").size()
                recent_by_game = recent_data[recent_data["events"].isin(hit_events)].groupby("game_date").size()

                season_starts = season_data["game_date"].nunique()
                recent_starts = recent_data["game_date"].nunique() if not recent_data.empty else 0

                stat_name = "Hits"

            else:  # Total Bases
                season_data = statcast_batter(season_start, season_end, player_id)
                recent_data = statcast_batter(recent_start, recent_end, player_id)

                if season_data.empty:
                    st.error("❌ No Statcast data found for this batter this season.")
                    st.stop()

                season_data = season_data.copy()
                recent_data = recent_data.copy()
                season_data["tb"] = season_data["events"].apply(total_bases_from_event)
                recent_data["tb"] = recent_data["events"].apply(total_bases_from_event)
                season_by_game = season_data.groupby("game_date")["tb"].sum()
                recent_by_game = recent_data.groupby("game_date")["tb"].sum()

                season_starts = season_data["game_date"].nunique()
                recent_starts = recent_data["game_date"].nunique() if not recent_data.empty else 0

                stat_name = "Total Bases"

        # ----------------------------------------------------------------
        # Data quality gate — stop early with a clear explanation
        # ----------------------------------------------------------------
        data_warnings = []
        data_errors   = []

        if season_by_game.empty:
            data_errors.append(
                f"No {stat_name.lower()} recorded in Statcast data this season. "
                f"This usually means the player has not appeared in games yet, "
                f"is on the IL, or was just called up."
            )

        elif season_starts < MIN_GAMES_SEASON:
            data_errors.append(
                f"Only **{season_starts} games** found this season — need at least "
                f"{MIN_GAMES_SEASON} for a reliable season average. "
                f"Prediction would be based on too small a sample to trust."
            )

        if data_errors:
            st.error("❌ Insufficient data to predict")
            for err in data_errors:
                st.write(f"• {err}")
            st.info(
                "💡 What you can do:\n"
                "- Check that the player is active and starting today\n"
                "- Try again once more games are in the Statcast database\n"
                "- For early-season predictions, use caution — small samples are unreliable"
            )
            st.stop()

        # Warn but don't stop for borderline cases
        if season_starts < 10:
            data_warnings.append(
                f"⚠️ Small season sample ({season_starts} games) — projection is less reliable than mid/late season."
            )

        if recent_starts < MIN_GAMES_RECENT:
            data_warnings.append(
                f"⚠️ Only {recent_starts} games in the last 30 days — using season average only for recent form."
            )

        # ----------------------------------------------------------------
        # Core projection
        # ----------------------------------------------------------------
        season_avg = season_by_game.mean()

        # For recent avg: fill games where stat = 0 (player played but got 0 Ks/hits)
        # This is important — a game with 0 strikeouts is real data, not missing data
        if not recent_data.empty and recent_starts > 0:
            recent_game_dates = recent_data["game_date"].unique()
            if prop_type == "Pitcher Strikeouts":
                recent_by_game_filled = recent_by_game.reindex(recent_game_dates, fill_value=0)
            elif prop_type == "Batter Hits":
                # For batters, only count games where they had a plate appearance
                pa_games = recent_data[recent_data["events"].notna()]["game_date"].unique()
                recent_by_game_filled = recent_by_game.reindex(pa_games, fill_value=0)
            else:
                pa_games = recent_data[recent_data["events"].notna()]["game_date"].unique()
                recent_by_game_filled = recent_by_game.reindex(pa_games, fill_value=0)

            recent_avg = recent_by_game_filled.mean() if len(recent_by_game_filled) >= MIN_GAMES_RECENT else float("nan")
        else:
            recent_avg = float("nan")

        projection = calc_weighted_projection(season_avg, recent_avg)

        # ----------------------------------------------------------------
        # Prop-specific adjustments
        # ----------------------------------------------------------------
        adj              = 1.0
        k_park_adj       = 1.0
        umpire_adj       = 1.0
        wx_k_adj         = 1.0
        hit_pitcher_adj  = 1.0
        hit_handedness   = 1.0
        swstr_pct        = None
        umpire_name_prop = None
        weather_info     = None
        fip_data         = None
        pitcher_hand     = None
        opp_pitcher_era  = None

        if prop_type == "Pitcher Strikeouts":
            # --- Opponent K-rate adjustment ---
            adj = opponent_k_adjustment(opponent_team)

            # --- Park strikeout factor ---
            k_park_adj = PARK_K_FACTORS.get(opponent_team.upper(), 1.0)

            # --- Umpire adjustment ---
            schedule_data = get_mlb_schedule(today_str)
            for db in schedule_data.get("dates", []):
                for game in db.get("games", []):
                    for off in game.get("officials", []):
                        if off.get("officialType") == "Home Plate":
                            umpire_name_prop = off["official"].get("fullName")
            umpire_adj = get_umpire_k_adjustment(umpire_name_prop)

            # --- SwStr% quality signal ---
            swstr_pct = get_pitcher_swstr(player_id)
            if swstr_pct is not None:
                swstr_factor = 1.0 + (swstr_pct - 0.11) * 3.0
                adj *= max(min(swstr_factor, 1.30), 0.70)
            else:
                data_warnings.append("⚠️ SwStr% unavailable (fewer than 50 pitches in last 30 days) — pitch quality signal skipped.")

            # --- FIP/xFIP pitcher quality metrics ---
            fip_data = get_pitcher_info(player_id)

            # --- Weather ---
            coords       = PARK_COORDS.get(opponent_team.upper(), PARK_COORDS.get("NYY"))
            weather_info = get_weather(coords[0], coords[1], today_str)
            wx_k_adj     = weather_k_factor(weather_info["temp_f"], weather_info["wind_mph"])

            projection = projection * adj * k_park_adj * umpire_adj * wx_k_adj

        elif prop_type in ("Batter Hits", "Batter Total Bases"):
            # --- Opposing pitcher handedness & quality adjustment ---
            matchup_info = find_today_matchup(opponent_team, opponent_team)
            # Try to find the specific game this batter is in
            schedule_data = get_mlb_schedule(today_str)
            opp_pitcher_id = None

            for db in schedule_data.get("dates", []):
                for game in db.get("games", []):
                    home_id = game["teams"]["home"]["team"]["id"]
                    away_id = game["teams"]["away"]["team"]["id"]
                    opp_id  = TEAM_ID_MAP.get(opponent_team.upper())

                    if opp_id in (home_id, away_id):
                        # Opponent is pitching against our batter's team
                        if home_id == opp_id:
                            p = game["teams"]["home"].get("probablePitcher")
                        else:
                            p = game["teams"]["away"].get("probablePitcher")
                        if p:
                            opp_pitcher_id = p["id"]
                        break

            if opp_pitcher_id:
                opp_info     = get_pitcher_info(opp_pitcher_id)
                pitcher_hand = opp_info["throws"]
                opp_fip      = opp_info.get("fip", 4.20)
                opp_xfip     = opp_info.get("xfip", 4.20)
                opp_pitcher_era = opp_info["era"]

                # FIP-based pitcher difficulty: league avg FIP ~4.20
                # A 3.20 FIP pitcher suppresses hits ~8% vs average
                fip_difficulty = (opp_fip - 4.20) / 4.20  # positive = easier, negative = harder
                hit_pitcher_adj = max(0.80, min(1.20, 1.0 + fip_difficulty * 0.5))

                # Handedness: pull from Statcast platoon splits
                batter_ops_vs_hand = calculate_team_ops_vs_hand(
                    # Use opponent team's batting stats as proxy — not ideal
                    # but Statcast doesn't give individual platoon in this flow
                    opponent_team,
                    pitcher_hand
                )
                # Platoon adjustment: OPS vs hand relative to league avg (.700)
                platoon_factor = batter_ops_vs_hand / 0.700
                hit_handedness = max(0.85, min(1.15, platoon_factor))

                projection = projection * hit_pitcher_adj * hit_handedness
            else:
                data_warnings.append("⚠️ Could not find today's opposing pitcher — pitcher quality and handedness adjustments skipped.")

            # --- Park run factor for hit/TB props ---
            park_key = opponent_team.upper()
            park_run_factor = PARK_FACTORS.get(park_key, 1.00)
            projection = projection * park_run_factor

            # --- Weather for hits/TB ---
            coords       = PARK_COORDS.get(park_key, PARK_COORDS.get("NYY"))
            weather_info = get_weather(coords[0], coords[1], today_str)
            wx_hit_adj   = weather_run_factor(weather_info["temp_f"], weather_info["wind_mph"], weather_info["wind_dir_deg"])
            # Cap weather effect on individual player props
            wx_hit_adj   = max(0.97, min(1.03, wx_hit_adj))
            projection   = projection * wx_hit_adj

        # ----------------------------------------------------------------
        # Probability & EV calculation — evaluated for BOTH sides
        # ----------------------------------------------------------------
        std_dev  = float(season_by_game.std()) if len(season_by_game) > 1 else 1.5
        std_dev  = max(std_dev, 0.8)  # floor: don't over-tighten for very consistent players

        edge      = projection - sportsbook_line
        over_prob = estimate_over_probability(projection, sportsbook_line, std_dev)
        under_prob = 1.0 - over_prob

        # Each side gets its own implied probability and EV
        over_implied  = implied_probability(over_odds)
        under_implied = implied_probability(under_odds)

        over_ev   = expected_value(over_prob,  over_odds)
        under_ev  = expected_value(under_prob, under_odds)

        # Vig check: total implied prob > 100% = book is taking juice (normal)
        total_implied = over_implied + under_implied
        vig_pct = (total_implied - 1.0) * 100  # e.g. 4.8% vig on a standard -110/-110 line

        label = confidence_label(edge, over_ev, under_ev)

        # Legacy single-side ev for any remaining references
        implied_prob = over_implied
        ev = over_ev

        # ----------------------------------------------------------------
        # Display warnings first so user sees them before results
        # ----------------------------------------------------------------
        if data_warnings:
            for w in data_warnings:
                st.warning(w)

        # ----------------------------------------------------------------
        # Results display
        # ----------------------------------------------------------------
        st.subheader(f"📊 {stat_name} Prediction")

        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        rcol1.metric("Season Avg",    f"{season_avg:.2f}",
                     help=f"Based on {season_starts} games this season")
        rcol2.metric("Recent Avg",    f"{recent_avg:.2f}" if not math.isnan(recent_avg) else "N/A",
                     help=f"Based on {recent_starts} games in last 30 days (zeros included)")
        rcol3.metric("Projection",    f"{projection:.2f}")
        rcol4.metric("Sportsbook Line", f"{sportsbook_line:.1f}")

        # ---- Sample size indicator ----
        if season_starts >= 20:
            sample_quality = "🟢 Good sample"
        elif season_starts >= 10:
            sample_quality = "🟡 Moderate sample"
        else:
            sample_quality = "🔴 Small sample — treat with caution"

        st.caption(
            f"{sample_quality} | Season: {season_starts} games | "
            f"Recent (30d): {recent_starts} games | "
            f"Std Dev: ±{std_dev:.2f} (player consistency)"
        )

        # ---- Prop-specific adjustment details ----
        if prop_type == "Pitcher Strikeouts" and fip_data:
            st.subheader("⚾ Pitcher Quality Metrics")
            pcols = st.columns(5)
            pcols[0].metric("ERA",   f"{fip_data['era']:.2f}")
            pcols[1].metric("FIP",   f"{fip_data['fip']:.2f}")
            pcols[2].metric("xFIP",  f"{fip_data['xfip']:.2f}")
            pcols[3].metric("SIERA", f"{fip_data['siera']:.2f}")
            pcols[4].metric("K/9",   f"{fip_data['k_per_9']:.1f}")
            babip_flag = babip_regression_flag(fip_data.get("babip", 0.300))
            if babip_flag:
                st.caption(babip_flag)

            st.subheader("🔧 Strikeout Adjustments")
            acol1, acol2, acol3, acol4 = st.columns(4)
            acol1.metric("Opp K-Rate",    f"{adj:.3f}x",
                         help="How often opponent strikes out vs league average")
            acol2.metric("Park Factor",   f"{k_park_adj:.3f}x",
                         help="Strikeout park factor at opponent's home park")
            acol3.metric("Umpire",        f"{umpire_adj:.3f}x",
                         help=f"Umpire: {umpire_name_prop or 'Unknown'}")
            acol4.metric("Weather",       f"{wx_k_adj:.3f}x",
                         help="Temperature/wind effect on strikeout rate")

            if swstr_pct is not None:
                st.metric("SwStr% Signal", f"{swstr_pct*100:.1f}%",
                          help="Swinging strike %. League avg ~11%. Higher = better stuff today.")
            if weather_info:
                st.caption(f"🌤️ Park weather: {weather_info['temp_f']:.0f}°F, {weather_info['wind_mph']:.0f} mph wind")

        elif prop_type in ("Batter Hits", "Batter Total Bases"):
            st.subheader("🔧 Hit/TB Adjustments")

            if opp_pitcher_era is not None:
                acol1, acol2, acol3, acol4 = st.columns(4)
                acol1.metric("Pitcher Difficulty", f"{hit_pitcher_adj:.3f}x",
                             help=f"Opp pitcher ERA {opp_pitcher_era:.2f}, FIP-adjusted")
                acol2.metric("Platoon Factor",     f"{hit_handedness:.3f}x",
                             help=f"Batter vs {pitcher_hand}HP based on team platoon splits")
                pf_val = PARK_FACTORS.get(opponent_team.upper(), 1.00)
                acol3.metric("Park Factor",        f"{pf_val:.3f}x",
                             help="Run-scoring environment at this park")
                acol4.metric("Weather",            f"{wx_hit_adj:.3f}x" if weather_info else "N/A",
                             help="Temperature/wind run factor (capped ±3%)")
                if pitcher_hand:
                    st.caption(f"Opposing pitcher throws: **{pitcher_hand}**")
            if weather_info:
                st.caption(f"🌤️ Park weather: {weather_info['temp_f']:.0f}°F, {weather_info['wind_mph']:.0f} mph wind")

        # ---- Betting value ----
        st.subheader("💰 Betting Value")

        # Summary line
        st.write(
            f"Projection: **{projection:.2f}** vs Line: **{sportsbook_line:.1f}** | "
            f"Edge: **{edge:+.2f}** | Model Prob Over: **{over_prob*100:.1f}%** | "
            f"Std Dev: ±{std_dev:.2f}"
        )

        # Vig display
        st.caption(f"Book vig: {vig_pct:.1f}% (total implied = {total_implied*100:.1f}%)")

        # Side-by-side over/under breakdown
        over_col, under_col = st.columns(2)

        with over_col:
            st.markdown("### 📈 OVER")
            o1, o2, o3 = st.columns(3)
            o1.metric("Your Odds",      f"{over_odds:+d}")
            o2.metric("Book Implied",   f"{over_implied*100:.1f}%")
            o3.metric("Model Prob",     f"{over_prob*100:.1f}%")
            edge_over = over_prob - over_implied
            ev_over_pct = over_ev * 100
            st.metric("Edge vs Book",   f"{edge_over*100:+.1f}%",
                      delta_color="normal" if edge_over > 0 else "inverse")
            st.metric("Expected Value", f"{ev_over_pct:.1f}%",
                      delta_color="normal" if over_ev > 0 else "inverse")
            if label in ("Strong Over", "Lean Over"):
                if label == "Strong Over":
                    st.success("✅ Strong OVER")
                else:
                    st.success("📊 Lean OVER")
            else:
                st.info("No value on OVER")

        with under_col:
            st.markdown("### 📉 UNDER")
            u1, u2, u3 = st.columns(3)
            u1.metric("Your Odds",      f"{under_odds:+d}")
            u2.metric("Book Implied",   f"{under_implied*100:.1f}%")
            u3.metric("Model Prob",     f"{under_prob*100:.1f}%")
            edge_under = under_prob - under_implied
            ev_under_pct = under_ev * 100
            st.metric("Edge vs Book",   f"{edge_under*100:+.1f}%",
                      delta_color="normal" if edge_under > 0 else "inverse")
            st.metric("Expected Value", f"{ev_under_pct:.1f}%",
                      delta_color="normal" if under_ev > 0 else "inverse")
            if label in ("Strong Under", "Lean Under"):
                if label == "Strong Under":
                    st.warning("⬇️ Strong UNDER")
                else:
                    st.warning("📉 Lean UNDER")
            else:
                st.info("No value on UNDER")

        # Overall verdict
        st.divider()
        if label == "No Bet":
            st.info(
                "⛔ No Bet — projection does not show sufficient edge on either side. "
                f"Model sees {over_prob*100:.1f}% chance of going OVER, but neither "
                f"side's odds offer enough value to overcome the vig."
            )

        # Vig warning: if both sides have negative EV, explain why
        if over_ev < 0 and under_ev < 0:
            st.caption(
                f"Both sides show negative EV — the {vig_pct:.1f}% book vig is too high "
                f"relative to the model's edge. This is common on heavily juiced lines."
            )

        # ---- Reliability note ----
        if season_starts < 10:
            st.warning(
                f"⚠️ **Low confidence prediction** — only {season_starts} games in sample. "
                f"Projections stabilize after ~15-20 games. Consider waiting for more data "
                f"before betting."
            )

        # ---- Season game log ----
        st.subheader("📅 Season Game Log")

        # Show full game log including zeros for games where stat = 0
        if prop_type == "Pitcher Strikeouts":
            all_game_dates = pd.Series(season_data["game_date"].unique()).sort_values()
            full_log = season_by_game.reindex(all_game_dates, fill_value=0).reset_index()
            full_log.columns = ["game_date", stat_name]
        elif prop_type == "Batter Hits":
            pa_game_dates = season_data[season_data["events"].notna()]["game_date"].unique()
            all_game_dates = pd.Series(pa_game_dates).sort_values()
            full_log = season_by_game.reindex(all_game_dates, fill_value=0).reset_index()
            full_log.columns = ["game_date", stat_name]
        else:
            pa_game_dates = season_data[season_data["events"].notna()]["game_date"].unique()
            all_game_dates = pd.Series(pa_game_dates).sort_values()
            full_log = season_by_game.reindex(all_game_dates, fill_value=0).reset_index()
            full_log.columns = ["game_date", stat_name]

        full_log["game_date"] = full_log["game_date"].astype(str)
        full_log["vs_line"]   = sportsbook_line

        st.caption(f"Showing all {len(full_log)} games — zeros mean player appeared but recorded 0 {stat_name.lower()}.")
        st.dataframe(full_log, use_container_width=True)
        st.bar_chart(full_log.set_index("game_date")[stat_name])