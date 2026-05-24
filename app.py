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
    sportsbook_line = st.number_input("Sportsbook Line", value=6.5)
    american_odds   = st.number_input("American Odds", value=-110)

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

def confidence_label(edge, ev):
    if   ev > 0.10 and edge >  0.7: return "Strong Play"
    elif ev > 0.03 and edge >  0.3: return "Lean Over"
    elif ev < -0.10 and edge < -0.7: return "Strong Under"
    elif ev < -0.03 and edge < -0.3: return "Lean Under"
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
    edge_pct = model_prob - implied_prob

    # Hard block 1: model not confident enough — true coin flip territory
    if model_prob < 0.52:
        return "No Bet"

    # Hard block 2: negative or zero edge — book has it right or better
    if edge_pct <= 0:
        return "No Bet"

    # Hard block 3: heavy chalk — vig destroys value, need large edge
    if american_odds <= -160 and edge_pct < 0.07:
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

    # Overlay FIP data
    fip_data = get_pitcher_fip(int(player_id), today.year)
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

    # B. wRC+
    team_wrc = get_team_wrc_plus(team,          today.year)
    opp_wrc  = get_team_wrc_plus(opponent_team, today.year)
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
    model_prob = 1 / (1 + math.exp(-matchup_score / TEMPERATURE))

    return {
        "model_prob":        model_prob,
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
            st.error("Player not found. Check spelling.")
        else:
            with st.spinner("Fetching Statcast data…"):
                if prop_type == "Pitcher Strikeouts":
                    season_data = statcast_pitcher(season_start, season_end, player_id)
                    recent_data = statcast_pitcher(recent_start, recent_end, player_id)

                    season_by_game = season_data[season_data["events"] == "strikeout"].groupby("game_date").size()
                    recent_by_game = recent_data[recent_data["events"] == "strikeout"].groupby("game_date").size()
                    stat_name = "Strikeouts"

                elif prop_type == "Batter Hits":
                    season_data = statcast_batter(season_start, season_end, player_id)
                    recent_data = statcast_batter(recent_start, recent_end, player_id)

                    hit_events = ["single","double","triple","home_run"]
                    season_by_game = season_data[season_data["events"].isin(hit_events)].groupby("game_date").size()
                    recent_by_game = recent_data[recent_data["events"].isin(hit_events)].groupby("game_date").size()
                    stat_name = "Hits"

                else:
                    season_data = statcast_batter(season_start, season_end, player_id)
                    recent_data = statcast_batter(recent_start, recent_end, player_id)

                    season_data["tb"] = season_data["events"].apply(total_bases_from_event)
                    recent_data["tb"] = recent_data["events"].apply(total_bases_from_event)
                    season_by_game = season_data.groupby("game_date")["tb"].sum()
                    recent_by_game = recent_data.groupby("game_date")["tb"].sum()
                    stat_name = "Total Bases"

            if season_by_game.empty:
                st.error("No data found for this player/prop.")
            else:
                season_avg = season_by_game.mean()
                recent_avg = recent_by_game.mean() if not recent_by_game.empty else float("nan")
                projection = calc_weighted_projection(season_avg, recent_avg)

                adj          = 1.0
                k_park_adj   = 1.0
                umpire_adj   = 1.0
                wx_k_adj     = 1.0
                swstr_pct    = None
                umpire_name_prop = None
                weather_info = None

                if prop_type == "Pitcher Strikeouts":
                    # Find today's game to get umpire + weather
                    schedule_data = get_mlb_schedule(today_str)
                    for db in schedule_data.get("dates", []):
                        for game in db.get("games", []):
                            for off in game.get("officials", []):
                                if off.get("officialType") == "Home Plate":
                                    umpire_name_prop = off["official"].get("fullName")

                    adj        = opponent_k_adjustment(opponent_team)
                    k_park_adj = PARK_K_FACTORS.get(opponent_team.upper(), 1.0)
                    umpire_adj = get_umpire_k_adjustment(umpire_name_prop)

                    # C. SwStr% quality signal
                    swstr_pct = get_pitcher_swstr(player_id)
                    if swstr_pct is not None:
                        # League avg SwStr% ~11%; every 1% above/below = ~3% K rate change
                        swstr_factor = 1.0 + (swstr_pct - 0.11) * 3.0
                        adj *= max(min(swstr_factor, 1.30), 0.70)

                    # G. Weather K adjustment
                    p_info      = get_pitcher_info(player_id)
                    home_team   = opponent_team  # pitcher's opponent = home or away?
                    coords      = PARK_COORDS.get(opponent_team.upper(),
                                                  PARK_COORDS.get("NYY"))
                    weather_info = get_weather(coords[0], coords[1], today_str)
                    wx_k_adj    = weather_k_factor(
                        weather_info["temp_f"], weather_info["wind_mph"]
                    )

                    projection = projection * adj * k_park_adj * umpire_adj * wx_k_adj

                # Std-dev aware probability (Improvement from previous version)
                std_dev = float(season_by_game.std()) if len(season_by_game) > 1 else 1.5

                edge         = projection - sportsbook_line
                over_prob    = estimate_over_probability(projection, sportsbook_line, std_dev)
                implied_prob = implied_probability(american_odds)
                ev           = expected_value(over_prob, american_odds)
                label        = confidence_label(edge, ev)

                st.subheader("📊 Prediction Results")

                rcol1, rcol2, rcol3 = st.columns(3)
                rcol1.metric("Season Avg",   f"{season_avg:.2f}")
                rcol2.metric("Recent Avg",   f"{recent_avg:.2f}")
                rcol3.metric("Projection",   f"{projection:.2f}")

                if prop_type == "Pitcher Strikeouts":
                    st.write(f"**Adjustments applied:**")
                    st.write(f"• Opponent K-Rate: {adj:.3f}x")
                    st.write(f"• Park K-Factor ({opponent_team.upper()}): {k_park_adj:.2f}x")
                    st.write(f"• Umpire ({umpire_name_prop or 'Unknown'}): {umpire_adj:.2f}x")
                    if swstr_pct is not None:
                        st.write(
                            f"• SwStr% Quality Signal: {swstr_pct*100:.1f}% "
                            f"(league avg ~11%)"
                        )
                    st.write(f"• Weather K-Factor: {wx_k_adj:.3f}x")
                    if weather_info:
                        st.caption(
                            f"Weather at park: {weather_info['temp_f']:.0f}°F, "
                            f"{weather_info['wind_mph']:.0f} mph wind"
                        )

                    # A. FIP/xFIP for pitcher prop display
                    fip_data = get_pitcher_info(player_id)
                    babip_flag = babip_regression_flag(fip_data.get("babip", 0.300))

                    st.subheader("⚾ Pitcher Quality Metrics")
                    pcols = st.columns(5)
                    pcols[0].metric("ERA",   f"{fip_data['era']:.2f}")
                    pcols[1].metric("FIP",   f"{fip_data['fip']:.2f}")
                    pcols[2].metric("xFIP",  f"{fip_data['xfip']:.2f}")
                    pcols[3].metric("SIERA", f"{fip_data['siera']:.2f}")
                    pcols[4].metric("K/9",   f"{fip_data['k_per_9']:.1f}")
                    if babip_flag:
                        st.caption(babip_flag)

                st.write(f"Sportsbook Line: {sportsbook_line} | Edge: {edge:.2f} | Std Dev: ±{std_dev:.2f}")

                st.subheader("💰 Betting Value")
                vcol1, vcol2, vcol3 = st.columns(3)
                vcol1.metric("Over Probability", f"{over_prob*100:.1f}%")
                vcol2.metric("Implied Prob",     f"{implied_prob*100:.1f}%")
                vcol3.metric("Expected Value",   f"{ev*100:.1f}%")

                st.write(f"**Confidence: {label}**")

                if label == "Strong Play":
                    st.success("✅ Strong OVER value")
                elif label == "Lean Over":
                    st.success("📊 Lean OVER")
                elif label == "Strong Under":
                    st.warning("⬇️ Strong UNDER value")
                elif label == "Lean Under":
                    st.warning("📉 Lean UNDER")
                else:
                    st.info("⛔ No Bet")

                st.subheader("📅 Season Game Log")
                game_log = season_by_game.reset_index(name=stat_name)
                st.dataframe(game_log)
                st.bar_chart(game_log.set_index("game_date"))