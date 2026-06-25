"""
MLB Player Prop + Moneyline Predictor — v3 (Sharper Calibration)
=========================================================
Key changes from v1:
  1. Book-anchored moneyline model: starts from book implied prob,
     only adjusts for factors books systematically misprice
     (FIP vs ERA gap, bullpen recency, rest, weather, recent form divergence)
  2. Moneyline probabilities adjusted in log-odds space instead of raw additive probability
  3. All moneyline factors are shrunk toward zero to reduce false positives
  4. Recent form now uses run-differential divergence instead of recent win rate
  5. Head-to-head and streaks are removed from the model because they are noisy
  6. Bullpen / late-inning / recent pitching stats are regressed toward league average
  7. Moneyline floors tightened: Lean ≥55%/3.5%edge/4%EV, Strong ≥57%/5%edge/6%EV
  8. Factor tracking panel: log each day's picks + outcomes for calibration
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
import json
import os

cache.enable()

# ---------------------------------------------------------------------------
# Backtest toggle: set False to disable heat / player-heat / upset adjustments
# and reproduce the v3 baseline. Used by backtest_compare_heat.py for A/B.
# ---------------------------------------------------------------------------
HEAT_FACTORS_ENABLED = True

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
# Factor tracking (local JSON log for calibration)
# ---------------------------------------------------------------------------

TRACKING_FILE = "mlb_factor_tracking.json"

def load_tracking():
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r") as f:
            return json.load(f)
    return {"picks": []}

def save_tracking(data):
    with open(TRACKING_FILE, "w") as f:
        json.dump(data, f, indent=2)

def log_pick(pick_data):
    data = load_tracking()
    data["picks"].append(pick_data)
    save_tracking(data)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

prop_type = st.selectbox(
    "Choose Bet Type",
    [
        "Today's Slate (Auto)",
        "Pitcher Strikeouts",
        "Batter Hits",
        "Batter Total Bases",
        "Team Moneyline",
    ],
)

if prop_type == "Today's Slate (Auto)":
    st.caption(
        "Runs moneyline + pitcher strikeout predictions on **every** MLB game "
        "scheduled today. Pulls live DraftKings odds when available."
    )
    use_live_odds = st.checkbox(
        "Pull live odds (The Odds API → DraftKings)",
        value=True,
        help="If unchecked or the API is unreachable, moneyline uses -110 and pitcher Ks show projection-only.",
    )
    min_edge_filter = st.slider(
        "Filter summary table: minimum edge over book (%)",
        min_value=0.0, max_value=10.0, value=0.0, step=0.5,
        help="Set to 4% to see only Lean+, 6% for Strong only, 0% to see all games.",
    )
    # placeholders so the moneyline manual-mode variables don't leak into auto mode
    team = opponent_team = ""
    american_odds = -110
    override_starters = False
    team_starter_last = team_starter_first = ""
    opponent_starter_last = opponent_starter_first = ""

elif prop_type == "Team Moneyline":
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

    st.caption("Enter the odds for each side separately.")
    odds_col1, odds_col2 = st.columns(2)
    with odds_col1:
        over_odds  = st.number_input("Over Odds (e.g. -115)", value=-115)
    with odds_col2:
        under_odds = st.number_input("Under Odds (e.g. -105)", value=-105)

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

def clamp(value, low, high):
    return max(low, min(high, value))

def logit(prob):
    prob = clamp(prob, 0.001, 0.999)
    return math.log(prob / (1 - prob))

def inverse_logit(value):
    return 1 / (1 + math.exp(-value))

def regress_to_mean(value, league_avg, weight):
    """Shrink noisy samples toward league average. weight=0.60 means 60% stat, 40% league."""
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return league_avg
        return value * weight + league_avg * (1 - weight)
    except Exception:
        return league_avg

def estimate_over_probability(projection, line, std_dev=None):
    scale = max(std_dev, 0.5) if std_dev is not None else 1.5
    return 1 / (1 + math.exp(-(projection - line) / scale))

def expected_value(model_prob, odds):
    profit = odds / 100 if odds > 0 else 100 / abs(odds)
    return model_prob * profit - (1 - model_prob)

def normalize(value, center, scale):
    return (value - center) / scale

def confidence_label(edge, over_ev, under_ev):
    if   over_ev  > 0.10 and edge >  0.5: return "Strong Over"
    elif over_ev  > 0.03 and edge >  0.2: return "Lean Over"
    elif under_ev > 0.10 and edge < -0.5: return "Strong Under"
    elif under_ev > 0.03 and edge < -0.2: return "Lean Under"
    return "No Bet"

def moneyline_label(ev, model_prob, implied_prob, american_odds):
    """
    v4 — Edge-based labels. An underdog the model likes more than the book
    is still +EV; the previous 55% absolute-prob floor wrongly killed those.

    Strong Bet: edge ≥4%, EV ≥4%
    Lean:       edge ≥2%, EV ≥2%
    No Bet:     non-positive edge, or heavy chalk without enough edge
    """
    edge_pct = round(model_prob - implied_prob, 4)
    ev_r     = round(ev, 4)

    if edge_pct <= 0:
        return "No Bet"

    # Heavy chalk: small edge isn't worth the limited upside
    if american_odds <= -200 and edge_pct < 0.05:
        return "No Bet"

    if edge_pct >= 0.04 and ev_r >= 0.04:
        return "Strong Bet"

    if edge_pct >= 0.02 and ev_r >= 0.02:
        return "Lean"

    return "No Bet"

def get_umpire_k_adjustment(name):
    return UMPIRE_K_RATES.get(name, 1.0) if name else 1.0

# ---------------------------------------------------------------------------
# A. FIP / xFIP for starters
# ---------------------------------------------------------------------------

@st.cache_data
def get_fip_stats(year):
    try:
        return pitching_stats(year, qual=1)
    except Exception:
        return pd.DataFrame()

def get_pitcher_fip(mlbam_id, year):
    defaults = {"fip": 4.20, "xfip": 4.20, "siera": 4.20,
                "babip": 0.300, "k_per_9": 8.0, "era": 4.30,
                "swstr_pct": 0.11, "name": "Unknown", "throws": "R"}
    try:
        df = get_fip_stats(year)
        if df.empty:
            return defaults

        match = df[df["MLBAMID"] == mlbam_id] if "MLBAMID" in df.columns else pd.DataFrame()
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
            "fip":       safe("FIP",    4.20),
            "xfip":      safe("xFIP",   4.20),
            "siera":     safe("SIERA",  4.20),
            "babip":     safe("BABIP",  0.300),
            "k_per_9":   safe("K/9",    8.0),
            "era":       safe("ERA",    4.30),
            "swstr_pct": safe("SwStr%", 0.11),
            "name":      str(row.get("Name", "Unknown")),
            "throws":    defaults["throws"],
        }
    except Exception:
        return defaults


@st.cache_data
def get_pitcher_info(player_id):
    url    = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
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

    fip_data = get_pitcher_fip(int(player_id), today.year)

    if fip_data["fip"] == 4.20 and fip_data["xfip"] == 4.20 and era != 4.30:
        fip_data["fip"]     = round(era * 0.95 + 4.20 * 0.05, 3)
        fip_data["xfip"]    = round(era * 0.88 + 4.20 * 0.12, 3)
        fip_data["siera"]   = round(era * 0.85 + 4.20 * 0.15, 3)
        fip_data["k_per_9"] = round(max(4.0, 12.0 - era * 1.0), 1)

    fip_data["era"]    = era
    fip_data["throws"] = throws
    fip_data["name"]   = name
    return fip_data


# ---------------------------------------------------------------------------
# B. wRC+ for team offense — with capped fallback (v2)
# ---------------------------------------------------------------------------

@st.cache_data
def get_team_wrc_plus(team_abbrev, year):
    try:
        df = batting_stats(year, qual=0, ind=1)
        if df.empty or "Team" not in df.columns or "wRC+" not in df.columns:
            return 100.0

        FG_MAP = {
            "WSH": "WSN", "WAS": "WSN", "CWS": "CHW",
            "SD":  "SDP", "SF":  "SFG", "TB":  "TBR",
            "KC":  "KCR", "OAK": "ATH",
        }
        fg_abbrev = FG_MAP.get(team_abbrev.upper(), team_abbrev.upper())
        team_df   = df[df["Team"].str.upper() == fg_abbrev].copy()

        if team_df.empty:
            team_df = df[df["Team"].str.upper().str.startswith(fg_abbrev[:3], na=False)].copy()
        if team_df.empty:
            return 100.0

        team_df["wRC+"] = pd.to_numeric(team_df["wRC+"], errors="coerce")
        team_df["PA"]   = pd.to_numeric(team_df["PA"],   errors="coerce") if "PA" in team_df.columns else 1.0
        team_df = team_df.dropna(subset=["wRC+"])
        if team_df.empty:
            return 100.0

        total_pa = team_df["PA"].sum()
        if total_pa > 0:
            return round(float((team_df["wRC+"] * team_df["PA"]).sum() / total_pa), 1)
        return round(float(team_df["wRC+"].mean()), 1)
    except Exception:
        return 100.0


# ---------------------------------------------------------------------------
# C. SwStr% — tightened thresholds (v2)
# ---------------------------------------------------------------------------

@st.cache_data
def get_pitcher_swstr(player_id):
    try:
        data = statcast_pitcher(recent_start, recent_end, player_id)
        if data.empty:
            return None

        total_pitches    = len(data)
        swinging_strikes = data[
            data["description"].isin([
                "swinging_strike", "swinging_strike_blocked", "foul_tip",
            ])
        ].shape[0]

        # v2: raised minimum from 50 to 75 pitches for reliability
        if total_pitches < 75:
            return None

        return round(swinging_strikes / total_pitches, 4)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# D. Injury / IL roster check
# ---------------------------------------------------------------------------

@st.cache_data
def get_injured_players(team_abbrev):
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
    try:
        data = get_team_schedule(today.year, team_abbrev)
        if data.empty or "Date" not in data.columns:
            return 1

        dates = pd.to_datetime(
            data["Date"].astype(str).str.extract(r"(\w+ \d+)")[0]
            + f" {today.year}",
            format="%b %d %Y",
            errors="coerce",
        ).dropna()

        if dates.empty:
            return 1

        last_game = dates.max()
        return max(int((today - last_game).days), 0)
    except Exception:
        return 1


def get_streak(team_abbrev):
    try:
        data    = get_team_schedule(today.year, team_abbrev)
        if data.empty:
            return 0
        results = data["win"].tolist()
        if not results:
            return 0
        last   = results[-1]
        streak = 0
        for r in reversed(results):
            if r == last:
                streak += 1
            else:
                break
        return streak if last == 1 else -streak
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# F. BABIP regression flag
# ---------------------------------------------------------------------------

def babip_regression_flag(babip):
    if babip < 0.260:
        return f"⚠️ Low BABIP ({babip:.3f}) — ERA likely better than true talent, expect regression"
    elif babip > 0.340:
        return f"✅ High BABIP ({babip:.3f}) — ERA likely worse than true talent, improvement likely"
    return None


# ---------------------------------------------------------------------------
# G. Weather via Open-Meteo
# ---------------------------------------------------------------------------

@st.cache_data
def get_weather(lat, lon, date_str):
    try:
        url    = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":           lat,
            "longitude":          lon,
            "hourly":             "temperature_2m,windspeed_10m,winddirection_10m",
            "temperature_unit":   "fahrenheit",
            "windspeed_unit":     "mph",
            "timezone":           "auto",
            "start_date":         date_str,
            "end_date":           date_str,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data   = resp.json()
        hourly = data.get("hourly", {})
        temps  = hourly.get("temperature_2m", [])
        winds  = hourly.get("windspeed_10m", [])
        dirs   = hourly.get("winddirection_10m", [])
        idx    = 19 if len(temps) > 19 else len(temps) // 2

        return {
            "temp_f":       round(temps[idx], 1) if temps else 70.0,
            "wind_mph":     round(winds[idx], 1) if winds else 5.0,
            "wind_dir_deg": round(dirs[idx],  1) if dirs  else 0.0,
        }
    except Exception:
        return {"temp_f": 70.0, "wind_mph": 5.0, "wind_dir_deg": 0.0}


def weather_run_factor(temp_f, wind_mph, wind_dir_deg=None):
    temp_factor = 1.0 + (temp_f - 72) * 0.004
    wind_factor = 1.0
    if wind_mph > 5:
        if wind_dir_deg is not None:
            if   45  <= wind_dir_deg <= 135: wind_factor = 1.0 + wind_mph * 0.005
            elif 225 <= wind_dir_deg <= 315: wind_factor = 1.0 - wind_mph * 0.004
            else:                            wind_factor = 1.0 + wind_mph * 0.001
        else:
            wind_factor = 1.0 + wind_mph * 0.002
    return round(temp_factor * wind_factor, 4)


def weather_k_factor(temp_f, wind_mph):
    return round(max((1.0 - (temp_f - 72) * 0.001) * (1.0 - wind_mph * 0.001), 0.90), 4)


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
                sel_p        = game["teams"]["home"].get("probablePitcher")
                opp_p        = game["teams"]["away"].get("probablePitcher")
                home_ab      = team
            else:
                sel_p        = game["teams"]["away"].get("probablePitcher")
                opp_p        = game["teams"]["home"].get("probablePitcher")
                home_ab      = opponent_team

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
        resp   = requests.get(url, params=params, timeout=20)
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
        data       = data.sort_values("game_date")
        game_dates = data["game_date"].unique()
        if len(game_dates) < 2:
            return None
        recent     = data[data["game_date"].isin(game_dates[-num_starts:])]
        bf         = recent[recent["events"].notna()].shape[0]
        ip_est     = bf / 3.0
        if ip_est < 1:
            return None
        hrs = (recent["events"] == "home_run").sum()
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

    if "batting_team" not in data.columns:
        is_top = data["inning_topbot"] == "Top"
        data["batting_team"] = data["away_team"].where(is_top, data["home_team"])
    mask = (
        (data["batting_team"] == team_abbrev)
        & (data["p_throws"] == pitcher_hand)
        & (data["events"].notna())
    )
    pa = data[mask].copy()
    if pa.empty:
        return 0.700

    for col in ["single", "double", "triple", "home_run", "walk", "hbp"]:
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

    if "fielding_team" not in late.columns:
        is_top = late["inning_topbot"] == "Top"
        late["fielding_team"] = late["home_team"].where(is_top, late["away_team"])
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
        return {"wins": 0, "losses": 0, "games": 0, "win_rate": 0.5,
                "runs_scored": 4.3, "runs_allowed": 4.3, "run_diff": 0}
    wins  = int(data["win"].sum())
    games = len(data)
    return {
        "wins":         wins,
        "losses":       games - wins,
        "games":        games,
        "win_rate":     wins / games,
        "runs_scored":  data["R"].mean(),
        "runs_allowed": data["RA"].mean(),
        "run_diff":     data["R"].mean() - data["RA"].mean(),
    }


# ---------------------------------------------------------------------------
# Team / player heat ("fire streak" detection for upset picks)
# ---------------------------------------------------------------------------

def team_record_last_n(team_abbrev, n=10):
    """Wins, run-diff, and quality-weighted streak over the last n games."""
    data = get_team_schedule(today.year, team_abbrev)
    if data.empty:
        return {"wins": 0, "games": 0, "win_rate": 0.5, "run_diff": 0.0,
                "quality_score": 0.0}
    tail = data.tail(n).copy()
    games = len(tail)
    if games == 0:
        return {"wins": 0, "games": 0, "win_rate": 0.5, "run_diff": 0.0,
                "quality_score": 0.0}
    wins = int(tail["win"].sum())
    run_diff = (tail["R"].mean() - tail["RA"].mean())

    # Quality-weighted score: each game weighted by margin (capped),
    # so a 10-run blowout matters more than a walk-off squeaker.
    margins = (tail["R"] - tail["RA"]).clip(lower=-8, upper=8)
    quality_score = float(margins.mean()) if not margins.empty else 0.0

    return {
        "wins":          wins,
        "games":         games,
        "win_rate":      wins / games,
        "run_diff":      float(run_diff),
        "quality_score": quality_score,
    }


@st.cache_data
def team_offense_heat(team_abbrev):
    """
    Recent offense vs season offense (last 30d window already cached).
    Positive = team is hitting better than its season baseline. Range ~[-1, +1].
    """
    try:
        season = get_statcast_data(season_start, season_end).copy()
        recent = get_statcast_data(recent_start, recent_end).copy()
    except Exception:
        return 0.0
    team = team_abbrev.upper()

    def woba_for(df):
        if df.empty:
            return None
        df = df.copy()
        if "batting_team" not in df.columns:
            is_top = df["inning_topbot"] == "Top"
            df["batting_team"] = df["away_team"].where(is_top, df["home_team"])
        tdf = df[(df["batting_team"] == team) & (df["events"].notna())]
        if tdf.empty:
            return None
        # Simple wOBA proxy using event weights
        weights = {"walk": 0.69, "hit_by_pitch": 0.72, "single": 0.89,
                   "double": 1.27, "triple": 1.62, "home_run": 2.10}
        num = sum(weights[k] * (tdf["events"] == k).sum() for k in weights)
        ab = len(tdf[~tdf["events"].isin(
            ["walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"]
        )])
        bb = (tdf["events"] == "walk").sum()
        hbp = (tdf["events"] == "hit_by_pitch").sum()
        sf = (tdf["events"] == "sac_fly").sum()
        denom = ab + bb + hbp + sf
        return num / denom if denom > 0 else None

    s_woba = woba_for(season)
    r_woba = woba_for(recent)
    if s_woba is None or r_woba is None or s_woba <= 0:
        return 0.0
    # Normalize the delta — a 50-point wOBA swing is a big hot streak
    delta = (r_woba - s_woba) / 0.050
    return float(max(-1.5, min(1.5, delta)))


@st.cache_data
def team_player_heat(team_abbrev):
    """
    Aggregate "player streak" boost for a team. Finds batters who have
    significantly outperformed their season baseline over the last 30 days,
    weights them by recent plate appearances (so star regulars matter most).
    Returns dict with score (~[-1, +1]) and list of top hot/cold names.
    """
    try:
        season = get_statcast_data(season_start, season_end)
        recent = get_statcast_data(recent_start, recent_end)
    except Exception:
        return {"score": 0.0, "hot": [], "cold": []}
    team = team_abbrev.upper()

    def player_woba_table(df):
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        if "batting_team" not in df.columns:
            is_top = df["inning_topbot"] == "Top"
            df["batting_team"] = df["away_team"].where(is_top, df["home_team"])
        tdf = df[(df["batting_team"] == team) & (df["events"].notna())].copy()
        if tdf.empty:
            return pd.DataFrame()
        weights = {"walk": 0.69, "hit_by_pitch": 0.72, "single": 0.89,
                   "double": 1.27, "triple": 1.62, "home_run": 2.10}
        for k in weights:
            tdf[k] = (tdf["events"] == k).astype(int)
        for k in ["sac_fly", "sac_bunt", "catcher_interf"]:
            tdf[k] = (tdf["events"] == k).astype(int)
        tdf["pa"] = 1
        tdf["ab_flag"] = (~tdf["events"].isin(
            ["walk", "hit_by_pitch", "sac_fly", "sac_bunt", "catcher_interf"]
        )).astype(int)
        agg = tdf.groupby("batter").agg(
            single=("single", "sum"), double=("double", "sum"),
            triple=("triple", "sum"), hr=("home_run", "sum"),
            bb=("walk", "sum"), hbp=("hit_by_pitch", "sum"),
            sf=("sac_fly", "sum"), pa=("pa", "sum"), ab=("ab_flag", "sum"),
            name=("player_name", "first") if "player_name" in tdf.columns else ("batter", "first"),
        )
        num = (weights["walk"]*agg["bb"] + weights["hit_by_pitch"]*agg["hbp"]
               + weights["single"]*agg["single"] + weights["double"]*agg["double"]
               + weights["triple"]*agg["triple"] + weights["home_run"]*agg["hr"])
        denom = (agg["ab"] + agg["bb"] + agg["hbp"] + agg["sf"]).replace(0, 1)
        agg["woba"] = num / denom
        return agg

    season_t = player_woba_table(season)
    recent_t = player_woba_table(recent)
    if season_t.empty or recent_t.empty:
        return {"score": 0.0, "hot": [], "cold": []}

    # Keep batters with at least 30 season PA (regulars) and 15 recent PA
    season_t = season_t[season_t["pa"] >= 30]
    recent_t = recent_t[recent_t["pa"] >= 15]
    joined = recent_t.join(season_t, lsuffix="_r", rsuffix="_s", how="inner")
    if joined.empty:
        return {"score": 0.0, "hot": [], "cold": []}

    joined["delta"] = joined["woba_r"] - joined["woba_s"]
    # PA-weighted average delta, normalized so 0.050 wOBA = +1
    total_pa = joined["pa_r"].sum()
    if total_pa <= 0:
        return {"score": 0.0, "hot": [], "cold": []}
    weighted_delta = (joined["delta"] * joined["pa_r"]).sum() / total_pa
    score = float(max(-1.5, min(1.5, weighted_delta / 0.050)))

    name_col = "name_r" if "name_r" in joined.columns else None
    hot = []
    cold = []
    top = joined.sort_values("delta", ascending=False)
    bot = joined.sort_values("delta", ascending=True)
    for _, row in top.head(3).iterrows():
        if row["delta"] <= 0:
            break
        nm = row[name_col] if name_col else str(row.name)
        hot.append((nm, float(row["delta"])))
    for _, row in bot.head(2).iterrows():
        if row["delta"] >= 0:
            break
        nm = row[name_col] if name_col else str(row.name)
        cold.append((nm, float(row["delta"])))

    return {"score": score, "hot": hot, "cold": cold}


def compute_team_heat(team_abbrev):
    """
    Composite team "fire streak" score, normalized to roughly [-1.5, +1.5].
    Combines: last-10 win rate vs .500, recent run-diff momentum,
    quality-weighted streak, and team offense heat. NOT player heat —
    that is added separately so we can surface it in the UI.
    """
    season = team_record_stats(team_abbrev, recent=False)
    last10 = team_record_last_n(team_abbrev, n=10)
    streak = get_streak(team_abbrev)

    wr_div = (last10["win_rate"] - season["win_rate"]) * 2.0     # ~[-0.5, +0.5]
    streak_norm = (float(streak) * 0.5 + last10["quality_score"] * 0.5) / 5.0
    rd_div = (last10["run_diff"] - season["run_diff"]) / 2.0     # ~[-1, +1]
    off_heat = team_offense_heat(team_abbrev)                    # ~[-1.5, +1.5]

    raw = wr_div + streak_norm * 0.6 + rd_div * 0.7 + off_heat * 0.6
    score = float(max(-1.5, min(1.5, raw)))

    return {
        "score":           score,
        "last10_record":   f"{last10['wins']}-{last10['games'] - last10['wins']}",
        "last10_run_diff": last10["run_diff"],
        "quality_streak":  streak_norm,
        "offense_heat":    off_heat,
        "current_streak":  streak,
    }


@st.cache_data
def get_season_k_rate(opponent_team):
    """
    Season-long K rate for a team (relative to league average = 1.0).
    Used to blend with recent K rate for a more stable opponent adjustment.
    """
    try:
        data = get_statcast_data(season_start, season_end)
        if "batting_team" not in data.columns:
            is_top = data["inning_topbot"] == "Top"
            data = data.assign(
                batting_team=data["away_team"].where(is_top, data["home_team"])
            )
        opp    = data[data["batting_team"] == opponent_team.upper()]
        tot_pa = opp[opp["events"].notna()]
        opp_ks = opp[opp["events"] == "strikeout"]
        lg_pa  = data[data["events"].notna()]
        lg_ks  = data[data["events"] == "strikeout"]
        if len(tot_pa) == 0 or len(lg_pa) == 0:
            return 1.0
        return (len(opp_ks) / len(tot_pa)) / (len(lg_ks) / len(lg_pa))
    except Exception:
        return 1.0


@st.cache_data
def opponent_k_adjustment_recent(opponent_team):
    """Recent (30-day) K rate relative to league average."""
    data = get_statcast_data(recent_start, recent_end)
    if "batting_team" not in data.columns:
        is_top = data["inning_topbot"] == "Top"
        data["batting_team"] = data["away_team"].where(is_top, data["home_team"])
    opp    = data[data["batting_team"] == opponent_team.upper()]
    tot_pa = opp[opp["events"].notna()]
    opp_ks = opp[opp["events"] == "strikeout"]
    lg_pa  = data[data["events"].notna()]
    lg_ks  = data[data["events"] == "strikeout"]
    if len(tot_pa) == 0 or len(lg_pa) == 0:
        return 1.0
    return (len(opp_ks) / len(tot_pa)) / (len(lg_ks) / len(lg_pa))


def opponent_k_adjustment(opponent_team):
    """
    v2: Blend 70% season / 30% recent K rate, capped at ±12%.
    Previously was 100% recent (noisy over 30 days).
    """
    season_adj = get_season_k_rate(opponent_team)
    recent_adj = opponent_k_adjustment_recent(opponent_team)
    blended    = season_adj * 0.70 + recent_adj * 0.30
    return max(0.88, min(1.12, blended))


# ---------------------------------------------------------------------------
# The Odds API integration
# ---------------------------------------------------------------------------
#
# Pulls moneylines and pitcher-strikeout props from The Odds API
# (https://the-odds-api.com). We anchor on DraftKings as the bookmaker so
# downstream EV/edge math stays consistent with the previous DK scrape.
#
#   - h2h moneylines for every MLB game today: ONE call
#   - pitcher_strikeouts props: per-event call (Odds API requires this)
#
# Free tier is ~500 req/month, so cache aggressively (5-min TTL).
# Each request decrements `x-requests-remaining` in the response headers.
# ---------------------------------------------------------------------------

ODDS_API_KEY     = "106ca720e4697cbd824d8919edc12713"
ODDS_API_BASE    = "https://api.the-odds-api.com/v4"
ODDS_API_SPORT   = "baseball_mlb"
ODDS_API_BOOK    = "draftkings"   # anchor book — keep parity with previous scrape
ODDS_API_REGIONS = "us"

# Full team name → our internal abbreviation. Names follow the MLB API /
# ESPN convention that The Odds API uses for team labels.
ODDS_TEAM_TO_ABBREV = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves":      "ATL",
    "Baltimore Orioles":    "BAL", "Boston Red Sox":      "BOS",
    "Chicago Cubs":         "CHC", "Chicago White Sox":   "CWS",
    "Cincinnati Reds":      "CIN", "Cleveland Guardians": "CLE",
    "Colorado Rockies":     "COL", "Detroit Tigers":      "DET",
    "Houston Astros":       "HOU", "Kansas City Royals":  "KC",
    "Los Angeles Angels":   "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins":        "MIA", "Milwaukee Brewers":   "MIL",
    "Minnesota Twins":      "MIN", "New York Mets":       "NYM",
    "New York Yankees":     "NYY", "Athletics":           "OAK",
    "Oakland Athletics":    "OAK", "Philadelphia Phillies":"PHI",
    "Pittsburgh Pirates":   "PIT", "San Diego Padres":    "SD",
    "Seattle Mariners":     "SEA", "San Francisco Giants":"SF",
    "St. Louis Cardinals":  "STL", "Tampa Bay Rays":      "TB",
    "Texas Rangers":        "TEX", "Toronto Blue Jays":   "TOR",
    "Washington Nationals": "WSH",
}


@st.cache_data(ttl=300)
def fetch_odds_api_moneylines():
    """
    Single call: every MLB event with h2h moneylines from DraftKings.
    Returns the raw event list (or [] on failure).
    """
    url = f"{ODDS_API_BASE}/sports/{ODDS_API_SPORT}/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    ODDS_API_REGIONS,
        "markets":    "h2h",
        "bookmakers": ODDS_API_BOOK,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


@st.cache_data(ttl=300)
def fetch_odds_api_pitcher_ks_for_event(event_id):
    """
    Per-event call for pitcher_strikeouts props. The Odds API exposes player
    props only at the event endpoint, so we make one request per game.
    """
    url = f"{ODDS_API_BASE}/sports/{ODDS_API_SPORT}/events/{event_id}/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    ODDS_API_REGIONS,
        "markets":    "pitcher_strikeouts",
        "bookmakers": ODDS_API_BOOK,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def parse_odds_api_moneylines(events):
    """
    Returns: {event_id: {
        'home_abbrev', 'away_abbrev', 'home_name', 'away_name',
        'home_ml', 'away_ml', 'start'
    }}
    Missing entries (no posted line yet) get None for odds.
    """
    out = {}
    for ev in events or []:
        event_id  = ev.get("id")
        home_name = ev.get("home_team")
        away_name = ev.get("away_team")
        if not event_id or not home_name or not away_name:
            continue
        home_abbrev = ODDS_TEAM_TO_ABBREV.get(home_name)
        away_abbrev = ODDS_TEAM_TO_ABBREV.get(away_name)
        if not home_abbrev or not away_abbrev:
            continue
        entry = {
            "home_abbrev": home_abbrev,
            "away_abbrev": away_abbrev,
            "home_name":   home_name,
            "away_name":   away_name,
            "start":       ev.get("commence_time"),
            "home_ml":     None,
            "away_ml":     None,
        }
        for bm in ev.get("bookmakers", []):
            if bm.get("key") != ODDS_API_BOOK:
                continue
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    try:
                        price = int(outcome.get("price"))
                    except (TypeError, ValueError):
                        continue
                    if outcome.get("name") == home_name:
                        entry["home_ml"] = price
                    elif outcome.get("name") == away_name:
                        entry["away_ml"] = price
        out[event_id] = entry
    return out


def parse_odds_api_pitcher_ks(event_data):
    """
    Returns: {pitcher_name_lower: {'line', 'over_odds', 'under_odds'}}
    The Odds API encodes pitcher_strikeouts as one outcome per (pitcher, side)
    with `description` = pitcher name and `name` = "Over" / "Under".
    """
    out = {}
    if not event_data:
        return out
    for bm in event_data.get("bookmakers", []):
        if bm.get("key") != ODDS_API_BOOK:
            continue
        for market in bm.get("markets", []):
            if market.get("key") != "pitcher_strikeouts":
                continue
            for outcome in market.get("outcomes", []):
                pitcher = outcome.get("description") or outcome.get("participant")
                side    = (outcome.get("name") or "").lower()
                if not pitcher or side not in ("over", "under"):
                    continue
                key = pitcher.lower().strip()
                try:
                    line  = float(outcome.get("point"))
                    price = int(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                entry = out.setdefault(key, {"line": line, "over_odds": -115, "under_odds": -115})
                entry["line"] = line
                if side == "over":
                    entry["over_odds"]  = price
                else:
                    entry["under_odds"] = price
    return out


def fetch_all_pitcher_ks(moneylines):
    """
    Loop the events from `moneylines` and fetch pitcher_strikeouts per event.
    Merges results into a single {pitcher_name_lower: {...}} dict.
    """
    combined = {}
    for event_id in moneylines.keys():
        event_data = fetch_odds_api_pitcher_ks_for_event(event_id)
        combined.update(parse_odds_api_pitcher_ks(event_data))
    return combined


def match_moneyline(lines, home_abbrev, away_abbrev):
    """Find moneyline entry matching this matchup; returns (home_ml, away_ml) or (None, None)."""
    for ml in lines.values():
        if ml["home_abbrev"] == home_abbrev and ml["away_abbrev"] == away_abbrev:
            return ml["home_ml"], ml["away_ml"]
    return None, None


def match_pitcher_k(props, pitcher_name):
    """Look up strikeout line for a pitcher by name. Returns dict or None."""
    if not pitcher_name:
        return None
    key = pitcher_name.lower().strip()
    if key in props:
        return props[key]
    # Last-name fallback (sometimes the book uses a shortened form)
    last = key.split()[-1] if " " in key else key
    for k, v in props.items():
        if k.split()[-1] == last:
            return v
    return None


# ---------------------------------------------------------------------------
# Pitcher quality score
# ---------------------------------------------------------------------------

def pitcher_quality_score(info):
    era   = info.get("era",   4.30)
    fip   = info.get("fip",   4.20)
    xfip  = info.get("xfip",  4.20)
    siera = info.get("siera", 4.20)
    return era * 0.15 + fip * 0.25 + xfip * 0.30 + siera * 0.30


# ---------------------------------------------------------------------------
# Pitcher K projection — shared helper for single-player and auto modes
# ---------------------------------------------------------------------------

def compute_pitcher_k_projection(player_id, opponent_team, home_park_abbrev=None, umpire_name=None):
    """
    Return a dict with the pitcher's projected strikeouts plus all the
    component adjustments and metadata used in the breakdown UI.

    Caller supplies the matchup context (opponent, where the game is
    played, umpire) — we don't look up the schedule here so this function
    is reusable for both the manual single-player flow and the auto slate.

    On insufficient data returns {'error': <str>}.
    """
    MIN_GAMES_SEASON = 5
    MIN_GAMES_RECENT = 3

    if home_park_abbrev is None:
        home_park_abbrev = opponent_team

    try:
        season_data = statcast_pitcher(season_start, season_end, player_id)
        recent_data = statcast_pitcher(recent_start, recent_end, player_id)
    except Exception as e:
        return {"error": f"Statcast fetch failed: {e}"}

    if season_data is None or season_data.empty:
        return {"error": "No Statcast data this season."}

    season_by_game = season_data[season_data["events"] == "strikeout"].groupby("game_date").size()
    season_starts  = season_data["game_date"].nunique()

    if season_starts < MIN_GAMES_SEASON:
        return {"error": f"Only {season_starts} games this season (need {MIN_GAMES_SEASON})."}

    if recent_data is not None and not recent_data.empty:
        recent_by_game = recent_data[recent_data["events"] == "strikeout"].groupby("game_date").size()
        recent_dates   = recent_data["game_date"].unique()
        recent_starts  = len(recent_dates)
        recent_filled  = recent_by_game.reindex(recent_dates, fill_value=0)
        recent_avg     = recent_filled.mean() if recent_starts >= MIN_GAMES_RECENT else float("nan")
    else:
        recent_starts = 0
        recent_avg    = float("nan")

    season_avg = season_by_game.mean()
    projection = calc_weighted_projection(season_avg, recent_avg)

    # Adjustments
    adj_opp_k   = opponent_k_adjustment(opponent_team)
    adj_park    = PARK_K_FACTORS.get(home_park_abbrev.upper(), 1.0)
    adj_umpire  = get_umpire_k_adjustment(umpire_name)

    swstr_pct   = get_pitcher_swstr(player_id)
    if swstr_pct is not None:
        swstr_factor = 1.0 + (swstr_pct - 0.11) * 2.0
        adj_swstr    = max(min(swstr_factor, 1.15), 0.85)
    else:
        adj_swstr    = 1.0

    adj_opp_k_with_swstr = adj_opp_k * adj_swstr

    fip_data = get_pitcher_info(player_id)

    coords      = PARK_COORDS.get(home_park_abbrev.upper(), PARK_COORDS.get("NYY"))
    weather_info = get_weather(coords[0], coords[1], today_str)
    adj_weather  = weather_k_factor(weather_info["temp_f"], weather_info["wind_mph"])

    projection = projection * adj_opp_k_with_swstr * adj_park * adj_umpire * adj_weather

    std_dev = float(season_by_game.std()) if len(season_by_game) > 1 else 1.5
    std_dev = max(std_dev, 0.8)

    return {
        "projection":      projection,
        "season_avg":      season_avg,
        "recent_avg":      recent_avg,
        "season_starts":   season_starts,
        "recent_starts":   recent_starts,
        "std_dev":         std_dev,
        "adj_opp_k":       adj_opp_k,
        "adj_park":        adj_park,
        "adj_umpire":      adj_umpire,
        "adj_swstr":       adj_swstr,
        "adj_weather":     adj_weather,
        "swstr_pct":       swstr_pct,
        "fip_data":        fip_data,
        "weather_info":    weather_info,
        "umpire_name":     umpire_name,
        "season_by_game":  season_by_game,
        "season_data":     season_data,
    }


# ---------------------------------------------------------------------------
# Moneyline model — v2: BOOK-ANCHORED
# ---------------------------------------------------------------------------
#
# Core philosophy change:
#   v1 built a score from scratch and converted via sigmoid.
#       Problem: the model's absolute win probabilities competed with the
#       book's, and the book is better calibrated on overall team quality.
#
#   v2 starts from the book's implied probability as a prior, then applies
#       ADJUSTMENTS only for factors that books systematically misprice:
#         1. FIP vs ERA gap (books use ERA; smart money uses FIP)
#         2. Bullpen ERA differential (books lag on bullpen recency)
#         3. Rest/fatigue (known after line is set)
#         4. Weather (known after line is set)
#         5. Recent form divergence (current trend vs season baseline)
#         6. wRC+ offense edge (park-adjusted, books sometimes lag)
#         7. Late-inning bullpen (books underweight 7-9 inning splits)
#
#   The adjustments are intentionally small (each ≤2% probability shift)
#   because the goal is to find REAL edge, not override the book.
# ---------------------------------------------------------------------------

def team_moneyline_probability(
    team, opponent_team,
    team_starter_first="", team_starter_last="",
    opponent_starter_first="", opponent_starter_last="",
    american_odds_input=-110,
):
    team          = team.upper()
    opponent_team = opponent_team.upper()

    # ---- Book prior ----
    book_implied = implied_probability(american_odds_input)

    # ---- Matchup / pitchers ----
    matchup         = find_today_matchup(team, opponent_team)
    default_pitcher = {"name":"Unknown","era":4.30,"fip":4.20,"xfip":4.20,
                       "siera":4.20,"babip":0.300,"k_per_9":8.0,
                       "swstr_pct":0.11,"throws":"R"}
    sel_p_info      = dict(default_pitcher)
    opp_p_info      = dict(default_pitcher)
    home_field_edge = 0
    umpire_name     = None
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

    # ---- Records ----
    team_season = team_record_stats(team,          recent=False)
    opp_season  = team_record_stats(opponent_team, recent=False)
    team_recent = team_record_stats(team,          recent=True)
    opp_recent  = team_record_stats(opponent_team, recent=True)

    # ---- Recent ERA for starters ----
    sel_recent_era = None
    opp_recent_era = None
    if matchup and matchup.get("selected_pitcher"):
        sel_recent_era = get_pitcher_recent_era(matchup["selected_pitcher"]["id"])
    if matchup and matchup.get("opponent_pitcher"):
        opp_recent_era = get_pitcher_recent_era(matchup["opponent_pitcher"]["id"])

    def blend_era(season, recent):
        # Recent starter ERA is noisy, so use it lightly.
        return season if recent is None else season * 0.75 + recent * 0.25

    sel_era_bl = blend_era(sel_p_info["era"], sel_recent_era)
    opp_era_bl = blend_era(opp_p_info["era"], opp_recent_era)

    sel_quality = pitcher_quality_score({**sel_p_info, "era": sel_era_bl})
    opp_quality = pitcher_quality_score({**opp_p_info, "era": opp_era_bl})

    # ---- wRC+ — capped fallback (v2) ----
    team_wrc = get_team_wrc_plus(team,          today.year)
    opp_wrc  = get_team_wrc_plus(opponent_team, today.year)

    if team_wrc == 100.0 and opp_wrc == 100.0:
        # Cap fallback to ±20 pts from 100 to avoid noisy extreme estimates
        team_wrc = round(min(max(100 + (team_season["runs_scored"] - 4.5) * 16, 80), 120), 1)
        opp_wrc  = round(min(max(100 + (opp_season["runs_scored"]  - 4.5) * 16, 80), 120), 1)

    # ---- Bullpen, regressed toward league average ----
    raw_team_bp_era = get_bullpen_era(team)
    raw_opp_bp_era  = get_bullpen_era(opponent_team)
    team_bp_era = regress_to_mean(raw_team_bp_era, 4.10, 0.60)
    opp_bp_era  = regress_to_mean(raw_opp_bp_era,  4.10, 0.60)
    bullpen_edge = opp_bp_era - team_bp_era  # positive = team advantage

    # ---- Late inning, regressed toward neutral ----
    raw_team_late = late_inning_runs_allowed(recent_start, recent_end, team)
    raw_opp_late  = late_inning_runs_allowed(recent_start, recent_end, opponent_team)
    team_late = regress_to_mean(raw_team_late, 1.50, 0.60)
    opp_late  = regress_to_mean(raw_opp_late,  1.50, 0.60)
    late_inning_edge = opp_late - team_late

    # ---- OPS vs hand ----
    team_ops = calculate_team_ops_vs_hand(team,          opp_p_info["throws"])
    opp_ops  = calculate_team_ops_vs_hand(opponent_team, sel_p_info["throws"])

    # ---- H2H / streaks intentionally removed from model ----
    # They are mostly noise in baseball and are usually already reflected in public pricing.
    h2h_w, h2h_l, h2h_rate = 0, 0, 0.50
    team_streak, opp_streak = 0, 0

    # ---- Rest / fatigue ----
    team_rest   = get_rest_days(team)
    opp_rest    = get_rest_days(opponent_team)

    # ---- Weather ----
    coords     = PARK_COORDS.get(home_team_abbrev.upper(), (39.0, -95.0))
    weather    = get_weather(coords[0], coords[1], today_str)
    wx_run_adj = weather_run_factor(
        weather["temp_f"], weather["wind_mph"], weather["wind_dir_deg"]
    )

    # ---- Injuries ----
    team_injured = get_injured_players(team)
    opp_injured  = get_injured_players(opponent_team)

    # ---- Park factor ----
    pf = PARK_FACTORS.get(home_team_abbrev.upper(), 1.00)

    # -----------------------------------------------------------------------
    # BOOK-ANCHORED ADJUSTMENTS (v2)
    #
    # Start from book_implied and apply small adjustments for factors
    # that books are known to systematically misprice.
    #
    # Each adjustment is scaled so a "1-sigma" edge = ~1-2% probability shift.
    # This keeps the model from dramatically overriding the book.
    # -----------------------------------------------------------------------

    adjustments = {}

    # v3 note:
    # These are LOGIT adjustments, not direct probability points.
    # A 0.10 logit shift usually moves a 50/50 line by about 2.5 percentage points.

    # 1. FIP vs ERA gap, shrunken
    sel_fip_era_gap = sel_p_info["era"] - sel_p_info["fip"]
    opp_fip_era_gap = opp_p_info["era"] - opp_p_info["fip"]
    fip_adj = (sel_fip_era_gap - opp_fip_era_gap) * 0.030
    fip_adj = clamp(fip_adj, -0.10, 0.10)
    adjustments["fip_vs_era_gap"] = round(fip_adj, 4)

    # 2. Bullpen ERA differential, regressed above
    bp_adj = bullpen_edge * 0.035
    bp_adj = clamp(bp_adj, -0.07, 0.07)
    adjustments["bullpen_era_regressed"] = round(bp_adj, 4)

    # 3. Rest / fatigue, small because books mostly price this in
    rest_diff = team_rest - opp_rest
    rest_adj  = rest_diff * 0.025
    rest_adj  = clamp(rest_adj, -0.04, 0.04)
    adjustments["rest_days"] = round(rest_adj, 4)

    # 4. Weather, only matters when one offense is clearly stronger
    offense_edge = team_season["run_diff"] - opp_season["run_diff"]
    wx_dir       = 1 if offense_edge >= 0 else -1
    wx_adj       = wx_dir * (wx_run_adj - 1.0) * 1.25
    wx_adj       = clamp(wx_adj, -0.04, 0.04)
    adjustments["weather"] = round(wx_adj, 4)

    # 5. Recent form now uses run-differential divergence, not win-rate noise
    team_rd_delta = team_recent["run_diff"] - team_season["run_diff"]
    opp_rd_delta  = opp_recent["run_diff"]  - opp_season["run_diff"]
    form_adj = (team_rd_delta - opp_rd_delta) * 0.035
    form_adj = clamp(form_adj, -0.06, 0.06)
    adjustments["recent_run_diff_divergence"] = round(form_adj, 4)

    # 5b/5c. Heat factors (gated by HEAT_FACTORS_ENABLED for A/B backtesting).
    if HEAT_FACTORS_ENABLED:
        # Team "fire streak" composite — last-10 vs season + quality streak +
        # offense heat (wOBA-based). Lets the model see hot underdogs.
        team_heat_data = compute_team_heat(team)
        opp_heat_data  = compute_team_heat(opponent_team)
        heat_diff      = team_heat_data["score"] - opp_heat_data["score"]
        heat_adj       = heat_diff * 0.060
        heat_adj       = clamp(heat_adj, -0.12, 0.12)
        adjustments["team_heat"] = round(heat_adj, 4)

        # Player heat — top hitters who have been on a tear (e.g. PCA for CHC).
        team_player_data = team_player_heat(team)
        opp_player_data  = team_player_heat(opponent_team)
        player_diff      = team_player_data["score"] - opp_player_data["score"]
        player_adj       = player_diff * 0.045
        player_adj       = clamp(player_adj, -0.08, 0.08)
        adjustments["player_heat"] = round(player_adj, 4)
    else:
        # v3 baseline: no heat signals.
        team_heat_data = {"score": 0.0, "last10_record": "—",
                          "last10_run_diff": 0.0, "quality_streak": 0.0,
                          "offense_heat": 0.0, "current_streak": 0}
        opp_heat_data  = dict(team_heat_data)
        team_player_data = {"score": 0.0, "hot": [], "cold": []}
        opp_player_data  = {"score": 0.0, "hot": [], "cold": []}
        heat_diff = 0.0
        player_diff = 0.0
        heat_adj = 0.0
        player_adj = 0.0

    # 6. wRC+ offense edge, smaller than v2
    wrc_diff = team_wrc - opp_wrc
    wrc_adj  = wrc_diff * 0.0020
    wrc_adj  = clamp(wrc_adj, -0.06, 0.06)
    adjustments["wrc_plus_offense"] = round(wrc_adj, 4)

    # 7. Late-inning bullpen, regressed above
    late_adj = late_inning_edge * 0.025
    late_adj = clamp(late_adj, -0.04, 0.04)
    adjustments["late_inning_bullpen_regressed"] = round(late_adj, 4)

    # 8. Injury load, smaller because IL counts do not measure player quality
    injury_diff = len(opp_injured) - len(team_injured)
    inj_adj     = injury_diff * 0.010
    inj_adj     = clamp(inj_adj, -0.035, 0.035)
    adjustments["injury_differential"] = round(inj_adj, 4)

    # 9. Home field, tiny because it is already priced into moneylines
    hf_adj = home_field_edge * 0.020
    adjustments["home_field"] = round(hf_adj, 4)

    # ---- Final probability ----
    total_logit_adjustment = sum(adjustments.values())

    # Upset boost: when this team is the dog AND has confluent heat signals
    # (positive team_heat + positive player_heat + positive form), give an
    # extra nudge. This is what actually lets the model pick a live dog over
    # the book favorite instead of just rubber-stamping Vegas.
    is_dog = book_implied < 0.50
    upset_score = 0.0
    if HEAT_FACTORS_ENABLED:
        positive_signals = sum(1 for v in [heat_adj, player_adj, form_adj] if v > 0.01)
        if is_dog and positive_signals >= 2:
            upset_raw = (
                heat_diff * 0.4
                + player_diff * 0.3
                + (team_rd_delta - opp_rd_delta) * 0.15
                + (1 if sel_quality < opp_quality else 0) * 0.2  # dog has SP edge
            )
            upset_score = float(max(0.0, min(1.5, upset_raw)))
            upset_boost = upset_score * 0.10
            adjustments["upset_boost"] = round(upset_boost, 4)
            total_logit_adjustment += upset_boost

    model_logit = logit(book_implied) + total_logit_adjustment
    model_prob = inverse_logit(model_logit)
    # v3 clamp when heat is off, wider clamp when heat is on (so genuine upsets
    # aren't capped). Keeps the A/B comparison apples-to-apples for v3.
    if HEAT_FACTORS_ENABLED:
        model_prob = round(clamp(model_prob, 0.30, 0.75), 6)
    else:
        model_prob = round(clamp(model_prob, 0.38, 0.68), 6)
    probability_adjustment = model_prob - book_implied

    return {
        "model_prob":         model_prob,
        "book_implied":       book_implied,
        "total_adjustment":   round(probability_adjustment, 4),
        "total_logit_adjustment": round(total_logit_adjustment, 4),
        "adjustments":        adjustments,
        # Keep legacy fields for display compatibility
        "team_season":        team_season,
        "opp_season":         opp_season,
        "team_recent":        team_recent,
        "opp_recent":         opp_recent,
        "team_late":          team_late,
        "opp_late":           opp_late,
        "raw_team_late":      raw_team_late,
        "raw_opp_late":       raw_opp_late,
        "late_inning_edge":   late_inning_edge,
        "team_ops":           team_ops,
        "opp_ops":            opp_ops,
        "team_wrc":           team_wrc,
        "opp_wrc":            opp_wrc,
        "sel_p_info":         sel_p_info,
        "opp_p_info":         opp_p_info,
        "sel_recent_era":     sel_recent_era,
        "opp_recent_era":     opp_recent_era,
        "sel_quality":        sel_quality,
        "opp_quality":        opp_quality,
        "team_bp_era":        team_bp_era,
        "opp_bp_era":         opp_bp_era,
        "raw_team_bp_era":    raw_team_bp_era,
        "raw_opp_bp_era":     raw_opp_bp_era,
        "bullpen_edge":       bullpen_edge,
        "park_factor":        pf,
        "home_team_abbrev":   home_team_abbrev,
        "h2h_wins":           h2h_w,
        "h2h_losses":         h2h_l,
        "h2h_rate":           h2h_rate,
        "team_rest":          team_rest,
        "opp_rest":           opp_rest,
        "team_streak":        team_streak,
        "opp_streak":         opp_streak,
        "weather":            weather,
        "wx_run_adj":         wx_run_adj,
        "team_injured":       team_injured,
        "opp_injured":        opp_injured,
        "matchup":            matchup,
        "home_field_edge":    home_field_edge,
        "umpire_name":        umpire_name,
        "sel_fip_era_gap":    sel_fip_era_gap,
        "opp_fip_era_gap":    opp_fip_era_gap,
        "team_heat":          team_heat_data,
        "opp_heat":           opp_heat_data,
        "team_player_heat":   team_player_data,
        "opp_player_heat":    opp_player_data,
        "upset_score":        round(upset_score, 3),
        "is_dog":             is_dog,
    }


# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------

if st.button("Predict"):

    # ======================== TODAY'S SLATE (AUTO) ========================
    if prop_type == "Today's Slate (Auto)":

        # ---- Pull today's schedule ----
        with st.spinner(f"Fetching MLB schedule for {today_str}…"):
            schedule = get_mlb_schedule(today_str)

        games = []
        for db in schedule.get("dates", []):
            games.extend(db.get("games", []))

        if not games:
            st.info(f"No MLB games scheduled for {today_str}.")
            st.stop()

        st.write(f"### Found {len(games)} games on today's slate")

        # ---- Pull live odds via The Odds API ----
        live_moneylines = {}
        live_pitcher_ks = {}
        if use_live_odds:
            with st.spinner("Fetching odds (The Odds API → DraftKings)…"):
                ml_events       = fetch_odds_api_moneylines()
                live_moneylines = parse_odds_api_moneylines(ml_events)
                live_pitcher_ks = fetch_all_pitcher_ks(live_moneylines) if live_moneylines else {}

            if live_moneylines:
                st.success(
                    f"✅ Live odds pulled: "
                    f"{len(live_moneylines)} moneylines, {len(live_pitcher_ks)} pitcher Ks"
                )
            else:
                st.warning(
                    "⚠️ Could not pull live moneylines — falling back to -110 defaults. "
                    "EV numbers are informational only in this mode."
                )

        # ---- Loop games and run predictions ----
        progress = st.progress(0.0, text="Computing predictions…")
        summary  = []
        details  = []

        for i, game in enumerate(games):
            home_id = game["teams"]["home"]["team"]["id"]
            away_id = game["teams"]["away"]["team"]["id"]
            home_abbrev = ID_TEAM_MAP.get(home_id, "?")
            away_abbrev = ID_TEAM_MAP.get(away_id, "?")
            if home_abbrev == "?" or away_abbrev == "?":
                continue

            home_p   = game["teams"]["home"].get("probablePitcher") or {}
            away_p   = game["teams"]["away"].get("probablePitcher") or {}
            home_pid = home_p.get("id")
            away_pid = away_p.get("id")

            umpire = None
            for off in game.get("officials", []):
                if off.get("officialType") == "Home Plate":
                    umpire = off["official"].get("fullName")
                    break

            gt_iso = game.get("gameDate", "")
            try:
                gt_dt    = datetime.fromisoformat(gt_iso.replace("Z", "+00:00"))
                gt_local = gt_dt.strftime("%H:%M UTC")
            except Exception:
                gt_local = gt_iso[11:16] if len(gt_iso) >= 16 else "—"

            # Match live moneyline
            home_ml, away_ml = match_moneyline(live_moneylines, home_abbrev, away_abbrev)
            home_ml_used = home_ml if home_ml is not None else -110
            away_ml_used = away_ml if away_ml is not None else -110

            # Moneyline predictions — both sides
            progress.progress(
                (i + 0.2) / len(games),
                text=f"({i+1}/{len(games)}) Moneyline {away_abbrev} @ {home_abbrev}…",
            )
            try:
                home_pred = team_moneyline_probability(
                    home_abbrev, away_abbrev,
                    american_odds_input=home_ml_used,
                )
                away_pred = team_moneyline_probability(
                    away_abbrev, home_abbrev,
                    american_odds_input=away_ml_used,
                )
            except Exception as e:
                st.warning(f"Skipped {away_abbrev} @ {home_abbrev}: {e}")
                continue

            home_implied = home_pred["book_implied"]
            home_model   = home_pred["model_prob"]
            home_edge    = home_model - home_implied
            home_ev      = expected_value(home_model, home_ml_used)
            home_label   = moneyline_label(home_ev, home_model, home_implied, home_ml_used)

            away_implied = away_pred["book_implied"]
            away_model   = away_pred["model_prob"]
            away_edge    = away_model - away_implied
            away_ev      = expected_value(away_model, away_ml_used)
            away_label   = moneyline_label(away_ev, away_model, away_implied, away_ml_used)

            # Pitcher names from the moneyline model's pulled pitcher info
            home_p_name = home_pred["sel_p_info"]["name"]
            away_p_name = home_pred["opp_p_info"]["name"]

            # Pitcher K projections
            progress.progress(
                (i + 0.6) / len(games),
                text=f"({i+1}/{len(games)}) Pitcher Ks {away_abbrev} @ {home_abbrev}…",
            )
            home_k = None
            away_k = None
            if home_pid:
                try:
                    home_k = compute_pitcher_k_projection(
                        home_pid, away_abbrev,
                        home_park_abbrev=home_abbrev, umpire_name=umpire,
                    )
                except Exception as e:
                    home_k = {"error": str(e)}
            if away_pid:
                try:
                    away_k = compute_pitcher_k_projection(
                        away_pid, home_abbrev,
                        home_park_abbrev=home_abbrev, umpire_name=umpire,
                    )
                except Exception as e:
                    away_k = {"error": str(e)}

            # Match live pitcher K lines
            home_k_dk = match_pitcher_k(live_pitcher_ks, home_p_name)
            away_k_dk = match_pitcher_k(live_pitcher_ks, away_p_name)

            def build_k_bet(k_result, dk_line):
                if not k_result or "error" in k_result:
                    return None
                if not dk_line:
                    return {
                        "projection":  k_result["projection"],
                        "std_dev":     k_result["std_dev"],
                        "line":        None,
                    }
                line       = dk_line["line"]
                over_odds  = dk_line["over_odds"]
                under_odds = dk_line["under_odds"]
                std_dev    = k_result["std_dev"]
                over_prob  = estimate_over_probability(k_result["projection"], line, std_dev)
                under_prob = 1.0 - over_prob
                over_ev    = expected_value(over_prob,  over_odds)
                under_ev   = expected_value(under_prob, under_odds)
                edge       = k_result["projection"] - line
                label      = confidence_label(edge, over_ev, under_ev)
                return {
                    "projection":  k_result["projection"],
                    "std_dev":     std_dev,
                    "line":        line,
                    "over_odds":   over_odds,
                    "under_odds":  under_odds,
                    "over_prob":   over_prob,
                    "edge":        edge,
                    "over_ev":     over_ev,
                    "under_ev":    under_ev,
                    "label":       label,
                }

            home_k_bet = build_k_bet(home_k, home_k_dk)
            away_k_bet = build_k_bet(away_k, away_k_dk)

            # Summary row — headline the team the model predicts to win,
            # and flag if the better-edge side is the other team.
            if home_model >= away_model:
                win_team, win_ml = home_abbrev, home_ml_used
                win_prob, win_edge, win_ev, win_label = home_model, home_edge, home_ev, home_label
            else:
                win_team, win_ml = away_abbrev, away_ml_used
                win_prob, win_edge, win_ev, win_label = away_model, away_edge, away_ev, away_label

            if home_edge >= away_edge:
                edge_team, edge_team_edge, edge_team_label = home_abbrev, home_edge, home_label
            else:
                edge_team, edge_team_edge, edge_team_label = away_abbrev, away_edge, away_label

            best_bet = (
                f"{edge_team} ({edge_team_edge*100:+.1f}%)"
                if edge_team_label != "No Bet"
                else "—"
            )

            summary.append({
                "Game":            f"{away_abbrev} @ {home_abbrev}",
                "Time":            gt_local,
                "Predicted Winner": f"{win_team} ({win_ml:+d})",
                "Win %":           f"{win_prob*100:.1f}%",
                "Edge":            f"{win_edge*100:+.1f}%",
                "EV":              f"{win_ev*100:+.1f}%",
                "Verdict":         win_label,
                "Best Bet":        best_bet,
                "_sort":           max(home_edge, away_edge),
            })

            details.append({
                "game":         f"{away_abbrev} @ {home_abbrev}",
                "time":         gt_local,
                "home_abbrev":  home_abbrev,
                "away_abbrev":  away_abbrev,
                "umpire":       umpire,
                "home_pred":    home_pred,
                "away_pred":    away_pred,
                "home_ml":      home_ml_used,
                "away_ml":      away_ml_used,
                "home_ml_real": home_ml,    # None if DK didn't have it
                "away_ml_real": away_ml,
                "home_implied": home_implied,
                "away_implied": away_implied,
                "home_model":   home_model,
                "away_model":   away_model,
                "home_edge":    home_edge,
                "away_edge":    away_edge,
                "home_ev":      home_ev,
                "away_ev":      away_ev,
                "home_label":   home_label,
                "away_label":   away_label,
                "home_p_name":  home_p_name,
                "away_p_name":  away_p_name,
                "home_k":       home_k,
                "away_k":       away_k,
                "home_k_bet":   home_k_bet,
                "away_k_bet":   away_k_bet,
                "_sort":        max(home_edge, away_edge),
            })

            progress.progress((i + 1) / len(games), text=f"Done {i+1}/{len(games)}")

        progress.empty()

        # ============ Summary: Moneyline ============
        st.subheader("📋 Moneyline Summary — Best Side per Game")
        filtered = [s for s in summary if abs(s["_sort"]) * 100 >= min_edge_filter]
        filtered.sort(key=lambda x: -x["_sort"])

        if not filtered:
            st.info(f"No games meet the ≥{min_edge_filter:.1f}% edge filter.")
        else:
            df_summary = pd.DataFrame(filtered).drop(columns=["_sort"])
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            n_strong = sum(1 for s in filtered if s["Verdict"] == "Strong Bet")
            n_lean   = sum(1 for s in filtered if s["Verdict"] == "Lean")
            st.caption(f"🟢 {n_strong} Strong Bet · 🟡 {n_lean} Lean · ⚪ {len(filtered) - n_strong - n_lean} No Bet")

        # ============ Summary: Pitcher Ks ============
        st.subheader("⚾ Pitcher Strikeouts Summary")
        k_rows = []
        for d in details:
            for side_label, p_name, k_bet in [
                ("Away", d["away_p_name"], d["away_k_bet"]),
                ("Home", d["home_p_name"], d["home_k_bet"]),
            ]:
                if k_bet is None:
                    continue
                row = {
                    "Game":    d["game"],
                    "Pitcher": p_name,
                    "Side":    side_label,
                    "Proj K":  f"{k_bet['projection']:.2f}",
                    "DK Line": f"{k_bet['line']:.1f}" if k_bet.get("line") is not None else "—",
                    "Over %":  f"{k_bet['over_prob']*100:.1f}%" if "over_prob" in k_bet else "—",
                    "Edge":    f"{k_bet['edge']:+.2f}"           if "edge"      in k_bet else "—",
                    "EV (O)":  f"{k_bet['over_ev']*100:+.1f}%"   if "over_ev"   in k_bet else "—",
                    "Verdict": k_bet.get("label", "Projection Only"),
                    "_sort":   abs(k_bet.get("over_ev", 0)),
                }
                k_rows.append(row)
        k_rows.sort(key=lambda x: -x["_sort"])
        if k_rows:
            df_k = pd.DataFrame(k_rows).drop(columns=["_sort"])
            st.dataframe(df_k, use_container_width=True, hide_index=True)
        else:
            st.info("No pitcher K data available (probable starters undetermined or Statcast missing).")

        # ============ Per-game detail ============
        st.subheader("🔬 Per-Game Detail")
        details_sorted = sorted(details, key=lambda d: -d["_sort"])
        for d in details_sorted:
            has_strong = "Strong" in (d["home_label"] + d["away_label"])
            has_lean   = "Lean"   in (d["home_label"] + d["away_label"])
            emoji      = "🟢" if has_strong else ("🟡" if has_lean else "⚪")
            with st.expander(f"{emoji} {d['game']} — {d['time']}"):
                # Game header
                ml_src = "DK" if d["home_ml_real"] is not None else "default -110"
                st.caption(
                    f"Moneyline odds source: **{ml_src}** · "
                    f"Home Plate Umpire: {d['umpire'] or 'TBD'}"
                )

                # ML side-by-side
                ml_cols = st.columns(2)
                for col, team_lbl, model, implied, edge_v, ev_v, label_v, ml_v, p_name in [
                    (ml_cols[0], d["away_abbrev"], d["away_model"], d["away_implied"],
                     d["away_edge"], d["away_ev"], d["away_label"], d["away_ml"], d["away_p_name"]),
                    (ml_cols[1], d["home_abbrev"], d["home_model"], d["home_implied"],
                     d["home_edge"], d["home_ev"], d["home_label"], d["home_ml"], d["home_p_name"]),
                ]:
                    with col:
                        st.markdown(f"**{team_lbl}** ({ml_v:+d}) — *{p_name}*")
                        st.write(f"Model: {model*100:.1f}% · Book: {implied*100:.1f}%")
                        st.write(f"Edge: **{edge_v*100:+.1f}%** · EV: **{ev_v*100:+.1f}%**")
                        if label_v == "Strong Bet":
                            st.success(f"✅ {label_v}")
                        elif label_v == "Lean":
                            st.info(f"📊 {label_v}")
                        else:
                            st.caption(f"⛔ {label_v}")

                # Top moneyline adjustments for the better side
                pred = d["home_pred"] if d["home_edge"] >= d["away_edge"] else d["away_pred"]
                side = d["home_abbrev"] if d["home_edge"] >= d["away_edge"] else d["away_abbrev"]
                adj_rows = sorted(pred["adjustments"].items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                st.markdown(f"**Top moneyline factors for {side}**")
                for factor, val in adj_rows:
                    arrow = "▲" if val > 0 else ("▼" if val < 0 else "—")
                    st.text(f"  {arrow} {factor:<30} logit {val:+.4f}")

                # ---- Upset Watch ----
                dog_pred = d["home_pred"] if d["home_implied"] < d["away_implied"] else d["away_pred"]
                dog_side = d["home_abbrev"] if d["home_implied"] < d["away_implied"] else d["away_abbrev"]
                if dog_pred.get("upset_score", 0) >= 0.30:
                    th = dog_pred.get("team_heat", {})
                    ph = dog_pred.get("team_player_heat", {})
                    st.warning(
                        f"🔥 **Upset Watch — {dog_side}** "
                        f"(upset score {dog_pred['upset_score']:.2f})"
                    )
                    bullets = []
                    if th:
                        bullets.append(
                            f"Last 10: {th.get('last10_record', '?')} · "
                            f"run diff {th.get('last10_run_diff', 0):+.2f} · "
                            f"streak {int(th.get('current_streak', 0)):+d}"
                        )
                    if ph and ph.get("hot"):
                        names = ", ".join(
                            f"{n.split(',')[0].strip() if ',' in n else n} (+{dlt:.3f} wOBA)"
                            for n, dlt in ph["hot"][:3]
                        )
                        bullets.append(f"Hot bats: {names}")
                    # Also flag if the favorite's roster has cold bats
                    fav_pred = d["home_pred"] if d["home_implied"] >= d["away_implied"] else d["away_pred"]
                    fav_side = d["home_abbrev"] if d["home_implied"] >= d["away_implied"] else d["away_abbrev"]
                    fav_ph = fav_pred.get("team_player_heat", {})
                    if fav_ph and fav_ph.get("cold"):
                        cold_names = ", ".join(
                            f"{n.split(',')[0].strip() if ',' in n else n} ({dlt:+.3f} wOBA)"
                            for n, dlt in fav_ph["cold"][:2]
                        )
                        bullets.append(f"{fav_side} cold bats: {cold_names}")
                    for b in bullets:
                        st.caption(f"  • {b}")

                # Pitcher K detail
                st.markdown("**Pitcher Strikeouts**")
                k_cols = st.columns(2)
                for col, p_name, k_result, k_bet in [
                    (k_cols[0], d["away_p_name"], d["away_k"], d["away_k_bet"]),
                    (k_cols[1], d["home_p_name"], d["home_k"], d["home_k_bet"]),
                ]:
                    with col:
                        st.markdown(f"*{p_name}*")
                        if not k_result:
                            st.caption("No starter listed yet.")
                            continue
                        if "error" in k_result:
                            st.caption(f"⚠️ {k_result['error']}")
                            continue
                        st.write(f"Projection: **{k_result['projection']:.2f}** Ks (±{k_result['std_dev']:.2f})")
                        st.caption(
                            f"Season: {k_result['season_avg']:.2f} ({k_result['season_starts']} GS) · "
                            f"Recent: {k_result['recent_avg']:.2f} ({k_result['recent_starts']} GS)"
                        )
                        if k_bet and k_bet.get("line") is not None:
                            st.write(
                                f"DK Line: {k_bet['line']:.1f} "
                                f"({k_bet['over_odds']:+d} / {k_bet['under_odds']:+d})"
                            )
                            st.write(
                                f"Over: {k_bet['over_prob']*100:.1f}% · "
                                f"Edge {k_bet['edge']:+.2f} · "
                                f"EV {k_bet['over_ev']*100:+.1f}%"
                            )
                            if k_bet["label"] in ("Strong Over", "Lean Over"):
                                st.success(f"✅ {k_bet['label']}")
                            elif k_bet["label"] in ("Strong Under", "Lean Under"):
                                st.warning(f"⬇️ {k_bet['label']}")
                            else:
                                st.caption(f"⛔ {k_bet['label']}")
                        else:
                            st.caption("No DK strikeout line found — projection only.")

        st.stop()  # don't fall through to manual modes

    # ======================== MONEYLINE ========================
    if prop_type == "Team Moneyline":

        with st.spinner("Fetching data and computing…"):
            result = team_moneyline_probability(
                team, opponent_team,
                team_starter_first, team_starter_last,
                opponent_starter_first, opponent_starter_last,
                american_odds_input=american_odds,
            )

        model_prob   = result["model_prob"]
        implied_prob = result["book_implied"]
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
            st.warning("Could not find today's matchup. Check abbreviations.")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Book Implied",       f"{implied_prob*100:.1f}%",
                    help="Starting point — what the book says the team's win prob is")
        col2.metric("Model Adjustment",   f"{result['total_adjustment']*100:+.1f}%",
                    help="Probability shift after log-odds adjustment")
        col3.metric("Model Win Prob",     f"{model_prob*100:.1f}%",
                    help="Book implied adjusted in log-odds space")
        col4.metric("Edge vs Book",       f"{edge_pct*100:+.1f}%")
        col5.metric("Expected Value",     f"{ev*100:.1f}%")

        st.caption(
            "🟢 Strong Bet = edge ≥4% + EV ≥4%  |  "
            "🟡 Lean = edge ≥2% + EV ≥2%  |  "
            "🔴 No Bet = non-positive edge, or chalk without sufficient edge  |  "
            "Heavy chalk (≤−200) needs ≥5% edge"
        )

        if label == "Strong Bet":
            st.success("✅ STRONG BET — Multiple mispriced factors + clear value over the book")
        elif label == "Lean":
            st.success("📊 LEAN — Model finds real edge the book may have missed")
        else:
            st.info("⛔ NO BET — Edge or EV below the Lean threshold (or heavy chalk without enough edge)")

        # ---- Bet filter checklist ----
        st.subheader("🔎 Bet Filter Checklist")
        edge_ok_strong     = edge_pct >= 0.04
        edge_ok_lean       = edge_pct >= 0.02
        ev_ok_strong       = ev >= 0.04
        ev_ok_lean         = ev >= 0.02
        positive_edge      = edge_pct > 0
        chalk_blocked      = american_odds <= -200 and edge_pct < 0.05

        def check(cond): return "✅" if cond else "❌"

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**Strong Bet (all must pass)**")
            st.write(f"{check(edge_ok_strong)} Edge over book ≥4%  →  {edge_pct*100:+.1f}%")
            st.write(f"{check(ev_ok_strong)} Expected value ≥4%  →  {ev*100:.1f}%")
            st.write(f"{check(positive_edge)} Positive edge (model > book)")
            if chalk_blocked:
                st.write(f"❌ Heavy chalk (≤−200) blocked — need ≥5% edge, have {edge_pct*100:.1f}%")
        with fc2:
            st.markdown("**Lean (all must pass)**")
            st.write(f"{check(edge_ok_lean)} Edge over book ≥2%  →  {edge_pct*100:+.1f}%")
            st.write(f"{check(ev_ok_lean)} Expected value ≥2%  →  {ev*100:.1f}%")
            st.write(f"{check(positive_edge)} Positive edge (model > book)")

        # ---- Adjustment breakdown (replaces score breakdown) ----
        st.subheader("🔬 Factor Adjustment Breakdown")
        st.caption(
            f"**Book implied: {implied_prob*100:.1f}%** + "
            f"**Total adjustment: {result['total_adjustment']*100:+.1f}%** = "
            f"**Model prob: {model_prob*100:.1f}%**"
        )
        st.caption(
            "Each factor is a log-odds adjustment; the total is converted back into probability. "
            "Positive = favors your team. Only factors that books are known to misprice are included."
            " Values below are log-odds deltas, not raw probability points."
        )

        adj_rows = sorted(result["adjustments"].items(), key=lambda x: abs(x[1]), reverse=True)
        for factor, adj_val in adj_rows:
            pct_str   = f"logit {adj_val:+.4f}"
            bar_len   = int(abs(adj_val) * 120)
            bar       = "█" * min(bar_len, 20)
            direction = "▲" if adj_val > 0 else ("▼" if adj_val < 0 else "—")
            label_map = {
                "fip_vs_era_gap":         "FIP vs ERA gap (starters)",
                "bullpen_era":            "Bullpen ERA differential",
                "bullpen_era_regressed":  "Bullpen ERA differential",
                "rest_days":              "Rest days advantage",
                "weather":                "Weather run factor",
                "recent_form_divergence": "Recent form vs season trend",
                "recent_run_diff_divergence": "Recent run-diff divergence",
                "wrc_plus_offense":       "wRC+ offense edge",
                "late_inning_bullpen":    "Late-inning (7-9) bullpen",
                "late_inning_bullpen_regressed": "Late-inning bullpen",
                "injury_differential":    "IL injury differential",
                "home_field":             "Home field",
                "team_heat":              "Team fire streak (last-10 + run-diff)",
                "player_heat":            "Player hot bats vs opponent",
                "upset_boost":            "Live underdog convergence boost",
            }
            display_name = label_map.get(factor, factor)
            st.text(f"  {direction} {display_name:<35} {pct_str}  {bar}")

        # FIP detail
        st.caption(
            f"FIP detail — {team.upper()} starter: ERA {result['sel_p_info']['era']:.2f} / "
            f"FIP {result['sel_p_info']['fip']:.2f} (gap: {result['sel_fip_era_gap']:+.2f}) | "
            f"{opponent_team.upper()} starter: ERA {result['opp_p_info']['era']:.2f} / "
            f"FIP {result['opp_p_info']['fip']:.2f} (gap: {result['opp_fip_era_gap']:+.2f})"
        )

        # ---- Upset Watch / Fire Streak panel ----
        st.subheader("🔥 Fire Streak / Upset Watch")
        th = result.get("team_heat", {})
        oh = result.get("opp_heat", {})
        ph = result.get("team_player_heat", {})
        oph = result.get("opp_player_heat", {})
        hc1, hc2 = st.columns(2)
        with hc1:
            st.markdown(f"**{team.upper()}**")
            st.write(f"Last 10: {th.get('last10_record', '?')}")
            st.write(f"Recent run diff: {th.get('last10_run_diff', 0):+.2f}")
            st.write(f"Current streak: {int(th.get('current_streak', 0)):+d}")
            st.write(f"Heat score: **{th.get('score', 0):+.2f}**")
            if ph.get("hot"):
                st.caption("Hot bats:")
                for n, dlt in ph["hot"][:3]:
                    short = n.split(",")[0].strip() if "," in n else n
                    st.text(f"  🔥 {short}: {dlt:+.3f} wOBA vs season")
            if ph.get("cold"):
                st.caption("Cold bats:")
                for n, dlt in ph["cold"][:2]:
                    short = n.split(",")[0].strip() if "," in n else n
                    st.text(f"  🧊 {short}: {dlt:+.3f} wOBA vs season")
        with hc2:
            st.markdown(f"**{opponent_team.upper()}**")
            st.write(f"Last 10: {oh.get('last10_record', '?')}")
            st.write(f"Recent run diff: {oh.get('last10_run_diff', 0):+.2f}")
            st.write(f"Current streak: {int(oh.get('current_streak', 0)):+d}")
            st.write(f"Heat score: **{oh.get('score', 0):+.2f}**")
            if oph.get("hot"):
                st.caption("Hot bats:")
                for n, dlt in oph["hot"][:3]:
                    short = n.split(",")[0].strip() if "," in n else n
                    st.text(f"  🔥 {short}: {dlt:+.3f} wOBA vs season")
            if oph.get("cold"):
                st.caption("Cold bats:")
                for n, dlt in oph["cold"][:2]:
                    short = n.split(",")[0].strip() if "," in n else n
                    st.text(f"  🧊 {short}: {dlt:+.3f} wOBA vs season")

        upset_score = result.get("upset_score", 0)
        if result.get("is_dog") and upset_score >= 0.30:
            st.warning(
                f"🚨 **Live Underdog signal — {team.upper()}** "
                f"(upset score {upset_score:.2f}). "
                f"Multiple heat factors agree against a Vegas favorite."
            )
        elif result.get("is_dog"):
            st.caption(
                f"No upset signal: dog needs ≥2 positive heat factors. "
                f"Current upset score {upset_score:.2f}."
            )

        # ---- Starting Pitchers ----
        st.subheader("⚾ Starting Pitchers")
        for abbrev, info, recent_era, quality in [
            (team.upper(),          result["sel_p_info"], result["sel_recent_era"], result["sel_quality"]),
            (opponent_team.upper(), result["opp_p_info"], result["opp_recent_era"], result["opp_quality"]),
        ]:
            st.markdown(f"**{abbrev}: {info['name']}** (Throws: {info['throws']})")
            pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns(5)
            pcol1.metric("ERA",           f"{info['era']:.2f}")
            pcol2.metric("FIP",           f"{info['fip']:.2f}")
            pcol3.metric("xFIP",          f"{info['xfip']:.2f}")
            pcol4.metric("SIERA",         f"{info['siera']:.2f}")
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
        st.caption("wRC+ is park-adjusted; 100 = league average. Capped at 80-120 when using fallback estimate.")
        st.write(f"{team.upper()} OPS vs {result['opp_p_info']['throws']}HP: {result['team_ops']:.3f}")
        st.write(f"{opponent_team.upper()} OPS vs {result['sel_p_info']['throws']}HP: {result['opp_ops']:.3f}")

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
        lcol1.metric(f"{team.upper()} RA/G (7-9)",          f"{result['team_late']:.2f}")
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

        # ---- Removed noisy factors ----
        st.subheader("🧹 Removed Noisy Factors")
        st.caption("Head-to-head record and win/loss streaks are shown nowhere in the model because they add noise and can create false edges.")

        # ---- Rest / Fatigue ----
        st.subheader("😴 Rest")
        rcol1, rcol2 = st.columns(2)
        rcol1.metric(f"{team.upper()} Days Rest",          result["team_rest"])
        rcol2.metric(f"{opponent_team.upper()} Days Rest", result["opp_rest"])

        # ---- Weather ----
        st.subheader("🌤️ Weather (Home Park)")
        wx = result["weather"]
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)
        wcol1.metric("Temp",       f"{wx['temp_f']:.0f}°F")
        wcol2.metric("Wind Speed", f"{wx['wind_mph']:.0f} mph")
        wcol3.metric("Wind Dir",   f"{wx['wind_dir_deg']:.0f}°")
        wcol4.metric("Run Factor", f"{result['wx_run_adj']:.3f}")
        if result["wx_run_adj"] > 1.04:
            st.caption("☀️ Hot/wind-out conditions favor hitters today.")
        elif result["wx_run_adj"] < 0.96:
            st.caption("❄️ Cold/wind-in conditions favor pitchers today.")

        # ---- Injuries ----
        st.subheader("🏥 Injured List")
        col_inj1, col_inj2 = st.columns(2)
        with col_inj1:
            st.write(f"**{team.upper()} IL ({len(result['team_injured'])} players)**")
            for p in result["team_injured"][:10]:
                st.caption(f"• {p}")
            if not result["team_injured"]:
                st.caption("No IL data retrieved.")
        with col_inj2:
            st.write(f"**{opponent_team.upper()} IL ({len(result['opp_injured'])} players)**")
            for p in result["opp_injured"][:10]:
                st.caption(f"• {p}")
            if not result["opp_injured"]:
                st.caption("No IL data retrieved.")

        # ---- Park ----
        st.subheader("🏟️ Park")
        st.write(
            f"Home Park ({result['home_team_abbrev'].upper()}) "
            f"Run Factor: {result['park_factor']:.3f}"
        )

        # ---- Factor Tracking Log ----
        st.subheader("📓 Log This Pick for Calibration")
        st.caption(
            "Track today's picks and enter outcomes later to calibrate which "
            "factors are actually predictive over time."
        )
        if st.button("📝 Log this pick"):
            pick_entry = {
                "date":             today_str,
                "team":             team,
                "opponent":         opponent_team,
                "american_odds":    american_odds,
                "book_implied":     round(implied_prob, 4),
                "model_prob":       round(model_prob, 4),
                "edge_pct":         round(edge_pct, 4),
                "ev":               round(ev, 4),
                "label":            label,
                "adjustments":      result["adjustments"],
                "outcome":          None,   # fill in later
            }
            log_pick(pick_entry)
            st.success(f"Logged: {team} vs {opponent_team} — {label}")

        # ---- Outcome entry for past logged picks ----
        tracking_data = load_tracking()
        open_picks    = [p for p in tracking_data["picks"] if p.get("outcome") is None]
        if open_picks:
            st.subheader("📊 Enter Outcomes for Logged Picks")
            for i, pick in enumerate(open_picks):
                cols = st.columns([3, 1, 1])
                cols[0].write(
                    f"{pick['date']} — {pick['team']} vs {pick['opponent']} "
                    f"({pick['label']}, {pick['model_prob']*100:.1f}%)"
                )
                if cols[1].button("✅ Won", key=f"win_{i}"):
                    pick["outcome"] = "W"
                    save_tracking(tracking_data)
                    st.rerun()
                if cols[2].button("❌ Lost", key=f"loss_{i}"):
                    pick["outcome"] = "L"
                    save_tracking(tracking_data)
                    st.rerun()

        # ---- Calibration summary ----
        resolved = [p for p in tracking_data["picks"] if p.get("outcome") in ("W", "L")]
        if len(resolved) >= 5:
            st.subheader("📈 Calibration Summary")
            wins   = sum(1 for p in resolved if p["outcome"] == "W")
            total  = len(resolved)
            avg_ev = sum(p["ev"] for p in resolved) / total

            st.write(f"Overall record: **{wins}-{total-wins}** ({wins/total*100:.1f}% win rate) | Avg EV: {avg_ev*100:.1f}%")

            # Per-factor performance
            factor_wins = {}
            factor_total = {}
            for pick in resolved:
                for factor, val in pick.get("adjustments", {}).items():
                    if abs(val) > 0.003:   # only count meaningful contributions
                        direction = "positive" if val > 0 else "negative"
                        key = f"{factor}_{direction}"
                        factor_total[key] = factor_total.get(key, 0) + 1
                        if pick["outcome"] == "W":
                            factor_wins[key] = factor_wins.get(key, 0) + 1

            if factor_total:
                st.write("**Factor win rates** (when that factor favored your team):")
                factor_rows = sorted(factor_total.items(), key=lambda x: -x[1])
                for key, count in factor_rows:
                    if count >= 3:
                        wr = factor_wins.get(key, 0) / count
                        st.write(f"  {key}: {wr*100:.0f}% ({factor_wins.get(key,0)}-{count-factor_wins.get(key,0)})")
                st.caption("Factors with win rate <50% over 10+ picks should have their weight reduced in the model.")

    # ======================== PLAYER PROPS ========================
    else:
        player_id = get_player_id(last_name, first_name)

        if player_id is None:
            st.error("❌ Player not found. Check spelling of first and last name.")
            st.stop()

        MIN_GAMES_SEASON = 5
        MIN_GAMES_RECENT = 3

        with st.spinner("Fetching Statcast data…"):
            if prop_type == "Pitcher Strikeouts":
                season_data    = statcast_pitcher(season_start, season_end, player_id)
                recent_data    = statcast_pitcher(recent_start, recent_end, player_id)

                if season_data.empty:
                    st.error("❌ No Statcast data found for this pitcher this season.")
                    st.stop()

                season_by_game = season_data[season_data["events"] == "strikeout"].groupby("game_date").size()
                recent_by_game = recent_data[recent_data["events"] == "strikeout"].groupby("game_date").size()
                season_starts  = season_data["game_date"].nunique()
                recent_starts  = recent_data["game_date"].nunique() if not recent_data.empty else 0
                stat_name      = "Strikeouts"

            elif prop_type == "Batter Hits":
                season_data    = statcast_batter(season_start, season_end, player_id)
                recent_data    = statcast_batter(recent_start, recent_end, player_id)

                if season_data.empty:
                    st.error("❌ No Statcast data found for this batter this season.")
                    st.stop()

                hit_events     = ["single", "double", "triple", "home_run"]
                season_by_game = season_data[season_data["events"].isin(hit_events)].groupby("game_date").size()
                recent_by_game = recent_data[recent_data["events"].isin(hit_events)].groupby("game_date").size()
                season_starts  = season_data["game_date"].nunique()
                recent_starts  = recent_data["game_date"].nunique() if not recent_data.empty else 0
                stat_name      = "Hits"

            else:  # Total Bases
                season_data    = statcast_batter(season_start, season_end, player_id)
                recent_data    = statcast_batter(recent_start, recent_end, player_id)

                if season_data.empty:
                    st.error("❌ No Statcast data found for this batter this season.")
                    st.stop()

                season_data    = season_data.copy()
                recent_data    = recent_data.copy()
                season_data["tb"] = season_data["events"].apply(total_bases_from_event)
                recent_data["tb"] = recent_data["events"].apply(total_bases_from_event)
                season_by_game = season_data.groupby("game_date")["tb"].sum()
                recent_by_game = recent_data.groupby("game_date")["tb"].sum()
                season_starts  = season_data["game_date"].nunique()
                recent_starts  = recent_data["game_date"].nunique() if not recent_data.empty else 0
                stat_name      = "Total Bases"

        # ---- Data quality gate ----
        data_warnings = []
        data_errors   = []

        if season_by_game.empty:
            data_errors.append(f"No {stat_name.lower()} recorded in Statcast data this season.")
        elif season_starts < MIN_GAMES_SEASON:
            data_errors.append(
                f"Only **{season_starts} games** found this season — need at least "
                f"{MIN_GAMES_SEASON} for a reliable season average."
            )

        if data_errors:
            st.error("❌ Insufficient data to predict")
            for err in data_errors:
                st.write(f"• {err}")
            st.stop()

        if season_starts < 10:
            data_warnings.append(f"⚠️ Small season sample ({season_starts} games) — projection less reliable.")
        if recent_starts < MIN_GAMES_RECENT:
            data_warnings.append(f"⚠️ Only {recent_starts} games in last 30 days — using season average only.")

        # ---- Core projection ----
        season_avg = season_by_game.mean()

        if not recent_data.empty and recent_starts > 0:
            recent_game_dates = recent_data["game_date"].unique()
            if prop_type == "Pitcher Strikeouts":
                recent_by_game_filled = recent_by_game.reindex(recent_game_dates, fill_value=0)
            else:
                pa_games = recent_data[recent_data["events"].notna()]["game_date"].unique()
                recent_by_game_filled = recent_by_game.reindex(pa_games, fill_value=0)

            recent_avg = recent_by_game_filled.mean() if len(recent_by_game_filled) >= MIN_GAMES_RECENT else float("nan")
        else:
            recent_avg = float("nan")

        projection = calc_weighted_projection(season_avg, recent_avg)

        # ---- Prop-specific adjustments ----
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
        wx_hit_adj       = 1.0

        if prop_type == "Pitcher Strikeouts":
            # v2: blended 70/30 season/recent K rate, capped at ±12%
            adj = opponent_k_adjustment(opponent_team)

            k_park_adj = PARK_K_FACTORS.get(opponent_team.upper(), 1.0)

            schedule_data = get_mlb_schedule(today_str)
            for db in schedule_data.get("dates", []):
                for game in db.get("games", []):
                    for off in game.get("officials", []):
                        if off.get("officialType") == "Home Plate":
                            umpire_name_prop = off["official"].get("fullName")
            umpire_adj = get_umpire_k_adjustment(umpire_name_prop)

            # v2: SwStr% capped at ±15% (was ±30%), min 75 pitches (was 50)
            swstr_pct = get_pitcher_swstr(player_id)
            if swstr_pct is not None:
                swstr_factor = 1.0 + (swstr_pct - 0.11) * 2.0   # was *3.0
                adj *= max(min(swstr_factor, 1.15), 0.85)         # was ±30%
            else:
                data_warnings.append("⚠️ SwStr% unavailable (fewer than 75 pitches in last 30 days) — skipped.")

            fip_data     = get_pitcher_info(player_id)
            coords       = PARK_COORDS.get(opponent_team.upper(), PARK_COORDS.get("NYY"))
            weather_info = get_weather(coords[0], coords[1], today_str)
            wx_k_adj     = weather_k_factor(weather_info["temp_f"], weather_info["wind_mph"])

            projection = projection * adj * k_park_adj * umpire_adj * wx_k_adj

        elif prop_type in ("Batter Hits", "Batter Total Bases"):
            schedule_data  = get_mlb_schedule(today_str)
            opp_pitcher_id = None

            for db in schedule_data.get("dates", []):
                for game in db.get("games", []):
                    home_id = game["teams"]["home"]["team"]["id"]
                    away_id = game["teams"]["away"]["team"]["id"]
                    opp_id  = TEAM_ID_MAP.get(opponent_team.upper())
                    if opp_id in (home_id, away_id):
                        p = game["teams"]["home"].get("probablePitcher") if home_id == opp_id \
                            else game["teams"]["away"].get("probablePitcher")
                        if p:
                            opp_pitcher_id = p["id"]
                        break

            if opp_pitcher_id:
                opp_info        = get_pitcher_info(opp_pitcher_id)
                pitcher_hand    = opp_info["throws"]
                opp_pitcher_era = opp_info["era"]
                opp_fip         = opp_info.get("fip", 4.20)

                fip_difficulty  = (opp_fip - 4.20) / 4.20
                hit_pitcher_adj = max(0.80, min(1.20, 1.0 + fip_difficulty * 0.5))

                batter_ops_vs_hand = calculate_team_ops_vs_hand(opponent_team, pitcher_hand)
                platoon_factor     = batter_ops_vs_hand / 0.700
                hit_handedness     = max(0.85, min(1.15, platoon_factor))

                projection = projection * hit_pitcher_adj * hit_handedness
            else:
                data_warnings.append("⚠️ Could not find today's opposing pitcher — pitcher adjustments skipped.")

            park_run_factor = PARK_FACTORS.get(opponent_team.upper(), 1.00)
            projection      = projection * park_run_factor

            coords       = PARK_COORDS.get(opponent_team.upper(), PARK_COORDS.get("NYY"))
            weather_info = get_weather(coords[0], coords[1], today_str)
            wx_hit_adj   = weather_run_factor(
                weather_info["temp_f"], weather_info["wind_mph"], weather_info["wind_dir_deg"]
            )
            wx_hit_adj   = max(0.97, min(1.03, wx_hit_adj))
            projection   = projection * wx_hit_adj

        # ---- Probability & EV ----
        std_dev    = float(season_by_game.std()) if len(season_by_game) > 1 else 1.5
        std_dev    = max(std_dev, 0.8)

        edge       = projection - sportsbook_line
        over_prob  = estimate_over_probability(projection, sportsbook_line, std_dev)
        under_prob = 1.0 - over_prob

        over_implied  = implied_probability(over_odds)
        under_implied = implied_probability(under_odds)

        over_ev   = expected_value(over_prob,  over_odds)
        under_ev  = expected_value(under_prob, under_odds)

        total_implied = over_implied + under_implied
        vig_pct       = (total_implied - 1.0) * 100

        label        = confidence_label(edge, over_ev, under_ev)
        implied_prob = over_implied
        ev           = over_ev

        # ---- Display warnings ----
        if data_warnings:
            for w in data_warnings:
                st.warning(w)

        # ---- Results ----
        st.subheader(f"📊 {stat_name} Prediction")

        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        rcol1.metric("Season Avg",      f"{season_avg:.2f}",
                     help=f"Based on {season_starts} games this season")
        rcol2.metric("Recent Avg",      f"{recent_avg:.2f}" if not math.isnan(recent_avg) else "N/A",
                     help=f"Based on {recent_starts} games in last 30 days")
        rcol3.metric("Projection",      f"{projection:.2f}")
        rcol4.metric("Sportsbook Line", f"{sportsbook_line:.1f}")

        if season_starts >= 20:
            sample_quality = "🟢 Good sample"
        elif season_starts >= 10:
            sample_quality = "🟡 Moderate sample"
        else:
            sample_quality = "🔴 Small sample — treat with caution"

        st.caption(
            f"{sample_quality} | Season: {season_starts} games | "
            f"Recent (30d): {recent_starts} games | Std Dev: ±{std_dev:.2f}"
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
            acol1.metric("Opp K-Rate",  f"{adj:.3f}x",
                         help="70% season / 30% recent K rate vs league avg — capped at ±12%")
            acol2.metric("Park Factor", f"{k_park_adj:.3f}x")
            acol3.metric("Umpire",      f"{umpire_adj:.3f}x",
                         help=f"Umpire: {umpire_name_prop or 'Unknown'}")
            acol4.metric("Weather",     f"{wx_k_adj:.3f}x")

            if swstr_pct is not None:
                st.metric("SwStr% Signal", f"{swstr_pct*100:.1f}%",
                          help="Swinging strike %. League avg ~11%. Capped at ±15% adjustment.")
            if weather_info:
                st.caption(f"🌤️ Park weather: {weather_info['temp_f']:.0f}°F, {weather_info['wind_mph']:.0f} mph wind")

        elif prop_type in ("Batter Hits", "Batter Total Bases"):
            st.subheader("🔧 Hit/TB Adjustments")
            if opp_pitcher_era is not None:
                acol1, acol2, acol3, acol4 = st.columns(4)
                acol1.metric("Pitcher Difficulty", f"{hit_pitcher_adj:.3f}x",
                             help=f"Opp pitcher ERA {opp_pitcher_era:.2f}, FIP-adjusted")
                acol2.metric("Platoon Factor",     f"{hit_handedness:.3f}x",
                             help=f"Batter vs {pitcher_hand}HP")
                pf_val = PARK_FACTORS.get(opponent_team.upper(), 1.00)
                acol3.metric("Park Factor",        f"{pf_val:.3f}x")
                acol4.metric("Weather",            f"{wx_hit_adj:.3f}x",
                             help="Capped ±3%")
                if pitcher_hand:
                    st.caption(f"Opposing pitcher throws: **{pitcher_hand}**")
            if weather_info:
                st.caption(f"🌤️ Park weather: {weather_info['temp_f']:.0f}°F, {weather_info['wind_mph']:.0f} mph wind")

        # ---- Betting value ----
        st.subheader("💰 Betting Value")
        st.write(
            f"Projection: **{projection:.2f}** vs Line: **{sportsbook_line:.1f}** | "
            f"Edge: **{edge:+.2f}** | Model Prob Over: **{over_prob*100:.1f}%** | "
            f"Std Dev: ±{std_dev:.2f}"
        )
        st.caption(f"Book vig: {vig_pct:.1f}% (total implied = {total_implied*100:.1f}%)")

        over_col, under_col = st.columns(2)
        with over_col:
            st.markdown("### 📈 OVER")
            o1, o2, o3 = st.columns(3)
            o1.metric("Your Odds",    f"{over_odds:+d}")
            o2.metric("Book Implied", f"{over_implied*100:.1f}%")
            o3.metric("Model Prob",   f"{over_prob*100:.1f}%")
            edge_over    = over_prob - over_implied
            ev_over_pct  = over_ev * 100
            st.metric("Edge vs Book",   f"{edge_over*100:+.1f}%",
                      delta_color="normal" if edge_over > 0 else "inverse")
            st.metric("Expected Value", f"{ev_over_pct:.1f}%",
                      delta_color="normal" if over_ev > 0 else "inverse")
            if label in ("Strong Over", "Lean Over"):
                st.success("✅ Strong OVER" if label == "Strong Over" else "📊 Lean OVER")
            else:
                st.info("No value on OVER")

        with under_col:
            st.markdown("### 📉 UNDER")
            u1, u2, u3 = st.columns(3)
            u1.metric("Your Odds",    f"{under_odds:+d}")
            u2.metric("Book Implied", f"{under_implied*100:.1f}%")
            u3.metric("Model Prob",   f"{under_prob*100:.1f}%")
            edge_under   = under_prob - under_implied
            ev_under_pct = under_ev * 100
            st.metric("Edge vs Book",   f"{edge_under*100:+.1f}%",
                      delta_color="normal" if edge_under > 0 else "inverse")
            st.metric("Expected Value", f"{ev_under_pct:.1f}%",
                      delta_color="normal" if under_ev > 0 else "inverse")
            if label in ("Strong Under", "Lean Under"):
                st.warning("⬇️ Strong UNDER" if label == "Strong Under" else "📉 Lean UNDER")
            else:
                st.info("No value on UNDER")

        st.divider()
        if label == "No Bet":
            st.info(
                f"⛔ No Bet — projection ({projection:.2f}) does not show sufficient edge "
                f"on either side. Model: {over_prob*100:.1f}% over probability."
            )

        if over_ev < 0 and under_ev < 0:
            st.caption(
                f"Both sides show negative EV — the {vig_pct:.1f}% book vig is too high "
                f"relative to the model's edge."
            )

        if season_starts < 10:
            st.warning(
                f"⚠️ **Low confidence prediction** — only {season_starts} games in sample. "
                f"Projections stabilize after ~15-20 games."
            )

        # ---- Season game log ----
        st.subheader("📅 Season Game Log")
        if prop_type == "Pitcher Strikeouts":
            all_game_dates = pd.Series(season_data["game_date"].unique()).sort_values()
            full_log       = season_by_game.reindex(all_game_dates, fill_value=0).reset_index()
        elif prop_type == "Batter Hits":
            pa_game_dates  = season_data[season_data["events"].notna()]["game_date"].unique()
            all_game_dates = pd.Series(pa_game_dates).sort_values()
            full_log       = season_by_game.reindex(all_game_dates, fill_value=0).reset_index()
        else:
            pa_game_dates  = season_data[season_data["events"].notna()]["game_date"].unique()
            all_game_dates = pd.Series(pa_game_dates).sort_values()
            full_log       = season_by_game.reindex(all_game_dates, fill_value=0).reset_index()

        full_log.columns  = ["game_date", stat_name]
        full_log["game_date"] = full_log["game_date"].astype(str)
        full_log["vs_line"]   = sportsbook_line

        st.caption(f"Showing all {len(full_log)} games — zeros mean player appeared but recorded 0 {stat_name.lower()}.")
        st.dataframe(full_log, use_container_width=True)
        st.bar_chart(full_log.set_index("game_date")[stat_name])