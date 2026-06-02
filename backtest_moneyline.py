"""
Backtest the moneyline model in app.py against historical MLB outcomes.

What it does
------------
For each completed game in the requested date range:
  1. Pretends "today" is that game's date (via _backtest_harness.as_of).
  2. Runs `team_moneyline_probability` for both sides at -110 odds.
  3. Picks the team with the higher model probability.
  4. Compares the pick to the actual winner (from MLB Stats API).
  5. Logs predicted_prob, picked team, actual winner, and correct/incorrect.

Outputs
-------
  - CSV: every game with its row of predictions/outcomes.
  - Stdout summary: total games, win-pick accuracy, Brier score, log loss,
    and an accuracy-by-confidence-bucket calibration table.

Honest caveats
--------------
The model has minor look-ahead from season-cumulative API calls
(FanGraphs FIP/wRC+, MLB Stats API team season stats, Baseball-Reference
schedules) — see the note at the top of _backtest_harness.py. Disclose this
alongside any quoted accuracy.

Usage
-----
  python backtest_moneyline.py
  python backtest_moneyline.py --start 2025-05-01 --end 2025-05-31
  python backtest_moneyline.py --out results.csv --max-games 100
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime

import _backtest_harness as bh
from _backtest_harness import app, as_of, daterange, default_date_ranges, fetch_schedule, winner_abbrev


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", help="YYYY-MM-DD (overrides default ranges)")
    p.add_argument("--end",   help="YYYY-MM-DD (overrides default ranges)")
    p.add_argument("--out",   default="backtest_moneyline_results.csv",
                   help="CSV output path (default: backtest_moneyline_results.csv)")
    p.add_argument("--max-games", type=int, default=None,
                   help="Stop after this many games (handy for smoke-testing)")
    p.add_argument("--verbose", action="store_true", help="Print per-game lines")
    return p.parse_args()


def resolve_ranges(args) -> list[tuple[str, str]]:
    if args.start and args.end:
        return [(args.start, args.end)]
    if args.start or args.end:
        sys.exit("--start and --end must be used together (or omit both for defaults)")
    return default_date_ranges()


def predict_game(home_abbrev: str, away_abbrev: str) -> tuple[float, float] | None:
    """Returns (home_model_prob, away_model_prob) at -110 odds, or None on error."""
    try:
        home_pred = app.team_moneyline_probability(
            home_abbrev, away_abbrev, american_odds_input=-110,
        )
        away_pred = app.team_moneyline_probability(
            away_abbrev, home_abbrev, american_odds_input=-110,
        )
    except Exception:
        return None
    return float(home_pred["model_prob"]), float(away_pred["model_prob"])


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        print("No completed games in range.")
        return

    correct = sum(1 for r in rows if r["correct"])
    brier   = sum((r["pick_prob"] - (1 if r["correct"] else 0)) ** 2 for r in rows) / n

    log_loss_terms = []
    for r in rows:
        p = max(min(r["pick_prob"], 1 - 1e-6), 1e-6)
        y = 1 if r["correct"] else 0
        log_loss_terms.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    log_loss = sum(log_loss_terms) / n

    print("\n" + "=" * 60)
    print("MONEYLINE BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Games scored:       {n}")
    print(f"Correct picks:      {correct}  ({correct / n * 100:.2f}%)")
    print(f"Brier score:        {brier:.4f}  (lower is better; coin flip ≈ 0.25)")
    print(f"Log loss:           {log_loss:.4f} (lower is better; coin flip ≈ 0.693)")

    # Accuracy by confidence bucket — does picking only "high-confidence" games help?
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]
    print("\nAccuracy by model confidence on the pick:")
    print(f"  {'bucket':<14} {'n':>5} {'win %':>8}")
    for lo, hi in buckets:
        b = [r for r in rows if lo <= r["pick_prob"] < hi]
        if not b:
            continue
        bc = sum(1 for r in b if r["correct"])
        print(f"  {lo:.2f}–{hi:.2f}     {len(b):>5} {bc / len(b) * 100:>7.2f}%")

    # Per-month accuracy — helps see drift across the season.
    by_month: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_month[r["date"][:7]].append(r)
    print("\nAccuracy by month:")
    print(f"  {'month':<8} {'n':>5} {'win %':>8}")
    for month in sorted(by_month):
        b = by_month[month]
        bc = sum(1 for r in b if r["correct"])
        print(f"  {month}   {len(b):>5} {bc / len(b) * 100:>7.2f}%")

    print("\nReminder: the model has minor look-ahead from season-cumulative")
    print("data sources (FanGraphs, MLB Stats API team season stats, B-Ref).")
    print("Disclose alongside any quoted accuracy.")


def main() -> None:
    args = parse_args()
    ranges = resolve_ranges(args)
    max_games = args.max_games

    rows: list[dict] = []
    games_done = 0

    with open(args.out, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "date", "away", "home", "picked", "pick_prob",
            "home_prob", "away_prob", "actual_winner", "correct",
        ])
        writer.writeheader()

        for r_start, r_end in ranges:
            print(f"\n--- Range {r_start} → {r_end} ---")
            for date_str in daterange(r_start, r_end):
                try:
                    games = fetch_schedule(date_str)
                except Exception as e:
                    print(f"  {date_str}: schedule fetch failed ({e}); skipping day")
                    continue

                day_games = 0
                day_correct = 0

                for g in games:
                    if max_games is not None and games_done >= max_games:
                        break

                    actual = winner_abbrev(g)
                    if actual is None:
                        continue  # not final, tied, or unknown abbrev

                    home_id = g["teams"]["home"]["team"]["id"]
                    away_id = g["teams"]["away"]["team"]["id"]
                    home_abbrev = app.ID_TEAM_MAP.get(home_id)
                    away_abbrev = app.ID_TEAM_MAP.get(away_id)
                    if not home_abbrev or not away_abbrev:
                        continue

                    with as_of(date_str):
                        probs = predict_game(home_abbrev, away_abbrev)

                    if probs is None:
                        if args.verbose:
                            print(f"  {date_str} {away_abbrev}@{home_abbrev}: model error")
                        continue

                    home_prob, away_prob = probs
                    if home_prob >= away_prob:
                        picked, pick_prob = home_abbrev, home_prob
                    else:
                        picked, pick_prob = away_abbrev, away_prob
                    correct = picked == actual

                    row = {
                        "date": date_str,
                        "away": away_abbrev,
                        "home": home_abbrev,
                        "picked": picked,
                        "pick_prob": round(pick_prob, 4),
                        "home_prob": round(home_prob, 4),
                        "away_prob": round(away_prob, 4),
                        "actual_winner": actual,
                        "correct": int(correct),
                    }
                    writer.writerow(row)
                    fp.flush()
                    rows.append(row)
                    games_done += 1
                    day_games += 1
                    day_correct += int(correct)

                    if args.verbose:
                        mark = "✓" if correct else "✗"
                        print(f"  {date_str} {away_abbrev}@{home_abbrev}: "
                              f"picked {picked} ({pick_prob:.2%}), actual {actual} {mark}")

                if day_games:
                    print(f"  {date_str}: {day_correct}/{day_games} "
                          f"({day_correct / day_games * 100:.1f}%)")

                if max_games is not None and games_done >= max_games:
                    print(f"\nHit --max-games={max_games}; stopping.")
                    break
            if max_games is not None and games_done >= max_games:
                break

    summarize(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
