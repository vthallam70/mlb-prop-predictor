"""
Backtest the pitcher-strikeouts projection in app.py against actual K counts.

What it does
------------
For each completed game in the requested date range:
  1. Finds both probable starters from MLB Stats API.
  2. Pretends "today" is that game's date (via _backtest_harness.as_of).
  3. Runs `compute_pitcher_k_projection` for each starter.
  4. Pulls the pitcher's actual K count for that date via statcast_pitcher.
  5. Logs projection vs actual.

Outputs
-------
  - CSV: one row per (game, starter) with projection + actual.
  - Stdout summary: MAE, RMSE, mean bias, over/under hit rate at projection-as-line,
    plus a "within ±1 K" hit rate.

Honest caveats
--------------
The K projection only fires if the model has ≥5 prior starts that season for
the pitcher (and ≥3 in the recent window for the recent component to kick in).
April-of-season starts are therefore mostly skipped. Disclose alongside numbers.

Some inputs (FanGraphs SwStr%, MLB Stats API season stats) are season-
cumulative; see the harness file for the leakage note.

Usage
-----
  python backtest_pitcher_ks.py
  python backtest_pitcher_ks.py --start 2025-05-01 --end 2025-05-31
  python backtest_pitcher_ks.py --out k_results.csv --max-starts 50
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict

import _backtest_harness as bh
from _backtest_harness import app, as_of, daterange, default_date_ranges, fetch_schedule, game_is_final


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", help="YYYY-MM-DD (overrides default ranges)")
    p.add_argument("--end",   help="YYYY-MM-DD (overrides default ranges)")
    p.add_argument("--out",   default="backtest_pitcher_ks_results.csv",
                   help="CSV output path (default: backtest_pitcher_ks_results.csv)")
    p.add_argument("--max-starts", type=int, default=None,
                   help="Stop after this many starts (handy for smoke-testing)")
    p.add_argument("--verbose", action="store_true", help="Print per-start lines")
    return p.parse_args()


def resolve_ranges(args) -> list[tuple[str, str]]:
    if args.start and args.end:
        return [(args.start, args.end)]
    if args.start or args.end:
        sys.exit("--start and --end must be used together (or omit both for defaults)")
    return default_date_ranges()


def actual_ks_on(date_str: str, player_id: int) -> int | None:
    """Pull actual strikeouts for `player_id` on `date_str` via statcast."""
    try:
        data = app.statcast_pitcher(date_str, date_str, player_id)
    except Exception:
        return None
    if data is None or data.empty:
        return None
    return int((data["events"] == "strikeout").sum())


def project_for(date_str: str, player_id: int, opponent_abbrev: str,
                park_abbrev: str, umpire_name: str | None) -> dict | None:
    """Run compute_pitcher_k_projection with as_of(date_str)."""
    with as_of(date_str):
        try:
            return app.compute_pitcher_k_projection(
                player_id, opponent_abbrev,
                home_park_abbrev=park_abbrev,
                umpire_name=umpire_name,
            )
        except Exception:
            return None


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        print("No scored starts in range.")
        return

    errors  = [r["projection"] - r["actual"]    for r in rows]
    abs_err = [abs(e)                            for e in errors]
    sq_err  = [e * e                             for e in errors]

    mae  = sum(abs_err) / n
    rmse = math.sqrt(sum(sq_err) / n)
    bias = sum(errors) / n

    within_1 = sum(1 for e in abs_err if e <= 1.0)
    within_2 = sum(1 for e in abs_err if e <= 2.0)

    # Treat the projection (rounded to nearest 0.5) as the line, then check
    # whether the actual is over/under. The model "picks" Over if projection >
    # line+0.0 ... since they're equal here we score the larger side.
    # More useful: compute hit rate using projection as the directional pick
    # against a notional half-step line, with the model's pick = Over.
    notional_correct = sum(1 for r in rows if r["actual"] > r["projection"])

    print("\n" + "=" * 60)
    print("PITCHER STRIKEOUTS BACKTEST SUMMARY")
    print("=" * 60)
    print(f"Starts scored:      {n}")
    print(f"MAE:                {mae:.3f}  (avg absolute K error per start)")
    print(f"RMSE:               {rmse:.3f}")
    print(f"Mean bias:          {bias:+.3f}  (positive = projection too high)")
    print(f"Within ±1 K:        {within_1}/{n}  ({within_1 / n * 100:.1f}%)")
    print(f"Within ±2 K:        {within_2}/{n}  ({within_2 / n * 100:.1f}%)")
    print(f"Actual > projection: {notional_correct}/{n} "
          f"({notional_correct / n * 100:.1f}%)  "
          "(near 50% = projection well-centered)")

    # Per-month MAE
    by_month: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_month[r["date"][:7]].append(abs(r["projection"] - r["actual"]))
    print("\nMAE by month:")
    print(f"  {'month':<8} {'n':>5} {'MAE':>7}")
    for m in sorted(by_month):
        vals = by_month[m]
        print(f"  {m}   {len(vals):>5} {sum(vals) / len(vals):>7.3f}")

    print("\nReminder: starts where the pitcher had <5 prior starts that season")
    print("were skipped (model insufficient-data guard). Disclose alongside numbers.")


def main() -> None:
    args = parse_args()
    ranges = resolve_ranges(args)
    max_starts = args.max_starts

    rows: list[dict] = []
    starts_done = 0
    skipped_insufficient = 0
    skipped_no_actuals = 0

    with open(args.out, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "date", "pitcher", "player_id", "team", "opponent",
            "projection", "actual", "abs_error",
            "season_avg", "recent_avg", "season_starts",
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

                day_n = 0
                day_abs_err = 0.0

                for g in games:
                    if max_starts is not None and starts_done >= max_starts:
                        break
                    if not game_is_final(g):
                        continue

                    home_id = g["teams"]["home"]["team"]["id"]
                    away_id = g["teams"]["away"]["team"]["id"]
                    home_abbrev = app.ID_TEAM_MAP.get(home_id)
                    away_abbrev = app.ID_TEAM_MAP.get(away_id)
                    if not home_abbrev or not away_abbrev:
                        continue

                    umpire = None
                    for off in g.get("officials", []):
                        if off.get("officialType") == "Home Plate":
                            umpire = off["official"].get("fullName")
                            break

                    for side in ("home", "away"):
                        if max_starts is not None and starts_done >= max_starts:
                            break

                        pitcher = g["teams"][side].get("probablePitcher") or {}
                        pid = pitcher.get("id")
                        pname = pitcher.get("fullName", "?")
                        if not pid:
                            continue

                        team_abbrev = home_abbrev if side == "home" else away_abbrev
                        opp_abbrev  = away_abbrev if side == "home" else home_abbrev

                        result = project_for(
                            date_str, int(pid), opp_abbrev,
                            park_abbrev=home_abbrev, umpire_name=umpire,
                        )
                        if result is None or "error" in result:
                            skipped_insufficient += 1
                            if args.verbose:
                                err = (result or {}).get("error", "no result")
                                print(f"  {date_str} {pname}: skipped ({err})")
                            continue

                        actual = actual_ks_on(date_str, int(pid))
                        if actual is None:
                            skipped_no_actuals += 1
                            continue

                        projection = float(result["projection"])
                        row = {
                            "date": date_str,
                            "pitcher": pname,
                            "player_id": int(pid),
                            "team": team_abbrev,
                            "opponent": opp_abbrev,
                            "projection": round(projection, 3),
                            "actual": actual,
                            "abs_error": round(abs(projection - actual), 3),
                            "season_avg": round(float(result.get("season_avg") or 0), 3),
                            "recent_avg": (
                                round(float(result["recent_avg"]), 3)
                                if result.get("recent_avg") is not None
                                and not (isinstance(result["recent_avg"], float)
                                         and math.isnan(result["recent_avg"]))
                                else ""
                            ),
                            "season_starts": result.get("season_starts", ""),
                        }
                        writer.writerow(row)
                        fp.flush()
                        rows.append(row)
                        starts_done += 1
                        day_n += 1
                        day_abs_err += abs(projection - actual)

                        if args.verbose:
                            print(f"  {date_str} {pname:<25} "
                                  f"proj {projection:.2f}  actual {actual:>2}  "
                                  f"err {projection - actual:+.2f}")

                if day_n:
                    print(f"  {date_str}: {day_n} starts, MAE {day_abs_err / day_n:.2f}")

                if max_starts is not None and starts_done >= max_starts:
                    print(f"\nHit --max-starts={max_starts}; stopping.")
                    break
            if max_starts is not None and starts_done >= max_starts:
                break

    print(f"\nSkipped (insufficient data / model error): {skipped_insufficient}")
    print(f"Skipped (no statcast actuals returned):    {skipped_no_actuals}")
    summarize(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
