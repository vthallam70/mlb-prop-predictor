"""
A/B backtest: v3 baseline (no heat) vs v4 (team heat + player heat + upset boost).

How it works
------------
For each completed game in the requested date range:
  1. Pretends "today" is that game's date.
  2. Runs `team_moneyline_probability` for both sides at -110 odds, twice:
       - once with app.HEAT_FACTORS_ENABLED = False  (v3 baseline)
       - once with app.HEAT_FACTORS_ENABLED = True   (v4 heat-on)
  3. Picks the higher-probability side from each model.
  4. Compares both picks against the actual winner.

Outputs
-------
  - CSV: per-game rows with both models' picks/probs/outcomes.
  - Stdout: side-by-side accuracy, Brier, log loss, ROI at -110, calibration
    by confidence bucket, and a summary of "disagreement" games where the two
    models picked different sides.

Usage
-----
  python backtest_compare_heat.py
  python backtest_compare_heat.py --start 2025-05-01 --end 2025-05-31
  python backtest_compare_heat.py --max-games 100 --verbose

Honest caveats
--------------
Same look-ahead caveats as backtest_moneyline.py — season-cumulative stats
(FanGraphs/MLB team stats/B-Ref) don't expose an as-of filter. Statcast and
recent-form windows DO respect the date. So the heat factors (which are
mostly Statcast + B-Ref) are evaluated more faithfully than the legacy v3
factors, but neither is perfectly leak-free.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from contextlib import contextmanager

import _backtest_harness as bh
from _backtest_harness import (
    app, as_of, daterange, default_date_ranges, fetch_schedule,
    preload_statcast, winner_abbrev,
)


@contextmanager
def heat_enabled(flag: bool):
    saved = app.HEAT_FACTORS_ENABLED
    app.HEAT_FACTORS_ENABLED = flag
    try:
        yield
    finally:
        app.HEAT_FACTORS_ENABLED = saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", help="YYYY-MM-DD (overrides default ranges)")
    p.add_argument("--end",   help="YYYY-MM-DD (overrides default ranges)")
    p.add_argument("--out",   default="backtest_compare_heat_results.csv",
                   help="CSV output path")
    p.add_argument("--max-games", type=int, default=None,
                   help="Stop after this many games (handy for smoke-testing)")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-game lines (only disagreements unless --all)")
    p.add_argument("--all", action="store_true",
                   help="With --verbose, print every game, not just disagreements")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel processes to split the date range across. "
                        "Each worker preloads its own statcast (RAM ~500MB-1GB per worker).")
    return p.parse_args()


def split_date_range(start: str, end: str, n: int) -> list[tuple[str, str]]:
    """
    Split [start, end] into n contiguous date chunks. Later chunks get FEWER
    days to roughly balance wall-clock — workers covering later dates have
    bigger preload windows (season runs March 27 → their end_date).
    """
    from datetime import datetime, timedelta
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    total_days = (e - s).days + 1
    if n <= 1 or total_days <= n:
        return [(start, end)]

    # Weight by sqrt to give later chunks (heavier preload) fewer days.
    # Approximation: total work for a chunk ≈ (days_in_chunk × avg_window_size).
    # Window grows linearly with date, so use 1/sqrt(rank) weighting.
    import math
    weights = [1.0 / math.sqrt(i + 1) for i in range(n)]
    total_w = sum(weights)
    fractions = [w / total_w for w in weights]
    day_counts = [max(1, int(round(f * total_days))) for f in fractions]
    # Fix off-by-one from rounding
    while sum(day_counts) > total_days:
        day_counts[day_counts.index(max(day_counts))] -= 1
    while sum(day_counts) < total_days:
        day_counts[day_counts.index(min(day_counts))] += 1

    chunks: list[tuple[str, str]] = []
    cursor = s
    for dc in day_counts:
        chunk_end = cursor + timedelta(days=dc - 1)
        chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def run_parallel(args: argparse.Namespace) -> None:
    """Spawn N subprocesses, each running its chunk; concatenate CSVs at end."""
    import os
    import subprocess

    if not (args.start and args.end):
        sys.exit("--workers requires --start and --end (no default ranges in parallel mode)")

    chunks = split_date_range(args.start, args.end, args.workers)
    print(f"Splitting {args.start} → {args.end} across {len(chunks)} workers:")
    for i, (cs, ce) in enumerate(chunks):
        print(f"  worker {i}: {cs} → {ce}")

    base, ext = os.path.splitext(args.out)
    part_paths = [f"{base}.part{i}{ext}" for i in range(len(chunks))]

    procs = []
    for i, (cs, ce) in enumerate(chunks):
        cmd = [
            sys.executable, sys.argv[0],
            "--start", cs, "--end", ce,
            "--out", part_paths[i],
            "--workers", "1",  # subprocess runs single-threaded
        ]
        if args.verbose:
            cmd.append("--verbose")
        if args.all:
            cmd.append("--all")
        log_path = f"{base}.part{i}.log"
        log_fp = open(log_path, "w")
        procs.append((i, subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT), log_fp, log_path))
        print(f"  worker {i} spawned (pid={procs[-1][1].pid}), logging to {log_path}")

    # Wait for all workers. Polite Ctrl-C handling propagates SIGINT to children.
    try:
        for i, p, fp, _ in procs:
            rc = p.wait()
            fp.close()
            print(f"  worker {i} done (exit {rc})")
    except KeyboardInterrupt:
        print("\nInterrupt — killing workers…")
        for _, p, _, _ in procs:
            p.terminate()
        for _, p, _, _ in procs:
            p.wait()
        sys.exit(130)

    # Concatenate part CSVs into final --out, write header once.
    print(f"\nMerging {len(part_paths)} part files into {args.out}…")
    rows: list[dict] = []
    header_written = False
    with open(args.out, "w", newline="") as out_fp:
        writer = None
        for path in part_paths:
            if not os.path.exists(path):
                print(f"  warning: {path} missing — skipping")
                continue
            with open(path, "r") as in_fp:
                reader = csv.DictReader(in_fp)
                for row in reader:
                    if not header_written:
                        writer = csv.DictWriter(out_fp, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        header_written = True
                    writer.writerow(row)
                    # Normalize types for the summary block
                    for k in ("v3_pick_prob", "v3_home_prob", "v3_away_prob",
                              "v4_pick_prob", "v4_home_prob", "v4_away_prob"):
                        if k in row:
                            row[k] = float(row[k])
                    for k in ("v3_correct", "v3_picked_was_dog",
                              "v4_correct", "v4_picked_was_dog", "disagreement"):
                        if k in row:
                            row[k] = int(row[k])
                    rows.append(row)

    print(f"Merged {len(rows)} rows to {args.out}")
    print_compare(rows)

    # Clean up part files
    for path in part_paths:
        try:
            os.remove(path)
        except OSError:
            pass


def resolve_ranges(args) -> list[tuple[str, str]]:
    if args.start and args.end:
        return [(args.start, args.end)]
    if args.start or args.end:
        sys.exit("--start and --end must be used together (or omit both for defaults)")
    return default_date_ranges()


def predict_both_sides(home_abbrev: str, away_abbrev: str) -> tuple[float, float] | None:
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


def profit_at_minus_110(correct: bool) -> float:
    """Unit-stake profit at -110: win returns +0.9091, loss returns -1.0."""
    return 100.0 / 110.0 if correct else -1.0


def summarize_one(name: str, rows: list[dict], prob_key: str, correct_key: str) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    correct = sum(1 for r in rows if r[correct_key])
    brier = sum((r[prob_key] - (1 if r[correct_key] else 0)) ** 2 for r in rows) / n
    log_loss_terms = []
    for r in rows:
        p = max(min(r[prob_key], 1 - 1e-6), 1e-6)
        y = 1 if r[correct_key] else 0
        log_loss_terms.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
    log_loss = sum(log_loss_terms) / n
    profit = sum(profit_at_minus_110(bool(r[correct_key])) for r in rows)
    roi = profit / n * 100  # units per game, times 100 for percent-of-stake terms
    return {
        "name": name,
        "n": n,
        "acc": correct / n,
        "brier": brier,
        "log_loss": log_loss,
        "roi_pct": roi,
    }


def print_compare(rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        print("No completed games in range.")
        return

    v3 = summarize_one("v3 baseline", rows, "v3_pick_prob", "v3_correct")
    v4 = summarize_one("v4 heat-on", rows, "v4_pick_prob", "v4_correct")

    print("\n" + "=" * 72)
    print("MONEYLINE A/B BACKTEST — v3 baseline vs v4 heat-on")
    print("=" * 72)
    print(f"Games scored: {n}")
    print()
    print(f"{'Metric':<20} {'v3 baseline':>14} {'v4 heat-on':>14} {'Δ':>10}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {v3['acc']*100:>13.2f}% {v4['acc']*100:>13.2f}% "
          f"{(v4['acc']-v3['acc'])*100:>+9.2f}%")
    print(f"{'Brier (lower=better)':<20} {v3['brier']:>14.4f} {v4['brier']:>14.4f} "
          f"{v4['brier']-v3['brier']:>+10.4f}")
    print(f"{'Log loss':<20} {v3['log_loss']:>14.4f} {v4['log_loss']:>14.4f} "
          f"{v4['log_loss']-v3['log_loss']:>+10.4f}")
    print(f"{'ROI @ -110 per bet':<20} {v3['roi_pct']:>13.2f}% {v4['roi_pct']:>13.2f}% "
          f"{v4['roi_pct']-v3['roi_pct']:>+9.2f}%")

    # Disagreement analysis: where they pick different sides
    dis = [r for r in rows if r["v3_picked"] != r["v4_picked"]]
    if dis:
        v3_dis_correct = sum(1 for r in dis if r["v3_correct"])
        v4_dis_correct = sum(1 for r in dis if r["v4_correct"])
        print()
        print(f"Disagreements: {len(dis)} games ({len(dis)/n*100:.1f}% of slate)")
        print(f"  v3 went {v3_dis_correct}/{len(dis)} ({v3_dis_correct/len(dis)*100:.1f}%) "
              f"on disagreement games")
        print(f"  v4 went {v4_dis_correct}/{len(dis)} ({v4_dis_correct/len(dis)*100:.1f}%) "
              f"on disagreement games")
        # Did the heat-on flip toward the underdog?
        v4_picked_dog = sum(1 for r in dis if r["v4_picked_was_dog"])
        v4_dog_hits   = sum(1 for r in dis
                            if r["v4_picked_was_dog"] and r["v4_correct"])
        if v4_picked_dog:
            print(f"  v4 picked the dog in {v4_picked_dog} disagreements, "
                  f"hit {v4_dog_hits} ({v4_dog_hits/v4_picked_dog*100:.1f}%)")
    else:
        print("\nNo disagreements — heat factors never flipped the pick.")

    # Calibration buckets — both models, side by side
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.01)]
    print("\nAccuracy by model confidence (v3 / v4):")
    print(f"  {'bucket':<14} {'v3 n':>5} {'v3 win%':>8}   {'v4 n':>5} {'v4 win%':>8}")
    for lo, hi in buckets:
        v3b = [r for r in rows if lo <= r["v3_pick_prob"] < hi]
        v4b = [r for r in rows if lo <= r["v4_pick_prob"] < hi]
        if not v3b and not v4b:
            continue
        v3c = sum(1 for r in v3b if r["v3_correct"]) if v3b else 0
        v4c = sum(1 for r in v4b if r["v4_correct"]) if v4b else 0
        v3_str = f"{v3c/len(v3b)*100:>7.2f}%" if v3b else "      —"
        v4_str = f"{v4c/len(v4b)*100:>7.2f}%" if v4b else "      —"
        print(f"  {lo:.2f}–{hi:.2f}     {len(v3b):>5} {v3_str}    {len(v4b):>5} {v4_str}")

    # Underdog performance — heat is supposed to help on dogs specifically
    dogs_v4 = [r for r in rows if r["v4_picked_was_dog"]]
    dogs_v3 = [r for r in rows if r["v3_picked_was_dog"]]
    if dogs_v3 or dogs_v4:
        print("\nUnderdog picks specifically:")
        if dogs_v3:
            c = sum(1 for r in dogs_v3 if r["v3_correct"])
            print(f"  v3 picked dogs: {len(dogs_v3)} times, hit {c} ({c/len(dogs_v3)*100:.1f}%)")
        if dogs_v4:
            c = sum(1 for r in dogs_v4 if r["v4_correct"])
            print(f"  v4 picked dogs: {len(dogs_v4)} times, hit {c} ({c/len(dogs_v4)*100:.1f}%)")

    print("\nReminder: minor look-ahead from season-cumulative stats (FanGraphs,"
          "\nMLB Stats API team stats, B-Ref). Disclose alongside any quoted numbers.")


def main() -> None:
    args = parse_args()

    if args.workers > 1:
        run_parallel(args)
        return

    ranges = resolve_ranges(args)
    max_games = args.max_games

    rows: list[dict] = []
    games_done = 0

    fieldnames = [
        "date", "away", "home", "actual_winner",
        "v3_picked", "v3_pick_prob", "v3_home_prob", "v3_away_prob",
        "v3_correct", "v3_picked_was_dog",
        "v4_picked", "v4_pick_prob", "v4_home_prob", "v4_away_prob",
        "v4_correct", "v4_picked_was_dog",
        "disagreement",
    ]

    with open(args.out, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()

        for r_start, r_end in ranges:
            print(f"\n--- Range {r_start} → {r_end} ---")
            year = int(r_start[:4])
            preload_statcast(year, end_date=r_end)
            for date_str in daterange(r_start, r_end):
                try:
                    games = fetch_schedule(date_str)
                except Exception as e:
                    print(f"  {date_str}: schedule fetch failed ({e}); skipping day")
                    continue

                day_v3 = day_v4 = day_games = 0

                for g in games:
                    if max_games is not None and games_done >= max_games:
                        break

                    actual = winner_abbrev(g)
                    if actual is None:
                        continue

                    home_id = g["teams"]["home"]["team"]["id"]
                    away_id = g["teams"]["away"]["team"]["id"]
                    home_abbrev = app.ID_TEAM_MAP.get(home_id)
                    away_abbrev = app.ID_TEAM_MAP.get(away_id)
                    if not home_abbrev or not away_abbrev:
                        continue

                    # Both models share the same `as_of` date — same caches,
                    # same look-ahead, just the heat factor flag flips.
                    with as_of(date_str):
                        with heat_enabled(False):
                            v3_probs = predict_both_sides(home_abbrev, away_abbrev)
                        with heat_enabled(True):
                            v4_probs = predict_both_sides(home_abbrev, away_abbrev)

                    if v3_probs is None or v4_probs is None:
                        if args.verbose:
                            print(f"  {date_str} {away_abbrev}@{home_abbrev}: model error")
                        continue

                    v3_home, v3_away = v3_probs
                    v4_home, v4_away = v4_probs

                    # Without historical odds data, both sides are scored at
                    # -110 (book_implied=0.5238 each). The model's internal
                    # `is_dog` check (book_implied<0.50) never fires, so the
                    # upset_boost path is NOT exercised by this backtest —
                    # only team_heat + player_heat adjustments are.
                    # For the "picked a dog" stat, we treat the lower v3-prob
                    # side as the implied dog (model's own consensus pre-heat).
                    home_is_dog = v3_home <= v3_away

                    if v3_home >= v3_away:
                        v3_picked, v3_pick_prob = home_abbrev, v3_home
                        v3_picked_was_dog = home_is_dog
                    else:
                        v3_picked, v3_pick_prob = away_abbrev, v3_away
                        v3_picked_was_dog = not home_is_dog

                    if v4_home >= v4_away:
                        v4_picked, v4_pick_prob = home_abbrev, v4_home
                        v4_picked_was_dog = home_is_dog
                    else:
                        v4_picked, v4_pick_prob = away_abbrev, v4_away
                        v4_picked_was_dog = not home_is_dog

                    v3_correct = (v3_picked == actual)
                    v4_correct = (v4_picked == actual)
                    disagreement = v3_picked != v4_picked

                    row = {
                        "date": date_str,
                        "away": away_abbrev,
                        "home": home_abbrev,
                        "actual_winner": actual,
                        "v3_picked": v3_picked,
                        "v3_pick_prob": round(v3_pick_prob, 4),
                        "v3_home_prob": round(v3_home, 4),
                        "v3_away_prob": round(v3_away, 4),
                        "v3_correct": int(v3_correct),
                        "v3_picked_was_dog": int(v3_picked_was_dog),
                        "v4_picked": v4_picked,
                        "v4_pick_prob": round(v4_pick_prob, 4),
                        "v4_home_prob": round(v4_home, 4),
                        "v4_away_prob": round(v4_away, 4),
                        "v4_correct": int(v4_correct),
                        "v4_picked_was_dog": int(v4_picked_was_dog),
                        "disagreement": int(disagreement),
                    }
                    writer.writerow(row)
                    fp.flush()
                    rows.append(row)
                    games_done += 1
                    day_games += 1
                    day_v3 += int(v3_correct)
                    day_v4 += int(v4_correct)

                    if args.verbose and (args.all or disagreement):
                        v3m = "✓" if v3_correct else "✗"
                        v4m = "✓" if v4_correct else "✗"
                        tag = "  ⚡FLIP" if disagreement else ""
                        print(f"  {date_str} {away_abbrev}@{home_abbrev}: "
                              f"actual={actual}  v3={v3_picked}({v3_pick_prob:.0%}){v3m}  "
                              f"v4={v4_picked}({v4_pick_prob:.0%}){v4m}{tag}")

                if day_games:
                    print(f"  {date_str}: v3 {day_v3}/{day_games} "
                          f"({day_v3/day_games*100:.1f}%)  ·  "
                          f"v4 {day_v4}/{day_games} ({day_v4/day_games*100:.1f}%)")

                if max_games is not None and games_done >= max_games:
                    print(f"\nHit --max-games={max_games}; stopping.")
                    break
            if max_games is not None and games_done >= max_games:
                break

    print_compare(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
