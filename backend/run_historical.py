"""
Script to run the pipeline for historical date ranges.
Splits the date range into intervals and saves to DB with sequential dates.

Example:
    python run_historical.py --start-date 2025-01-01 --end-date 2025-01-21

This will:
    - Create 7 intervals of 3 days each
    - Save to database with dates from (today-6) to today
"""
import argparse
import sys
from datetime import datetime, timedelta

sys.path.insert(0, ".")

from app.tasks.pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pipeline for historical date range with interval splitting"
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="End date (YYYY-MM-DD format, default: today)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3,
        help="Interval size in days (default: 3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    else:
        end_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    interval_days = args.interval
    total_days = (end_date - start_date).days

    if total_days <= 0:
        print("Error: end-date must be after start-date")
        sys.exit(1)

    # Calculate number of intervals
    num_intervals = (total_days + interval_days - 1) // interval_days
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Total days: {total_days}")
    print(f"Interval size: {interval_days} days")
    print(f"Number of intervals: {num_intervals}")
    print()

    # Generate intervals and save dates
    today = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)

    for i in range(num_intervals):
        # Calculate interval date range (for Copernicus query)
        interval_start = start_date + timedelta(days=i * interval_days)
        interval_end = min(
            interval_start + timedelta(days=interval_days),
            end_date
        )

        # Calculate save date (sequential from today backwards)
        # Interval 0 -> today - (num_intervals - 1)
        # Interval 1 -> today - (num_intervals - 2)
        # ...
        # Last interval -> today
        save_date = today - timedelta(days=num_intervals - 1 - i)

        date_from_str = interval_start.strftime("%Y-%m-%dT%H:%M:%SZ")
        date_to_str = interval_end.strftime("%Y-%m-%dT%H:%M:%SZ")

        print(f"=" * 60)
        print(f"Interval {i + 1}/{num_intervals}")
        print(f"  Query range: {interval_start.date()} to {interval_end.date()}")
        print(f"  Save date:   {save_date.date()}")
        print(f"=" * 60)

        try:
            result = run_pipeline(
                date_from_override=date_from_str,
                date_to_override=date_to_str,
                save_date_override=save_date,
            )
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error in interval {i + 1}: {e}")
            continue

        print()

    print("Historical pipeline completed!")


if __name__ == "__main__":
    main()
