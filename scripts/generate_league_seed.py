import argparse
from pathlib import Path

from football_analytics.league_scope import load_distinct_leagues_from_response_file, write_dbt_seed


ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "dbt" / "seeds" / "senior_mens_international_leagues.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or inspect the senior men's league dbt seed.")
    parser.add_argument(
        "--inspect-api-response",
        type=Path,
        help="Path to a saved API-Football fixtures response JSON file to list distinct leagues.",
    )
    args = parser.parse_args()

    if args.inspect_api_response:
        leagues = load_distinct_leagues_from_response_file(args.inspect_api_response)
        print(f"Total distinct leagues: {len(leagues)}\n")
        for league_id, league_name in leagues:
            print(f"ID: {league_id:4d} | {league_name}")
    else:
        write_dbt_seed(SEED_PATH)
        print(f"Wrote {SEED_PATH}")
