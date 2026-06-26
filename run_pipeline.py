import argparse
import sys
from football_analytics.pipeline import FootballDataPipeline
from football_analytics.orchestrator import run_matchup_pipeline, poll_and_orchestrate

def main():
    parser = argparse.ArgumentParser(
        description="World Cup 2026 Syndicate Alpha Engine Pipeline",
        epilog=(
            'Example: python run_pipeline.py --home "Curaçao" '
            '--away "Ivory Coast" --date 2026-06-25'
        ),
    )
    parser.add_argument("--home", type=str, default="Germany", help="Home team name")
    parser.add_argument("--away", type=str, default="Ecuador", help="Away team name")
    fixture_group = parser.add_mutually_exclusive_group(required=True)
    fixture_group.add_argument(
        "--fixture-id",
        type=int,
        help="Exact API fixture ID",
    )
    fixture_group.add_argument(
        "--date",
        type=str,
        help="Fixture date in UTC using YYYY-MM-DD",
    )
    parser.add_argument("--sims", type=int, default=10000, help="Number of Monte Carlo simulations to run")
    parser.add_argument("--offline", action="store_true", default=False, help="Run offline using local cached JSON files")
    parser.add_argument("--poll", action="store_true", default=False, help="Poll for confirmed lineups at T-60 before kickoff")
    parser.add_argument("--db-path", type=str, default="world_cup_2026_completed_data.json", help="Path to tournament match database")
    parser.add_argument("--api-key", type=str, default=None, help="Football API-Sports API Key")
    
    args = parser.parse_args()
    
    try:
        fixture_id = args.fixture_id
        home_team = args.home
        away_team = args.away

        if args.date:
            resolver = FootballDataPipeline(
                api_key=args.api_key,
                cache_file=args.db_path,
                offline=args.offline,
            )
            resolved = resolver.resolve_fixture_by_date(
                home_team,
                away_team,
                args.date,
            )
            fixture_id = resolved["fixture_id"]
            home_team = resolved["home_team"]
            away_team = resolved["away_team"]
            print(
                "Resolved UTC fixture: "
                f"{home_team} vs {away_team} | "
                f"fixture {fixture_id} | {resolved['fixture_date']}"
            )

        if args.poll:
            poll_and_orchestrate(
                home_team=home_team,
                away_team=away_team,
                fixture_id=fixture_id,
                offline=args.offline,
                api_key=args.api_key,
                sims=args.sims,
                db_path=args.db_path
            )
        else:
            run_matchup_pipeline(
                home_team=home_team,
                away_team=away_team,
                fixture_id=fixture_id,
                offline=args.offline,
                api_key=args.api_key,
                sims=args.sims,
                db_path=args.db_path
            )
    except Exception as e:
        print(f"\nPipeline Execution Failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
