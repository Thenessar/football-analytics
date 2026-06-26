import os
import time
import datetime
import re
import unicodedata
import pandas as pd
from typing import Dict, List, Optional
from football_analytics.pipeline import FootballDataPipeline, ConfirmedLineupDataError
from football_analytics.modeling import (
    calculate_common_opponent_modifiers,
    run_squad_simulation,
    normalize_player_name,
)
from football_analytics.reporting import generate_html_report, compile_pdf_report

# Explicit sample lineups used only when --offline is present.
OFFLINE_SAMPLE_LINEUPS = {
    "Germany": [
        {"name": "Oliver Baumann", "position": "G"},
        {"name": "Joshua Kimmich", "position": "D"},
        {"name": "Malick Thiaw", "position": "D"},
        {"name": "Antonio Rüdiger", "position": "D"},
        {"name": "David Raum", "position": "D"},
        {"name": "Leon Goretzka", "position": "M"},
        {"name": "Angelo Stiller", "position": "M"},
        {"name": "Jamie Leweling", "position": "M"},
        {"name": "Nadiem Amiri", "position": "M"},
        {"name": "Maximilian Beier", "position": "M"},
        {"name": "Deniz Undav", "position": "F"}
    ],
    "Ecuador": [
        {"name": "Hernán Galíndez", "position": "G"},
        {"name": "Piero Hincapié", "position": "D"},
        {"name": "Willian Pacho", "position": "D"},
        {"name": "Joel Ordóñez", "position": "D"},
        {"name": "Pervis Estupiñán", "position": "M"},
        {"name": "Moisés Caicedo", "position": "M"},
        {"name": "Alan Franco", "position": "M"},
        {"name": "Ángelo Preciado", "position": "M"},
        {"name": "Gonzalo Plata", "position": "F"},
        {"name": "John Yeboah", "position": "F"},
        {"name": "Enner Valencia", "position": "F"}
    ]
}


def _filename_slug(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")


def _is_complete_starting_xi(lineup: List[Dict]) -> bool:
    """Requires exactly 11 uniquely named starters before a lineup is trusted."""
    if len(lineup) != 11:
        return False
    normalized_names = [normalize_player_name(player.get("name")) for player in lineup]
    return all(normalized_names) and len(set(normalized_names)) == 11


def _offline_sample_lineup(team_name: str) -> List[Dict]:
    for configured_team, lineup in OFFLINE_SAMPLE_LINEUPS.items():
        if normalize_player_name(configured_team) == normalize_player_name(team_name):
            return [dict(player) for player in lineup]
    return []


def resolve_starting_lineups(
    home_team: str,
    away_team: str,
    lineups_data: Optional[List[Dict]],
    offline: bool,
) -> tuple[List[Dict], List[Dict], str, str]:
    """
    Resolves validated starting XIs.

    Live mode fails closed when either confirmed XI is unavailable. Offline mode
    uses the explicit configured mock XI instead of deriving a stale historical
    roster from prior player-stat payloads.
    """
    parsed_by_team: Dict[str, List[Dict]] = {}
    for team_data in lineups_data or []:
        team_name = team_data.get("team", {}).get("name", "")
        parsed = []
        for entry in team_data.get("startXI", []):
            player = entry.get("player", {})
            if player.get("name"):
                parsed.append({
                    "name": player["name"],
                    "position": player.get("pos", "M"),
                })
        if team_name:
            parsed_by_team[normalize_player_name(team_name)] = parsed

    def resolve(team_name: str) -> tuple[List[Dict], str]:
        confirmed = parsed_by_team.get(normalize_player_name(team_name), [])
        if _is_complete_starting_xi(confirmed):
            source = "cached offline XI" if offline else "verified live API XI"
            return confirmed, source
        if not offline:
            raise ConfirmedLineupDataError(
                f"Confirmed starting XI for {team_name} is unavailable or invalid; "
                "live report generation halted."
            )
        configured = _offline_sample_lineup(team_name)
        if not _is_complete_starting_xi(configured):
            raise RuntimeError(f"Offline sample starting XI for {team_name} is invalid.")
        return configured, "explicit offline sample XI"

    home_lineup, home_source = resolve(home_team)
    away_lineup, away_source = resolve(away_team)
    return home_lineup, away_lineup, home_source, away_source


def run_matchup_pipeline(
    home_team: str, 
    away_team: str, 
    fixture_id: int, 
    offline: bool = False,
    api_key: Optional[str] = None,
    sims: int = 10000,
    db_path: str = "world_cup_2026_completed_data.json",
    confirmed_lineups: Optional[List[Dict]] = None,
):
    """Orchestrates ingestion, simulation, Parquet export, and PDF reporting."""
    print("="*80)
    print(f"RUNNING SYNDICATE ALPHA ENGINE PIPELINE: {home_team} vs {away_team}")
    print("="*80)
    
    # 1. Initialize Pipeline
    pipeline = FootballDataPipeline(api_key=api_key, cache_file=db_path, offline=offline)
    
    # 2. Ingest squad historical statistics (trailing 5 games)
    print("\n[Step 1/5] Ingesting historical performance stats...")
    df_home_stats = pipeline.load_historical_team_stats(
        home_team,
        limit=5,
        exclude_fixture_id=fixture_id,
    )
    df_away_stats = pipeline.load_historical_team_stats(
        away_team,
        limit=5,
        exclude_fixture_id=fixture_id,
    )
    print(f" - Ingested {len(df_home_stats)} stat lines for {home_team}")
    print(f" - Ingested {len(df_away_stats)} stat lines for {away_team}")

    # 3. Calculate Opponent Defensive modifiers (containment factors)
    print("\n[Step 2/5] Evaluating defensive modifiers via Common-Opponent Bridge...")
    away_def_factor, home_def_factor = calculate_common_opponent_modifiers(
        team_home=home_team,
        team_away=away_team,
        database_path=db_path
    )
    print(f" - {away_team} defensive containment modifier on {home_team} attack: {away_def_factor:.4f}")
    print(f" - {home_team} defensive containment modifier on {away_team} attack: {home_def_factor:.4f}")

    # 4. Fetch lineup (starting XI)
    print("\n[Step 3/5] Extracting kickoff rosters...")
    lineups_data = (
        confirmed_lineups
        if confirmed_lineups is not None
        else pipeline.fetch_confirmed_lineups(fixture_id)
    )
    home_lineup, away_lineup, home_source, away_source = resolve_starting_lineups(
        home_team,
        away_team,
        lineups_data,
        offline,
    )
    home_lineup = [{**player, "team": home_team} for player in home_lineup]
    away_lineup = [{**player, "team": away_team} for player in away_lineup]

    print(f" - Resolved {len(home_lineup)} starters for {home_team} via {home_source}")
    print(f" - Resolved {len(away_lineup)} starters for {away_team} via {away_source}")

    # 5. Run Monte Carlo Simulation
    print(f"\n[Step 4/5] Executing {sims} Monte Carlo simulations...")
    home_sims = run_squad_simulation(home_lineup, df_home_stats, away_def_factor, sims=sims)
    away_sims = run_squad_simulation(away_lineup, df_away_stats, home_def_factor, sims=sims)
    
    # 6. Simulation intelligence dataset
    print("\n[Step 5/5] Building comprehensive Starting XI simulation matrix...")
    combined_sim_results = home_sims + away_sims

    # 7. Exports & Reporting
    # A. Save the complete outfield simulation matrix as Parquet
    matchup_slug = f"{_filename_slug(home_team)}_vs_{_filename_slug(away_team)}"
    parquet_filename = f"{matchup_slug}_shooting_projections.parquet"
    parquet_records = []
    for item in combined_sim_results:
        if item["position"] == "G":
            continue
        parquet_records.append({
            "Team": item["team"],
            "Player Name": item["player_name"],
            "Position": item["position"],
            "Projected Shots (Mean)": round(item["mean_shots"], 4),
            "Projected Shots on Target (Mean)": round(item["mean_sot"], 4),
            "Projected Shots Missed (Mean)": round(item["mean_missed"], 4),
            "Projected Goals (Mean)": round(item["mean_goals"], 4),
            "Simulated Any Shot Probability (%)": round(
                item["any_shot_probability"] * 100,
                4,
            ),
        })
    projections_df = pd.DataFrame(parquet_records)
    projections_df = projections_df.sort_values(
        by="Projected Shots (Mean)",
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    projections_df.to_parquet(parquet_filename, index=False)
    print(
        projections_df[
            [
                "Team",
                "Player Name",
                "Position",
                "Projected Shots (Mean)",
                "Projected Shots on Target (Mean)",
                "Simulated Any Shot Probability (%)",
            ]
        ].to_string(index=False)
    )
    print(f"\nProjections exported successfully to Parquet dataset: {parquet_filename}")

    # B. Generate PDF
    pdf_filename = f"{matchup_slug}_syndicate_report.pdf"
    date_str = datetime.date.today().strftime("%B %d, %Y")
    fixture_str = f"{home_team} vs. {away_team}"
    
    html_content = generate_html_report(
        date_str,
        fixture_str,
        projections_df,
        home_team=home_team,
        away_team=away_team,
        home_attack_modifier=away_def_factor,
        away_attack_modifier=home_def_factor,
        simulation_count=sims,
    )
    compile_pdf_report(html_content, pdf_filename)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    return {
        "home_lineup": home_lineup,
        "away_lineup": away_lineup,
        "projections": projections_df,
        "context": {
            "home_attack_modifier": away_def_factor,
            "away_attack_modifier": home_def_factor,
            "simulation_count": sims,
        },
        "parquet_path": parquet_filename,
        "pdf_path": pdf_filename,
    }


def poll_and_orchestrate(
    home_team: str,
    away_team: str,
    fixture_id: int,
    offline: bool = False,
    api_key: Optional[str] = None,
    sims: int = 10000,
    db_path: str = "world_cup_2026_completed_data.json"
):
    """
    Checks kickoff time, polls confirmed lineups, and runs the simulation at T-60.
    """
    pipeline = FootballDataPipeline(
        api_key=api_key,
        cache_file=db_path,
        offline=offline,
    )
    
    # 1. Check kickoff time
    print(f"Querying fixture {fixture_id} details...")
    try:
        meta = pipeline.get_fixture_details(fixture_id)
    except Exception as error:
        if not offline:
            raise ConfirmedLineupDataError(
                f"Fixture metadata fetch failed for live fixture {fixture_id}; "
                "T-60 monitoring halted."
            ) from error
        raise
    if not meta:
        if not offline:
            raise ConfirmedLineupDataError(
                f"Fixture metadata is unavailable for live fixture {fixture_id}; "
                "T-60 monitoring halted."
            )
        run_matchup_pipeline(home_team, away_team, fixture_id, offline, api_key, sims, db_path)
        return
        
    date_str = meta.get("fixture", {}).get("date")
    # Date parsing
    if date_str:
        # standard ISO date format (e.g. 2026-06-25T22:00:00+00:00)
        # Parse ISO date with timezone info
        try:
            kickoff_dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            # Fallback format check
            kickoff_dt = datetime.datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S")
            
        print(f"Match Kickoff Time: {kickoff_dt}")
        
        while True:
            now_dt = datetime.datetime.now(datetime.timezone.utc)
            # convert kickoff to UTC if needed
            kickoff_utc = kickoff_dt.astimezone(datetime.timezone.utc)
            time_diff_sec = (kickoff_utc - now_dt).total_seconds()
            time_diff_min = time_diff_sec / 60.0
            
            print(f"Current UTC Time: {now_dt.strftime('%H:%M:%S')} | Kickoff UTC: {kickoff_utc.strftime('%H:%M:%S')} | Diff: {time_diff_min:.2f} mins")
            
            # Acceptance Criteria: exactly T-60 minutes before international match kickoff
            if time_diff_min <= 60.5:
                print(f"Automation Window reached (T-60 kickoff check). Querying official lineup endpoints...")
                lineups = pipeline.fetch_confirmed_lineups(fixture_id)
                if not lineups and offline:
                    print("Offline mode: no cached XI found; using explicit offline sample lineups.")
                else:
                    print("Official starting lineups confirmed by API. Running simulation engine...")
                run_matchup_pipeline(
                    home_team,
                    away_team,
                    fixture_id,
                    offline,
                    api_key,
                    sims,
                    db_path,
                    confirmed_lineups=lineups,
                )
                break
            else:
                # Wait until T-60
                sleep_seconds = int((time_diff_min - 60.0) * 60)
                # Sleep a minimum of 30 seconds, maximum of 10 minutes to verify status
                sleep_seconds = max(30, min(600, sleep_seconds))
                print(f"Waiting {sleep_seconds} seconds until T-60 automation window...")
                time.sleep(sleep_seconds)
    else:
        if not offline:
            raise ConfirmedLineupDataError(
                f"Kickoff time is unavailable for live fixture {fixture_id}; "
                "T-60 monitoring halted."
            )
        run_matchup_pipeline(home_team, away_team, fixture_id, offline, api_key, sims, db_path)
