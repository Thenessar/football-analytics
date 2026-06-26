import json
import os
import unicodedata
from datetime import date
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from football_analytics.config import POSITIONAL_PRIORS, RECENCY_HALF_LIFE_DAYS


def normalize_player_name(value: str) -> str:
    """Produces an accent-insensitive, case-insensitive key for player joins."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.casefold().split())

def calculate_common_opponent_modifiers(
    team_home: str, 
    team_away: str, 
    database_path: str = "world_cup_2026_completed_data.json"
) -> Tuple[float, float]:
    """
    Calculates independent defensive containment ratings for both teams.

    For each mutual opponent, the opponent's shot output against the evaluated
    defense is compared with that opponent's average output in matches that do
    not involve the evaluated team. This leave-one-defense-out baseline prevents
    the evaluated match from contaminating its own expectation and avoids
    manufacturing zero-sum mirrored coefficients.

    Returns:
        (away_def_factor_on_home_attack, home_def_factor_on_away_attack)
    """
    if not os.path.exists(database_path):
        print(f"Database file '{database_path}' not found. Using neutral modifiers.")
        return 1.0, 1.0

    try:
        with open(database_path, "r", encoding="utf-8") as f:
            db = json.load(f)
    except Exception as e:
        print(f"Error reading database: {e}. Defaulting modifiers to 1.0.")
        return 1.0, 1.0

    # Helper function to get team shots in a match
    def get_team_shots_in_match(match_data: Dict, team_name: str) -> int:
        shots = 0
        for team_stat in match_data.get("player_statistics", []):
            if team_stat["team"]["name"].lower() == team_name.lower():
                for p in team_stat["players"]:
                    for s in p.get("statistics", []):
                        shots += (s.get("shots", {}).get("total") or 0)
        return shots

    # Helper to find all opponents and their matches
    # Maps: team_name -> {opponent_name: [fixture_id1, fixture_id2]}
    team_opponents: Dict[str, Dict[str, List[str]]] = {}
    
    # Maps: team_name -> fixture_id -> team shots in that fixture
    team_shots_by_fixture: Dict[str, Dict[str, int]] = {}

    for fid, match_data in db.items():
        home = match_data.get("match_info", {}).get("home_team")
        away = match_data.get("match_info", {}).get("away_team")
        if not home or not away:
            continue
            
        home_shots = get_team_shots_in_match(match_data, home)
        away_shots = get_team_shots_in_match(match_data, away)
        
        team_shots_by_fixture.setdefault(home, {})[fid] = home_shots
        team_shots_by_fixture.setdefault(away, {})[fid] = away_shots

        # Track home's opponent
        team_opponents.setdefault(home, {}).setdefault(away, []).append(fid)
        # Track away's opponent
        team_opponents.setdefault(away, {}).setdefault(home, []).append(fid)

    # Find mutual opponents of team_home and team_away
    home_opps = set(team_opponents.get(team_home, {}).keys())
    away_opps = set(team_opponents.get(team_away, {}).keys())
    mutual_opponents = home_opps.intersection(away_opps)

    if not mutual_opponents:
        print(f"No mutual opponents found between '{team_home}' and '{team_away}'. Returning neutral modifiers.")
        return 1.0, 1.0

    print(f"Mutual opponents found: {list(mutual_opponents)}")

    all_shot_observations = [
        shots
        for fixture_shots in team_shots_by_fixture.values()
        for shots in fixture_shots.values()
    ]
    tournament_baseline = (
        float(np.mean(all_shot_observations))
        if all_shot_observations
        else 1.0
    )

    def calculate_team_defensive_factor(
        defending_team: str,
        opponents: set[str],
    ) -> float:
        actual_shots_allowed = 0.0
        expected_shots_allowed = 0.0

        for opponent in opponents:
            evaluated_fixture_ids = set(
                team_opponents.get(defending_team, {}).get(opponent, [])
            )
            if not evaluated_fixture_ids:
                continue

            opponent_fixture_shots = team_shots_by_fixture.get(opponent, {})
            actual_values = [
                opponent_fixture_shots[fixture_id]
                for fixture_id in evaluated_fixture_ids
                if fixture_id in opponent_fixture_shots
            ]
            if not actual_values:
                continue

            reference_values = [
                shots
                for fixture_id, shots in opponent_fixture_shots.items()
                if fixture_id not in evaluated_fixture_ids
            ]
            opponent_baseline = (
                float(np.mean(reference_values))
                if reference_values
                else tournament_baseline
            )
            if opponent_baseline <= 0:
                continue

            actual_shots_allowed += float(np.sum(actual_values))
            expected_shots_allowed += opponent_baseline * len(actual_values)

        if expected_shots_allowed <= 0:
            return 1.0
        return actual_shots_allowed / expected_shots_allowed

    home_def_factor = calculate_team_defensive_factor(
        team_home,
        mutual_opponents,
    )
    away_def_factor = calculate_team_defensive_factor(
        team_away,
        mutual_opponents,
    )

    # Clip to standard thresholds [0.5, 1.5] to prevent mathematical instability
    home_def_factor = float(np.clip(home_def_factor, 0.5, 1.5))
    away_def_factor = float(np.clip(away_def_factor, 0.5, 1.5))

    print(f"Calculated Common-Opponent defensive containment rating:")
    print(f" - {team_home} containment (reduces opponent shot rate to): {home_def_factor:.4f}")
    print(f" - {team_away} containment (reduces opponent shot rate to): {away_def_factor:.4f}")

    # Return (away_def_factor_on_home_attack, home_def_factor_on_away_attack)
    return away_def_factor, home_def_factor


def map_position(pos_str: str) -> str:
    """Standardizes API position names into baselines (F, M, D, G)."""
    if not pos_str:
        return 'M'
    pos_str = pos_str.upper()
    if pos_str in ['F', 'A', 'ATTACKER', 'FORWARD', 'STRIKER']:
        return 'F'
    elif pos_str in ['M', 'MIDFIELDER', 'MIDFIELD']:
        return 'M'
    elif pos_str in ['D', 'DEFENDER', 'DEFENCE']:
        return 'D'
    elif pos_str in ['G', 'GK', 'GOALKEEPER']:
        return 'G'
    return 'M'


def add_recency_weights(
    player_history: pd.DataFrame,
    target_date: str | date | pd.Timestamp | None,
    half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> pd.DataFrame:
    """Adds exponential half-life weights using fixture_date relative to target_date."""
    if player_history is None or player_history.empty or target_date is None:
        weighted = player_history.copy() if player_history is not None else pd.DataFrame()
        if not weighted.empty:
            weighted["recency_weight"] = 1.0
        return weighted
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")

    weighted = player_history.copy()
    if "fixture_date" not in weighted.columns:
        weighted["recency_weight"] = 1.0
        return weighted

    target_ts = pd.to_datetime(target_date, utc=True)
    fixture_dates = pd.to_datetime(weighted["fixture_date"], utc=True, errors="coerce")
    age_days = (target_ts - fixture_dates).dt.total_seconds() / 86400.0
    age_days = age_days.clip(lower=0).fillna(0.0)
    decay_lambda = np.log(2.0) / float(half_life_days)
    weighted["recency_weight"] = np.exp(-decay_lambda * age_days)
    return weighted


def run_player_monte_carlo(
    player_name: str,
    position: str,
    player_history: pd.DataFrame,
    opp_def_factor: float,
    sims: int = 10000,
    target_date: str | date | pd.Timestamp | None = None,
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> Dict[str, Any]:
    """Runs a Bayesian-smoothed Monte Carlo simulation for a single starting player."""
    pos_key = map_position(position)
    prior = POSITIONAL_PRIORS.get(pos_key, POSITIONAL_PRIORS["M"])
    
    # 1. Bayesian Positional Smoothing Matrix Calculations
    if player_history is not None and not player_history.empty:
        player_history = add_recency_weights(
            player_history,
            target_date,
            recency_half_life_days,
        )
        recency_weight = (
            player_history["recency_weight"]
            if "recency_weight" in player_history.columns
            else 1.0
        )
        # Sum total minutes played in historical window after recency decay
        mins_played = (player_history['games_minutes'] * recency_weight).sum()
        # Full validation requires 5 games (450 minutes)
        weight = min(1.0, mins_played / 450.0)

        total_shots = (player_history['shots_total'] * recency_weight).sum()
        total_sot = (player_history['shots_on'] * recency_weight).sum()
        total_goals = (player_history['goals_total'] * recency_weight).sum()
        
        emp_shots_per_90 = (total_shots / mins_played) * 90 if mins_played > 0 else prior["shots"]
        emp_sot_pct = total_sot / total_shots if total_shots > 0 else prior["sot_pct"]
        emp_conv = total_goals / total_sot if total_sot > 0 else prior["conversion"]
        
        # Smooth parameters toward positional priors
        shots_per_90 = (weight * emp_shots_per_90) + ((1 - weight) * prior["shots"])
        sot_pct = (weight * emp_sot_pct) + ((1 - weight) * prior["sot_pct"])
        conversion = (weight * emp_conv) + ((1 - weight) * prior["conversion"])
    else:
        # If no player history, use pure positional priors
        mins_played = 0
        weight = 0.0
        shots_per_90 = prior["shots"]
        sot_pct = prior["sot_pct"]
        conversion = prior["conversion"]

    # Ensure parameters don't exceed logical mathematical limits
    sot_pct = np.clip(sot_pct, 0.01, 0.99)
    conversion = np.clip(conversion, 0.01, 0.99)

    # 2. Apply Opponent Defensive Factor (Damping Vector)
    # This directly dampens the lambda parameter for the shots Poisson process
    adjusted_shot_lambda = shots_per_90 * opp_def_factor
    
    # 3. Simulate shot generation and conversion loops
    # Poisson distribution for shot generation
    sim_shots = np.random.poisson(adjusted_shot_lambda, sims)
    # Binomial distribution for shots on target
    sim_sot = np.random.binomial(sim_shots, sot_pct)
    # Binomial distribution for goals
    sim_goals = np.random.binomial(sim_sot, conversion)
    
    sim_missed = sim_shots - sim_sot

    return {
        "player_name": player_name,
        "position": pos_key,
        "minutes_played": float(mins_played),
        "smoothing_weight": weight,
        "adjusted_lambda": adjusted_shot_lambda,
        "sot_pct": sot_pct,
        "conversion": conversion,
        # Simulation Output Arrays
        "sim_shots": sim_shots,
        "sim_sot": sim_sot,
        "sim_goals": sim_goals,
        "sim_missed": sim_missed,
        # Expected Means
        "mean_shots": float(np.mean(sim_shots)),
        "mean_sot": float(np.mean(sim_sot)),
        "mean_goals": float(np.mean(sim_goals)),
        "mean_missed": float(np.mean(sim_missed)),
        "any_shot_probability": float(np.mean(sim_shots > 0)),
    }


def run_squad_simulation(
    lineup: List[Dict[str, Any]], 
    history_df: pd.DataFrame, 
    opp_def_factor: float, 
    sims: int = 10000,
    target_date: str | date | pd.Timestamp | None = None,
    recency_half_life_days: float = RECENCY_HALF_LIFE_DAYS,
) -> List[Dict[str, Any]]:
    """Runs Monte Carlo simulations for an entire starting XI lineup."""
    results = []
    normalized_history = history_df.copy() if history_df is not None else pd.DataFrame()
    if not normalized_history.empty:
        normalized_history["_normalized_player_name"] = normalized_history["player_name"].map(
            normalize_player_name
        )

    for member in lineup:
        p_name = member["name"]
        pos = member["position"]
        p_history = (
            normalized_history[
                normalized_history["_normalized_player_name"] == normalize_player_name(p_name)
            ]
            if not normalized_history.empty
            else None
        )
        res = run_player_monte_carlo(
            p_name,
            pos,
            p_history,
            opp_def_factor,
            sims=sims,
            target_date=target_date,
            recency_half_life_days=recency_half_life_days,
        )
        res["team"] = member.get("team", "")
        results.append(res)
    return results
