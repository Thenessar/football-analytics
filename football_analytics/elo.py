"""Chronological team and player Elo feature engineering.

The functions in this module are deliberately pandas-based. They are intended
for Databricks driver-side state updates after Spark/dbt has reduced the raw API
payloads to fixture and appearance-level tables. The important contract is that
all returned ratings are pre-match snapshots: the current fixture is appended to
the output before any state update from that fixture is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
import math
import unicodedata

import numpy as np
import pandas as pd


DEFAULT_TEAM_ELO = 1500.0
DEFAULT_BASE_GOAL_RATE = 1.25
DEFAULT_TEAM_ELO_K = 24.0
DEFAULT_GOAL_STATE_LEARNING_RATE = 0.035
DEFAULT_PLAYER_DECAY_ALPHA = 0.85
DEFAULT_PLAYER_MODIFIER_LEARNING_RATE = 0.05
DEFAULT_PLAYER_MODIFIER_CLIP = 1.5

STRUCTURAL_ZERO_EVENT_COLUMNS = [
    "offsides",
    "shots_total",
    "shots_on",
    "goals_total",
    "goals_assists",
    "goals_conceded",
    "goals_saves",
    "passes_total",
    "passes_key",
    "passes_accuracy",
    "tackles_total",
    "tackles_blocks",
    "tackles_interceptions",
    "duels_total",
    "duels_won",
    "dribbles_attempts",
    "dribbles_success",
    "dribbles_past",
    "fouls_drawn",
    "fouls_committed",
    "cards_yellow",
    "cards_red",
    "penalty_won",
    "penalty_committed",
    "penalty_scored",
    "penalty_missed",
    "penalty_saved",
]


@dataclass
class TeamEloState:
    """Mutable latent state for one national team."""

    team_key: str
    team_name: str
    elo_general: float = DEFAULT_TEAM_ELO
    elo_attack: float = 0.0
    elo_defense: float = 0.0


@dataclass
class PlayerModifierState:
    """Player-specific deviations around the current national-team baseline."""

    player_key: str
    player_name: str
    offensive_modifier: float = 0.0
    defensive_modifier: float = 0.0
    missed_fixture_count: int = 0


def normalize_key(value: Any) -> str:
    """Normalizes team/player names for fallback keys and ranking joins."""

    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.casefold().split())


def _clean_key(value: Any, fallback: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return normalize_key(fallback)
    return str(value)


def _optional_float(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = row.get(column, default)
    if pd.isna(value):
        return default
    return float(value)


def coalesce_structural_event_zeros(
    frame: pd.DataFrame,
    event_columns: Sequence[str] = STRUCTURAL_ZERO_EVENT_COLUMNS,
) -> pd.DataFrame:
    """Treats missing event counts on appearance rows as structural zeros."""

    out = frame.copy()
    for column in event_columns:
        if column in out.columns:
            out[column] = out[column].fillna(0.0).astype(float)
    return out


def assert_no_unhandled_event_nulls(
    frame: pd.DataFrame,
    event_columns: Sequence[str] = STRUCTURAL_ZERO_EVENT_COLUMNS,
) -> None:
    """Raises when any event-count column still contains NULL/NaN values."""

    missing = [
        column
        for column in event_columns
        if column in frame.columns and frame[column].isna().any()
    ]
    if missing:
        raise ValueError(f"Unhandled NULL event columns: {', '.join(missing)}")


def _ranking_baselines(
    rankings_df: Optional[pd.DataFrame],
    *,
    team_name_col: str = "team_name",
    fifa_points_col: str = "rating",
    fifa_points_scale: float = 1.0,
) -> Dict[str, float]:
    if rankings_df is None or rankings_df.empty:
        return {}

    rankings = rankings_df.copy()
    if team_name_col not in rankings or fifa_points_col not in rankings:
        return {}

    points = pd.to_numeric(rankings[fifa_points_col], errors="coerce")
    anchor = float(points.dropna().median()) if points.notna().any() else 1500.0
    baselines: Dict[str, float] = {}
    for _, row in rankings.iterrows():
        rating = row.get(fifa_points_col)
        if pd.isna(rating):
            continue
        key = normalize_key(row.get(team_name_col))
        baselines[key] = DEFAULT_TEAM_ELO + (float(rating) - anchor) * fifa_points_scale
    return baselines


def _get_team_state(
    states: Dict[str, TeamEloState],
    *,
    team_id: Any,
    team_name: Any,
    baseline_by_name: Dict[str, float],
) -> TeamEloState:
    team_key = _clean_key(team_id, team_name)
    if team_key not in states:
        name_key = normalize_key(team_name)
        states[team_key] = TeamEloState(
            team_key=team_key,
            team_name=str(team_name),
            elo_general=float(baseline_by_name.get(name_key, DEFAULT_TEAM_ELO)),
        )
    return states[team_key]


def expected_match_score(elo_for: float, elo_against: float) -> float:
    """Standard Elo win/draw/loss expectation."""

    return 1.0 / (1.0 + 10.0 ** ((float(elo_against) - float(elo_for)) / 400.0))


def _result_points(goals_for: float, goals_against: float) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def _margin_multiplier(goal_difference: float) -> float:
    return math.log(abs(float(goal_difference)) + 1.0) + 1.0


def _expected_goals(
    attack_for: float,
    defense_against: float,
    *,
    base_goal_rate: float,
    log_advantage: float = 0.0,
) -> float:
    log_lambda = math.log(base_goal_rate) + float(attack_for) - float(defense_against) + log_advantage
    return float(math.exp(np.clip(log_lambda, -4.0, 4.0)))


def build_team_elo_history(
    fixtures_df: pd.DataFrame,
    rankings_df: Optional[pd.DataFrame] = None,
    *,
    k_factor: float = DEFAULT_TEAM_ELO_K,
    goal_state_learning_rate: float = DEFAULT_GOAL_STATE_LEARNING_RATE,
    base_goal_rate: float = DEFAULT_BASE_GOAL_RATE,
    fifa_points_scale: float = 1.0,
    home_elo_advantage: float = 0.0,
    home_goal_log_advantage: float = 0.0,
    state_clip: float = 1.75,
) -> pd.DataFrame:
    """Builds point-in-time team Elo snapshots from completed fixtures.

    Expected fixture columns are `fixture_id`, `fixture_date_utc`,
    `home_team_id`, `home_team_name`, `away_team_id`, `away_team_name`,
    `home_goals`, and `away_goals`.
    """

    required = {
        "fixture_id",
        "fixture_date_utc",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "home_goals",
        "away_goals",
    }
    missing = sorted(required.difference(fixtures_df.columns))
    if missing:
        raise ValueError(f"fixtures_df is missing required columns: {', '.join(missing)}")

    baseline_by_name = _ranking_baselines(
        rankings_df,
        fifa_points_scale=fifa_points_scale,
    )
    fixtures = fixtures_df.copy()
    fixtures["fixture_date_utc"] = pd.to_datetime(fixtures["fixture_date_utc"], utc=True)
    fixtures = fixtures.sort_values(["fixture_date_utc", "fixture_id"]).reset_index(drop=True)

    states: Dict[str, TeamEloState] = {}
    snapshots = []

    for _, row in fixtures.iterrows():
        home = _get_team_state(
            states,
            team_id=row["home_team_id"],
            team_name=row["home_team_name"],
            baseline_by_name=baseline_by_name,
        )
        away = _get_team_state(
            states,
            team_id=row["away_team_id"],
            team_name=row["away_team_name"],
            baseline_by_name=baseline_by_name,
        )

        home_lambda = _expected_goals(
            home.elo_attack,
            away.elo_defense,
            base_goal_rate=base_goal_rate,
            log_advantage=home_goal_log_advantage,
        )
        away_lambda = _expected_goals(
            away.elo_attack,
            home.elo_defense,
            base_goal_rate=base_goal_rate,
        )

        common = {
            "fixture_id": row["fixture_id"],
            "fixture_date_utc": row["fixture_date_utc"],
        }
        snapshots.extend([
            {
                **common,
                "team_id": row["home_team_id"],
                "team_name": row["home_team_name"],
                "opponent_team_id": row["away_team_id"],
                "opponent_team_name": row["away_team_name"],
                "home_away": "home",
                "team_elo_general_pre": home.elo_general,
                "opponent_elo_general_pre": away.elo_general,
                "team_elo_attack_pre": home.elo_attack,
                "team_elo_defense_pre": home.elo_defense,
                "opponent_elo_attack_pre": away.elo_attack,
                "opponent_elo_defense_pre": away.elo_defense,
                "expected_goals_for_pre": home_lambda,
                "expected_goals_against_pre": away_lambda,
            },
            {
                **common,
                "team_id": row["away_team_id"],
                "team_name": row["away_team_name"],
                "opponent_team_id": row["home_team_id"],
                "opponent_team_name": row["home_team_name"],
                "home_away": "away",
                "team_elo_general_pre": away.elo_general,
                "opponent_elo_general_pre": home.elo_general,
                "team_elo_attack_pre": away.elo_attack,
                "team_elo_defense_pre": away.elo_defense,
                "opponent_elo_attack_pre": home.elo_attack,
                "opponent_elo_defense_pre": home.elo_defense,
                "expected_goals_for_pre": away_lambda,
                "expected_goals_against_pre": home_lambda,
            },
        ])
        snapshot_start = len(snapshots) - 2

        if pd.isna(row["home_goals"]) or pd.isna(row["away_goals"]):
            snapshots[snapshot_start].update({
                "team_elo_attack_post": home.elo_attack,
                "team_elo_defense_post": home.elo_defense,
            })
            snapshots[snapshot_start + 1].update({
                "team_elo_attack_post": away.elo_attack,
                "team_elo_defense_post": away.elo_defense,
            })
            continue

        home_goals = float(row["home_goals"])
        away_goals = float(row["away_goals"])
        importance = _optional_float(row, "game_importance_scalar", 1.0)

        home_expected_score = expected_match_score(
            home.elo_general + home_elo_advantage,
            away.elo_general,
        )
        home_actual_score = _result_points(home_goals, away_goals)
        general_delta = (
            k_factor
            * importance
            * _margin_multiplier(home_goals - away_goals)
            * (home_actual_score - home_expected_score)
        )
        home.elo_general += general_delta
        away.elo_general -= general_delta

        home_residual = home_goals - home_lambda
        away_residual = away_goals - away_lambda
        lr = goal_state_learning_rate * importance
        home.elo_attack = float(np.clip(home.elo_attack + lr * home_residual, -state_clip, state_clip))
        away.elo_defense = float(np.clip(away.elo_defense - lr * home_residual, -state_clip, state_clip))
        away.elo_attack = float(np.clip(away.elo_attack + lr * away_residual, -state_clip, state_clip))
        home.elo_defense = float(np.clip(home.elo_defense - lr * away_residual, -state_clip, state_clip))

        snapshots[snapshot_start].update({
            "team_elo_attack_post": home.elo_attack,
            "team_elo_defense_post": home.elo_defense,
        })
        snapshots[snapshot_start + 1].update({
            "team_elo_attack_post": away.elo_attack,
            "team_elo_defense_post": away.elo_defense,
        })

    return pd.DataFrame(snapshots)


def _safe_per90(numerator: pd.Series, minutes: pd.Series) -> pd.Series:
    minutes = pd.to_numeric(minutes, errors="coerce").fillna(0.0)
    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0.0)
    return np.where(minutes > 0, numerator * 90.0 / minutes.clip(lower=1.0), 0.0)


def add_player_match_signals(appearances_df: pd.DataFrame) -> pd.DataFrame:
    """Adds compact offensive and defensive per-90 signals for modifiers."""

    out = coalesce_structural_event_zeros(appearances_df)
    minutes = pd.to_numeric(out.get("games_minutes", 0.0), errors="coerce").fillna(0.0)
    out["games_minutes"] = minutes

    shots = out["shots_total"] if "shots_total" in out else 0.0
    shots_on = out["shots_on"] if "shots_on" in out else 0.0
    dribbles = out["dribbles_attempts"] if "dribbles_attempts" in out else 0.0
    assists = out["goals_assists"] if "goals_assists" in out else 0.0
    tackles = out["tackles_total"] if "tackles_total" in out else 0.0
    interceptions = out["tackles_interceptions"] if "tackles_interceptions" in out else 0.0
    fouls = out["fouls_committed"] if "fouls_committed" in out else 0.0

    out["player_offensive_signal_p90"] = _safe_per90(
        pd.Series(shots, index=out.index)
        + 0.50 * pd.Series(shots_on, index=out.index)
        + 0.35 * pd.Series(dribbles, index=out.index)
        + 0.70 * pd.Series(assists, index=out.index),
        minutes,
    )
    out["player_defensive_signal_p90"] = _safe_per90(
        pd.Series(interceptions, index=out.index)
        + 0.60 * pd.Series(tackles, index=out.index)
        + 0.15 * pd.Series(fouls, index=out.index),
        minutes,
    )
    return out


def _team_fixture_key(fixture_id: Any, team_id: Any, team_name: Any) -> Tuple[Any, str]:
    return fixture_id, _clean_key(team_id, team_name)


def _get_player_state(
    states: Dict[Tuple[str, str], PlayerModifierState],
    *,
    team_key: str,
    player_id: Any,
    player_name: Any,
) -> PlayerModifierState:
    player_key = _clean_key(player_id, player_name)
    state_key = (team_key, player_key)
    if state_key not in states:
        states[state_key] = PlayerModifierState(
            player_key=player_key,
            player_name=str(player_name),
        )
    return states[state_key]


def build_player_elo_history(
    appearances_df: pd.DataFrame,
    team_elo_history_df: pd.DataFrame,
    *,
    decay_alpha: float = DEFAULT_PLAYER_DECAY_ALPHA,
    offensive_learning_rate: float = DEFAULT_PLAYER_MODIFIER_LEARNING_RATE,
    defensive_learning_rate: float = DEFAULT_PLAYER_MODIFIER_LEARNING_RATE,
    modifier_clip: float = DEFAULT_PLAYER_MODIFIER_CLIP,
) -> pd.DataFrame:
    """Builds pre-match hierarchical player ratings with inactivity decay.

    Player ratings are represented as current national-team baseline plus a
    player-specific modifier. Known players who miss a national-team fixture have
    their modifiers multiplied by `decay_alpha` before any appearance snapshots
    for that fixture are emitted.
    """

    if not 0.0 < decay_alpha <= 1.0:
        raise ValueError("decay_alpha must be in the interval (0, 1]")

    required_team_columns = {
        "fixture_id",
        "fixture_date_utc",
        "team_id",
        "team_name",
        "team_elo_general_pre",
        "team_elo_attack_pre",
        "team_elo_defense_pre",
    }
    missing_team = sorted(required_team_columns.difference(team_elo_history_df.columns))
    if missing_team:
        raise ValueError(
            "team_elo_history_df is missing required columns: "
            + ", ".join(missing_team)
        )

    required_appearance_columns = {
        "fixture_id",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "games_minutes",
    }
    missing_apps = sorted(required_appearance_columns.difference(appearances_df.columns))
    if missing_apps:
        raise ValueError(
            "appearances_df is missing required columns: "
            + ", ".join(missing_apps)
        )

    appearances = add_player_match_signals(appearances_df)
    if "fixture_date_utc" in appearances.columns:
        appearances["fixture_date_utc"] = pd.to_datetime(
            appearances["fixture_date_utc"],
            utc=True,
            errors="coerce",
        )
    appearances["_team_key"] = appearances.apply(
        lambda row: _clean_key(row["team_id"], row["team_name"]),
        axis=1,
    )

    grouped_apps = {
        key: group.copy()
        for key, group in appearances.groupby(["fixture_id", "_team_key"], dropna=False)
    }

    team_rows = team_elo_history_df.copy()
    team_rows["fixture_date_utc"] = pd.to_datetime(team_rows["fixture_date_utc"], utc=True)
    team_rows = team_rows.sort_values(
        ["fixture_date_utc", "fixture_id", "team_id"],
        kind="mergesort",
    )

    states: Dict[Tuple[str, str], PlayerModifierState] = {}
    team_rosters: Dict[str, set[str]] = {}
    snapshots = []

    for _, team_row in team_rows.iterrows():
        team_key = _clean_key(team_row["team_id"], team_row["team_name"])
        team_rosters.setdefault(team_key, set())
        fixture_key = _team_fixture_key(
            team_row["fixture_id"],
            team_row["team_id"],
            team_row["team_name"],
        )
        apps = grouped_apps.get(fixture_key, pd.DataFrame())
        if apps.empty:
            present_players: set[str] = set()
        else:
            present_players = {
                _clean_key(row["player_id"], row["player_name"])
                for _, row in apps.iterrows()
            }

        for player_key in sorted(team_rosters[team_key].difference(present_players)):
            state = states[(team_key, player_key)]
            state.offensive_modifier *= decay_alpha
            state.defensive_modifier *= decay_alpha
            state.missed_fixture_count += 1

        if apps.empty:
            continue

        active_apps = apps[apps["games_minutes"] > 0].copy()
        inactive_apps = apps[apps["games_minutes"] <= 0].copy()
        if active_apps.empty:
            team_off_avg = 0.0
            team_def_avg = 0.0
        else:
            team_off_avg = float(
                np.average(
                    active_apps["player_offensive_signal_p90"],
                    weights=active_apps["games_minutes"].clip(lower=1.0),
                )
            )
            team_def_avg = float(
                np.average(
                    active_apps["player_defensive_signal_p90"],
                    weights=active_apps["games_minutes"].clip(lower=1.0),
                )
            )

        for _, app_row in inactive_apps.sort_values(["player_id", "player_name"]).iterrows():
            player_key = _clean_key(app_row["player_id"], app_row["player_name"])
            state = _get_player_state(
                states,
                team_key=team_key,
                player_id=app_row["player_id"],
                player_name=app_row["player_name"],
            )
            if player_key in team_rosters[team_key]:
                state.offensive_modifier *= decay_alpha
                state.defensive_modifier *= decay_alpha
                state.missed_fixture_count += 1
            team_rosters[team_key].add(player_key)

        for _, app_row in apps.sort_values(["player_id", "player_name"]).iterrows():
            player_key = _clean_key(app_row["player_id"], app_row["player_name"])
            state = _get_player_state(
                states,
                team_key=team_key,
                player_id=app_row["player_id"],
                player_name=app_row["player_name"],
            )
            team_rosters[team_key].add(player_key)

            snapshots.append({
                "fixture_id": app_row["fixture_id"],
                "fixture_date_utc": team_row["fixture_date_utc"],
                "team_id": team_row["team_id"],
                "team_name": team_row["team_name"],
                "player_id": app_row["player_id"],
                "player_name": app_row["player_name"],
                "games_minutes": float(app_row["games_minutes"]),
                "team_elo_general_pre": float(team_row["team_elo_general_pre"]),
                "team_elo_attack_pre": float(team_row["team_elo_attack_pre"]),
                "team_elo_defense_pre": float(team_row["team_elo_defense_pre"]),
                "player_offensive_modifier_pre": float(state.offensive_modifier),
                "player_defensive_modifier_pre": float(state.defensive_modifier),
                "player_offensive_elo_pre": float(
                    team_row["team_elo_attack_pre"] + state.offensive_modifier
                ),
                "player_defensive_elo_pre": float(
                    team_row["team_elo_defense_pre"] + state.defensive_modifier
                ),
                "player_offensive_rating_pre": float(
                    team_row["team_elo_attack_pre"] + state.offensive_modifier
                ),
                "player_defensive_rating_pre": float(
                    team_row["team_elo_defense_pre"] + state.defensive_modifier
                ),
                "missed_fixture_count_pre": int(state.missed_fixture_count),
                "player_offensive_signal_p90": float(app_row["player_offensive_signal_p90"]),
                "player_defensive_signal_p90": float(app_row["player_defensive_signal_p90"]),
                "team_offensive_signal_p90": team_off_avg,
                "team_defensive_signal_p90": team_def_avg,
            })

        if active_apps.empty:
            continue

        for _, app_row in active_apps.iterrows():
            state = _get_player_state(
                states,
                team_key=team_key,
                player_id=app_row["player_id"],
                player_name=app_row["player_name"],
            )
            exposure = min(max(float(app_row["games_minutes"]) / 90.0, 0.0), 1.3)
            offensive_residual = float(app_row["player_offensive_signal_p90"]) - team_off_avg
            defensive_residual = float(app_row["player_defensive_signal_p90"]) - team_def_avg
            state.offensive_modifier = float(
                np.clip(
                    state.offensive_modifier
                    + offensive_learning_rate * offensive_residual * exposure,
                    -modifier_clip,
                    modifier_clip,
                )
            )
            state.defensive_modifier = float(
                np.clip(
                    state.defensive_modifier
                    + defensive_learning_rate * defensive_residual * exposure,
                    -modifier_clip,
                    modifier_clip,
                )
            )
            state.missed_fixture_count = 0

    return pd.DataFrame(snapshots)


def assemble_hierarchical_feature_frame(
    player_features_df: pd.DataFrame,
    team_elo_history_df: pd.DataFrame,
    player_elo_history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merges dbt player features with team and player pre-match Elo features."""

    out = coalesce_structural_event_zeros(player_features_df)
    team_columns = [
        "fixture_id",
        "team_id",
        "team_elo_general_pre",
        "opponent_elo_general_pre",
        "team_elo_attack_pre",
        "team_elo_defense_pre",
        "opponent_elo_attack_pre",
        "opponent_elo_defense_pre",
        "expected_goals_for_pre",
        "expected_goals_against_pre",
    ]
    existing_team_columns = [column for column in team_columns if column in team_elo_history_df.columns]
    team_payload_columns = [
        column
        for column in existing_team_columns
        if column not in {"fixture_id", "team_id"}
    ]
    out = out.drop(columns=[column for column in team_payload_columns if column in out.columns])
    out = out.merge(
        team_elo_history_df[existing_team_columns],
        on=["fixture_id", "team_id"],
        how="left",
    )

    if player_elo_history_df is not None and not player_elo_history_df.empty:
        player_columns = [
            "fixture_id",
            "team_id",
            "player_id",
            "player_offensive_modifier_pre",
            "player_defensive_modifier_pre",
            "player_offensive_elo_pre",
            "player_defensive_elo_pre",
            "player_offensive_rating_pre",
            "player_defensive_rating_pre",
            "missed_fixture_count_pre",
        ]
        existing_player_columns = [
            column for column in player_columns if column in player_elo_history_df.columns
        ]
        player_payload_columns = [
            column
            for column in existing_player_columns
            if column not in {"fixture_id", "team_id", "player_id"}
        ]
        out = out.drop(columns=[column for column in player_payload_columns if column in out.columns])
        out = out.merge(
            player_elo_history_df[existing_player_columns],
            on=["fixture_id", "team_id", "player_id"],
            how="left",
        )

    modifier_columns = [
        "player_offensive_modifier_pre",
        "player_defensive_modifier_pre",
        "missed_fixture_count_pre",
    ]
    for column in modifier_columns:
        if column in out.columns:
            out[column] = out[column].fillna(0.0)

    if "player_offensive_rating_pre" in out and "team_elo_attack_pre" in out:
        out["player_offensive_rating_pre"] = out["player_offensive_rating_pre"].fillna(
            out["team_elo_attack_pre"]
        )
    if "player_defensive_rating_pre" in out and "team_elo_defense_pre" in out:
        out["player_defensive_rating_pre"] = out["player_defensive_rating_pre"].fillna(
            out["team_elo_defense_pre"]
        )
    if "player_offensive_elo_pre" in out and "team_elo_attack_pre" in out:
        out["player_offensive_elo_pre"] = out["player_offensive_elo_pre"].fillna(
            out["team_elo_attack_pre"]
        )
    if "player_defensive_elo_pre" in out and "team_elo_defense_pre" in out:
        out["player_defensive_elo_pre"] = out["player_defensive_elo_pre"].fillna(
            out["team_elo_defense_pre"]
        )

    return out
