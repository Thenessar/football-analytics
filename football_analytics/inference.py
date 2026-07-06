"""Pre-match player event inference (business_logic.md §14–§15).

Pure logic lives here so it is unit-testable; Databricks notebooks wire the
Spark/MLflow effects. The flow implemented across this module and the
`prematch_inference_pipeline` job follows §14.3:

1.  fetch/refresh fixture metadata           (notebook 03, bronze ingestion)
2.  fetch confirmed lineups                  (notebook 03)
3.  write lineups to Bronze                  (notebook 03)
4.  run required dbt Silver/Gold transforms  (dbt task in the job)
5.  build inference rows                     (fct_football__player_event_features)
6.  estimate expected minutes                (fct_football__player_expected_minutes,
                                              joined into the feature mart)
7.  load trained models from MLflow          (load_registered_event_models)
8.  predict per player/target                (build_prediction_records)
9.  store predictions                        (write_player_event_predictions)
10. display in notebook                      (notebook 04)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from football_analytics.ml_training import (
    DEFAULT_TARGET_COLUMNS,
    GOALKEEPER_ONLY_TARGETS,
    add_model_interaction_features,
    count_threshold_probabilities,
)

PREDICTION_TABLE_NAME = "pred_football__player_event_predictions"
PREDICTION_THRESHOLDS = (1, 2, 3)
INFERENCE_MODE_LIVE = "live"
INFERENCE_MODE_EXPLORATORY = "exploratory"
LINEUP_SOURCE_CONFIRMED = "confirmed_api"
LINEUP_SOURCE_PROJECTED = "projected"


class MissingLineupError(RuntimeError):
    """Raised in live mode when a fixture lacks confirmed lineups (§14.4)."""


@dataclass(frozen=True)
class LoadedEventModel:
    """A registered per-target model plus the metadata predictions must carry."""

    target_event: str
    booster: Any
    feature_columns: tuple[str, ...]
    model_name: str
    model_version: str
    model_stage_or_alias: Optional[str] = None
    goalkeeper_only: bool = False


def fixture_has_confirmed_starting_xi(feature_rows: pd.DataFrame) -> bool:
    """True when both teams have at least 11 confirmed starters in the rows."""

    if feature_rows.empty:
        return False
    required = {"team_id", "player_id", "is_starting"}
    if not required.issubset(feature_rows.columns):
        return False
    starters = feature_rows[feature_rows["is_starting"].fillna(False).astype(bool)]
    if starters.empty:
        return False
    counts = starters.groupby("team_id")["player_id"].nunique()
    return len(counts) >= 2 and bool((counts >= 11).all())


def resolve_lineup_source(mode: str, *, lineups_confirmed: bool) -> str:
    """Applies the §14.4 guard rails and returns the lineup_source tag.

    Live/production mode refuses to produce predictions without confirmed
    lineups; exploratory mode proceeds but tags output as non-official.
    """

    if lineups_confirmed:
        return LINEUP_SOURCE_CONFIRMED
    if mode == INFERENCE_MODE_LIVE:
        raise MissingLineupError(
            "Confirmed starting XI is not available for both teams; "
            "live-mode inference is skipped (business_logic.md §14.4). "
            "Rerun in exploratory mode to generate non-official predictions."
        )
    return LINEUP_SOURCE_PROJECTED


def lineup_signature(feature_rows: pd.DataFrame) -> str:
    """Deterministic digest of the fixture's lineup composition."""

    entries = sorted(
        (
            int(row.team_id),
            int(row.player_id),
            bool(row.is_starting) if pd.notna(row.is_starting) else False,
        )
        for row in feature_rows[["team_id", "player_id", "is_starting"]].itertuples()
    )
    return hashlib.sha256(json.dumps(entries).encode("utf-8")).hexdigest()


def deterministic_prediction_set_id(
    fixture_id: int,
    *,
    lineup_digest: str,
    model_versions: Mapping[str, str],
) -> str:
    """Deterministic id per fixture inference run (§15.4 Option C).

    The same fixture, lineup composition, and model versions always produce
    the same prediction_set_id, so redundant polling runs replace themselves
    instead of stacking new active sets.
    """

    payload = {
        "fixture_id": int(fixture_id),
        "lineup_digest": lineup_digest,
        "model_versions": {str(k): str(v) for k, v in sorted(model_versions.items())},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _numeric_features(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    for column in feature_columns:
        if column in frame.columns:
            features[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            features[column] = 0.0
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def expected_exposure_from_rows(feature_rows: pd.DataFrame) -> np.ndarray:
    """Exposure offset from expected minutes (§13.3 inference-time rule)."""

    if "expected_minutes" not in feature_rows.columns:
        raise ValueError("Inference rows are missing expected_minutes")
    minutes = pd.to_numeric(feature_rows["expected_minutes"], errors="coerce")
    return np.clip(minutes.to_numpy(dtype=float) / 90.0, 1e-6, None)


def build_prediction_records(
    feature_rows: pd.DataFrame,
    models: Sequence[LoadedEventModel],
    *,
    prediction_set_id: str,
    prediction_run_id: str,
    lineup_source: str,
    feature_table_name: str,
    feature_table_version: Optional[str] = None,
    thresholds: Sequence[int] = PREDICTION_THRESHOLDS,
    created_at_utc: Optional[datetime] = None,
) -> pd.DataFrame:
    """Builds §15.3-shaped prediction rows for one fixture's inference rows."""

    if feature_rows.empty:
        return pd.DataFrame()
    fixture_ids = feature_rows["fixture_id"].unique()
    if len(fixture_ids) != 1:
        raise ValueError("build_prediction_records expects rows for exactly one fixture")

    enriched = add_model_interaction_features(feature_rows.reset_index(drop=True))
    exposure = expected_exposure_from_rows(enriched)
    created_at = created_at_utc or datetime.now(timezone.utc)
    goalkeeper_mask = (
        enriched["is_goalkeeper"].fillna(False).astype(bool).to_numpy()
        if "is_goalkeeper" in enriched.columns
        else np.zeros(len(enriched), dtype=bool)
    )

    records = []
    for model in models:
        X = _numeric_features(enriched, model.feature_columns)
        raw_scores = np.asarray(model.booster.predict(X, raw_score=True), dtype=float)
        means = np.exp(raw_scores + np.log(exposure))
        if model.goalkeeper_only:
            # Structural zero for outfield players (§13.3, ADR 0003).
            means = np.where(goalkeeper_mask, means, 0.0)

        for index, row in enumerate(enriched.itertuples()):
            probabilities = count_threshold_probabilities(
                means[index], thresholds=tuple(thresholds)
            )
            records.append({
                "prediction_set_id": prediction_set_id,
                "prediction_run_id": prediction_run_id,
                "fixture_id": int(row.fixture_id),
                "fixture_date_utc": getattr(row, "fixture_date_utc", None),
                "prediction_created_at_utc": created_at,
                "team_id": int(row.team_id),
                "team_name": getattr(row, "team_name", None),
                "opponent_team_id": getattr(row, "opponent_team_id", None),
                "opponent_team_name": getattr(row, "opponent_team_name", None),
                "home_away": getattr(row, "home_away", None),
                "player_id": int(row.player_id),
                "player_name": getattr(row, "player_name", None),
                "position_group": getattr(row, "position_group", None),
                "is_starting": bool(row.is_starting) if pd.notna(row.is_starting) else False,
                "formation": getattr(row, "formation", None),
                "target_event": model.target_event,
                "expected_minutes": float(row.expected_minutes) if pd.notna(row.expected_minutes) else None,
                "predicted_mean": float(means[index]),
                "predicted_p_ge_1": probabilities.get("p_ge_1"),
                "predicted_p_ge_2": probabilities.get("p_ge_2"),
                "predicted_p_ge_3": probabilities.get("p_ge_3"),
                "model_name": model.model_name,
                "model_version": str(model.model_version),
                "model_stage_or_alias": model.model_stage_or_alias,
                "feature_table_name": feature_table_name,
                "feature_table_version": feature_table_version,
                "lineup_source": lineup_source,
                "is_active_prediction": True,
            })
    return pd.DataFrame.from_records(records)


def load_registered_event_models(
    *,
    registered_model_name_prefix: str,
    target_events: Sequence[str] = DEFAULT_TARGET_COLUMNS,
    alias: Optional[str] = None,
) -> list[LoadedEventModel]:
    """Loads the per-target boosters from the Unity Catalog model registry."""

    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError as exc:  # pragma: no cover - exercised in Databricks
        raise RuntimeError(
            "MLflow is required to load registered models. Install the "
            "`football-analytics[modeling]` optional dependencies."
        ) from exc

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    models = []
    for target in target_events:
        model_name = f"{registered_model_name_prefix}__{target}"
        if alias:
            version = client.get_model_version_by_alias(model_name, alias).version
            model_uri = f"models:/{model_name}@{alias}"
        else:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise RuntimeError(f"No registered versions found for {model_name}")
            version = max(int(entry.version) for entry in versions)
            model_uri = f"models:/{model_name}/{version}"
        booster = mlflow.lightgbm.load_model(model_uri)
        models.append(LoadedEventModel(
            target_event=target,
            booster=booster,
            feature_columns=tuple(booster.feature_name()),
            model_name=model_name,
            model_version=str(version),
            model_stage_or_alias=alias,
            goalkeeper_only=target in GOALKEEPER_ONLY_TARGETS,
        ))
    return models


def write_player_event_predictions(
    spark,
    records: pd.DataFrame,
    *,
    prediction_table: str,
) -> dict[str, int]:
    """Stores predictions using §15.4 Option C: append + active flag.

    - Rows for the same prediction_set_id are replaced (idempotent reruns).
    - After append, older rows for the same fixture/model/version are flipped
      to is_active_prediction=false, leaving exactly one active set.
    """

    if records.empty:
        return {"written": 0, "deactivated_sets": 0}

    frame = spark.createDataFrame(records)
    table_exists = bool(spark.catalog.tableExists(prediction_table))
    set_ids = sorted(records["prediction_set_id"].unique())
    if table_exists:
        quoted = ", ".join(f"'{set_id}'" for set_id in set_ids)
        spark.sql(
            f"DELETE FROM {prediction_table} WHERE prediction_set_id IN ({quoted})"
        )
    frame.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(prediction_table)

    deactivated = 0
    keys = records[["fixture_id", "model_name", "model_version", "prediction_set_id"]].drop_duplicates()
    for row in keys.itertuples():
        spark.sql(
            f"""
            UPDATE {prediction_table}
            SET is_active_prediction = false
            WHERE fixture_id = {int(row.fixture_id)}
              AND model_name = '{row.model_name}'
              AND model_version = '{row.model_version}'
              AND prediction_set_id != '{row.prediction_set_id}'
              AND is_active_prediction = true
            """
        )
        deactivated += 1
    return {"written": len(records), "deactivated_sets": deactivated}
