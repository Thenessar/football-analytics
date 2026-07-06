# Databricks notebook source
"""Pre-match inference steps 5-10 (business_logic.md §14.3).

Reads inference rows from the gold player event feature mart for fixtures in
the trigger window, enforces the §14.4 lineup guard rails, loads the
registered per-target models from the Unity Catalog registry, and stores
predictions with the append + active-flag strategy (§15.4 Option C).
"""
import logging
from datetime import datetime, timedelta, timezone

from football_analytics.databricks.config import DatabricksPipelineConfig, load_config_from_env
from football_analytics.databricks.logging import configure_json_logging
from football_analytics.databricks.tables import table_name
from football_analytics.databricks_ingestion import utc_now_iso
from football_analytics.inference import (
    INFERENCE_MODE_LIVE,
    PREDICTION_TABLE_NAME,
    MissingLineupError,
    build_prediction_records,
    deterministic_prediction_set_id,
    fixture_has_confirmed_starting_xi,
    lineup_signature,
    load_registered_event_models,
    resolve_lineup_source,
    write_player_event_predictions,
)
from football_analytics.ml_training import DEFAULT_TARGET_COLUMNS

dbutils.widgets.text("fixture_id", "")
dbutils.widgets.text("window_minutes", "75")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("catalog", "football_analytics")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("gold_schema", "gold")
dbutils.widgets.dropdown("mode", INFERENCE_MODE_LIVE, ["live", "exploratory"])
dbutils.widgets.text("registered_model_prefix", "football_analytics.gold.player_event")
dbutils.widgets.text("model_alias", "")
dbutils.widgets.text("targets", "")

env_config = load_config_from_env()
config = DatabricksPipelineConfig(
    catalog=dbutils.widgets.get("catalog"),
    bronze_schema=dbutils.widgets.get("bronze_schema"),
    silver_schema=dbutils.widgets.get("silver_schema"),
    gold_schema=dbutils.widgets.get("gold_schema"),
    api_key=env_config.api_key,
)
fixture_id_widget = dbutils.widgets.get("fixture_id").strip()
window_minutes = max(1, int(dbutils.widgets.get("window_minutes").strip() or "75"))
run_id = dbutils.widgets.get("run_id").strip() or f"inference-{utc_now_iso()}"
mode = dbutils.widgets.get("mode").strip().lower()
registered_model_prefix = dbutils.widgets.get("registered_model_prefix").strip()
model_alias = dbutils.widgets.get("model_alias").strip() or None
targets_widget = dbutils.widgets.get("targets").strip()
target_events = [t.strip() for t in targets_widget.split(",") if t.strip()] or list(DEFAULT_TARGET_COLUMNS)
logger = configure_json_logging(level=logging.INFO, logger_name="football_analytics.prematch_inference_predict")

feature_table = table_name(config, "gold", "fct_football__player_event_features")
prediction_table = table_name(config, "gold", PREDICTION_TABLE_NAME)

if fixture_id_widget:
    fixture_filter = f"fixture_id = {int(fixture_id_widget)}"
else:
    now_utc = datetime.now(timezone.utc)
    window_end = now_utc + timedelta(minutes=window_minutes)
    fixture_filter = (
        f"fixture_date_utc >= TIMESTAMP'{now_utc.isoformat()}' "
        f"AND fixture_date_utc <= TIMESTAMP'{window_end.isoformat()}'"
    )

inference_frame = spark.sql(
    f"""
    SELECT *
    FROM {feature_table}
    WHERE NOT is_completed_fixture
      AND {fixture_filter}
    """
).toPandas()

results = []
if inference_frame.empty:
    results.append({"status": "SKIPPED", "reason": "no inference rows in the trigger window"})
else:
    models = load_registered_event_models(
        registered_model_name_prefix=registered_model_prefix,
        target_events=target_events,
        alias=model_alias,
    )
    model_versions = {model.target_event: model.model_version for model in models}

    for fixture_id, fixture_rows in inference_frame.groupby("fixture_id"):
        confirmed = fixture_has_confirmed_starting_xi(fixture_rows)
        try:
            lineup_source = resolve_lineup_source(mode, lineups_confirmed=confirmed)
        except MissingLineupError as error:
            logger.warning(
                "inference_skipped_missing_lineups",
                extra={
                    "event": "inference_skipped_missing_lineups",
                    "run_id": run_id,
                    "fixture_id": int(fixture_id),
                    "reason": str(error),
                },
            )
            results.append({
                "fixture_id": int(fixture_id),
                "status": "SKIPPED",
                "reason": str(error),
            })
            continue

        prediction_set_id = deterministic_prediction_set_id(
            int(fixture_id),
            lineup_digest=lineup_signature(fixture_rows),
            model_versions=model_versions,
        )
        records = build_prediction_records(
            fixture_rows,
            models,
            prediction_set_id=prediction_set_id,
            prediction_run_id=run_id,
            lineup_source=lineup_source,
            feature_table_name=feature_table,
        )
        write_summary = write_player_event_predictions(
            spark,
            records,
            prediction_table=prediction_table,
        )
        results.append({
            "fixture_id": int(fixture_id),
            "status": "PREDICTED",
            "lineup_source": lineup_source,
            "prediction_set_id": prediction_set_id,
            "players": int(fixture_rows["player_id"].nunique()),
            "rows_written": write_summary["written"],
        })

display({
    "run_id": run_id,
    "mode": mode,
    "targets": target_events,
    "prediction_table": prediction_table,
    "results": results,
})
