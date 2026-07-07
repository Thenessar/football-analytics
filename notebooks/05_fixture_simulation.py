# Databricks notebook source
"""Monte Carlo fixture simulation (ml_upgrade_backlog.md N5, ADR 0005).

Consumes the ACTIVE prediction set written by notebook 04, simulates each
fixture n_sims times with the coherent sampling design, writes summary rows
to gold.sim_football__fixture_simulation (append + active flag), and displays
the full game view: per-player event table, team totals with intervals, and
the scoreline probability matrix.
"""
import json
import logging
from datetime import datetime, timedelta, timezone

from football_analytics.databricks.config import DatabricksPipelineConfig, load_config_from_env
from football_analytics.databricks.logging import configure_json_logging
from football_analytics.databricks.tables import table_name
from football_analytics.databricks_ingestion import utc_now_iso
from football_analytics.inference import PREDICTION_TABLE_NAME, load_team_dispersion_tags
from football_analytics.simulation import (
    SIMULATION_TABLE_NAME,
    SimulationConfig,
    SimulationInputError,
    build_simulation_inputs,
    deterministic_sim_set_id,
    simulate_fixture,
    summarize_simulation,
    write_fixture_simulation,
)

dbutils.widgets.text("fixture_id", "")
dbutils.widgets.text("window_minutes", "75")
dbutils.widgets.text("n_sims", "10000")
dbutils.widgets.text("seed", "7")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("catalog", "football_analytics")
dbutils.widgets.text("bronze_schema", "bronze")
dbutils.widgets.text("silver_schema", "silver")
dbutils.widgets.text("gold_schema", "gold")

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
n_sims = max(100, int(dbutils.widgets.get("n_sims").strip() or "10000"))
seed = int(dbutils.widgets.get("seed").strip() or "7")
run_id = dbutils.widgets.get("run_id").strip() or f"simulation-{utc_now_iso()}"
logger = configure_json_logging(level=logging.INFO, logger_name="football_analytics.fixture_simulation")

prediction_table = table_name(config, "gold", PREDICTION_TABLE_NAME)
simulation_table = table_name(config, "gold", SIMULATION_TABLE_NAME)

if fixture_id_widget:
    fixture_filter = f"fixture_id = {int(fixture_id_widget)}"
else:
    now_utc = datetime.now(timezone.utc)
    window_end = now_utc + timedelta(minutes=window_minutes)
    fixture_filter = (
        f"fixture_date_utc >= TIMESTAMP'{now_utc.isoformat()}' "
        f"AND fixture_date_utc <= TIMESTAMP'{window_end.isoformat()}'"
    )

active_predictions = spark.sql(
    f"""
    SELECT *
    FROM {prediction_table}
    WHERE is_active_prediction
      AND {fixture_filter}
    """
).toPandas()

sim_config = SimulationConfig(n_sims=n_sims, seed=seed)
results = []
if active_predictions.empty:
    results.append({"status": "SKIPPED", "reason": "no active prediction sets in the window"})
else:
    for fixture_id, fixture_rows in active_predictions.groupby("fixture_id"):
        try:
            # Team-total dispersion from the registry tags of the exact model
            # versions that produced this prediction set (cheap: no artifact
            # loads; unreadable tags fall back to Poisson totals).
            model_versions = {
                row.target_event: (row.model_name, row.model_version)
                for row in fixture_rows[["target_event", "model_name", "model_version"]]
                .drop_duplicates()
                .itertuples()
            }
            try:
                alpha_team = load_team_dispersion_tags(model_versions)
            except Exception as exc:
                logger.warning(
                    "alpha_team_tags_unavailable",
                    extra={"event": "alpha_team_tags_unavailable", "reason": str(exc)},
                )
                alpha_team = {}

            inputs = build_simulation_inputs(fixture_rows, alpha_team=alpha_team)
            sim_set_id = deterministic_sim_set_id(
                int(fixture_id),
                prediction_set_id=inputs.prediction_set_id,
                config=sim_config,
            )
            result = simulate_fixture(inputs, sim_config)
            summary = summarize_simulation(result, sim_set_id=sim_set_id)
            write_summary = write_fixture_simulation(
                spark, summary, simulation_table=simulation_table
            )
            results.append({
                "fixture_id": int(fixture_id),
                "status": "SIMULATED",
                "sim_set_id": sim_set_id,
                "n_sims": n_sims,
                "players": int(len(inputs.players)),
                "rows_written": write_summary["written"],
            })

            # Full game view (step: display). Per-player means across events,
            # team totals with 5-95% intervals, scoreline matrix.
            player_rows = summary[summary["entity_type"] == "player"]
            game_view = player_rows.pivot_table(
                index=["team_id", "entity_name", "position_group", "is_starting"],
                columns="target_event",
                values="sim_mean",
            ).round(3).reset_index()
            print(f"=== Fixture {fixture_id}: simulated player means ===")
            display(game_view)

            team_rows = summary[summary["entity_type"] == "team"][
                ["team_id", "target_event", "sim_mean", "p05", "p95", "p_ge_1"]
            ].round(3)
            print(f"=== Fixture {fixture_id}: team totals (mean, 5-95% interval) ===")
            display(team_rows)

            scoreline = summary[summary["entity_type"] == "fixture"]["score_matrix_json"].iloc[0]
            matrix = json.loads(scoreline)
            top_scores = sorted(matrix["matrix"].items(), key=lambda kv: -kv[1])[:10]
            print(f"=== Fixture {fixture_id}: most likely scorelines "
                  f"(team {matrix['team_a_id']} vs team {matrix['team_b_id']}) ===")
            display([{"scoreline": score, "probability": prob} for score, prob in top_scores])
        except SimulationInputError as error:
            logger.warning(
                "simulation_skipped_invalid_inputs",
                extra={
                    "event": "simulation_skipped_invalid_inputs",
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

display({
    "run_id": run_id,
    "simulation_table": simulation_table,
    "n_sims": n_sims,
    "seed": seed,
    "results": results,
})
