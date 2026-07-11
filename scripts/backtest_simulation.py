"""Historical simulation-calibration backtest (ml_upgrade_backlog.md N6).

Replays a holdout season's completed fixtures as pseudo-pre-match
simulations and scores calibration: player/team interval coverage,
randomized-PIT uniformity, and leader-allocation hit rates. Logs one
consolidated artifact bundle to a model-free MLflow run.

Example:
    python scripts/backtest_simulation.py \
        --feature-table football_analytics.gold.fct_football__player_event_features \
        --registered-model-prefix football_analytics.gold.player_event \
        --season 2026
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from football_analytics.inference import load_registered_event_models
from football_analytics.simulation import SimulationConfig
from football_analytics.simulation_backtest import (
    score_simulation_backtest,
    simulate_completed_fixtures,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-table",
        default="football_analytics.gold.fct_football__player_event_features",
    )
    parser.add_argument(
        "--registered-model-prefix",
        default="football_analytics.gold.player_event",
    )
    parser.add_argument("--season", type=int, required=True, help="Holdout league_season to replay.")
    parser.add_argument("--model-alias", default=None)
    parser.add_argument("--n-sims", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument(
        "--max-fixtures",
        type=int,
        default=None,
        help="Optional cap for cheap smoke runs.",
    )
    parser.add_argument(
        "--team-goal-anchor-weight",
        type=float,
        default=0.0,
        help=(
            "FA-105 Elo goal anchor blend weight; sweep this over "
            "{0, 0.25, 0.5, 0.75, 1.0} to produce FA-106's fitting evidence "
            "for the anchor adoption (0.0 = anchor off, v1 behavior)."
        ),
    )
    parser.add_argument(
        "--player-dispersion-targets",
        default="",
        help=(
            "FA-108 per-player allocation dispersion: comma-separated volume "
            "targets (e.g. 'passes_total'). Empty = off (v1 allocation); run "
            "the N6 re-run with and without 'passes_total' to produce the "
            "adoption evidence for the passes coverage fix."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from pyspark.sql import SparkSession

    import mlflow

    try:
        mlflow.autolog(disable=True)
    except Exception:
        pass

    spark = SparkSession.builder.getOrCreate()
    feature_rows = (
        spark.table(args.feature_table)
        .where(f"is_completed_fixture AND league_season = {int(args.season)}")
        .toPandas()
    )
    if args.max_fixtures:
        keep = sorted(feature_rows["fixture_id"].unique())[: args.max_fixtures]
        feature_rows = feature_rows[feature_rows["fixture_id"].isin(keep)]

    models = load_registered_event_models(
        registered_model_name_prefix=args.registered_model_prefix,
        alias=args.model_alias,
    )
    player_dispersion_targets = tuple(
        part.strip()
        for part in args.player_dispersion_targets.split(",")
        if part.strip()
    )
    config = SimulationConfig(
        n_sims=args.n_sims,
        seed=args.seed,
        team_goal_anchor_weight=args.team_goal_anchor_weight,
        player_dispersion_targets=player_dispersion_targets,
    )
    results, skipped = simulate_completed_fixtures(feature_rows, models, config=config)
    report = score_simulation_backtest(results, feature_rows)

    # N6 acceptance flag: targets whose 80% player coverage leaves [0.70, 0.90]
    # feed the ADR 0005 revisit triggers.
    coverage = report["player_coverage"].copy()
    coverage["outside_acceptance_band"] = ~coverage["coverage"].between(0.70, 0.90)

    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    run_suffix = (
        f"-w{args.team_goal_anchor_weight:g}" if args.team_goal_anchor_weight else ""
    )
    if player_dispersion_targets:
        run_suffix += "-pd-" + "-".join(player_dispersion_targets)
    with mlflow.start_run(run_name=f"simulation-backtest-{args.season}{run_suffix}"):
        mlflow.log_param("mode", "simulation_backtest")
        mlflow.log_param("season", args.season)
        mlflow.log_param("n_sims", args.n_sims)
        mlflow.log_param("team_goal_anchor_weight", args.team_goal_anchor_weight)
        mlflow.log_param(
            "player_dispersion_targets",
            ",".join(player_dispersion_targets) or "(off)",
        )
        mlflow.log_param("fixtures_simulated", len(results))
        mlflow.log_param("fixtures_skipped", len(skipped))
        for row in coverage.itertuples():
            mlflow.log_metric(f"{row.target_event}_player_coverage", row.coverage)
            mlflow.log_metric(
                f"{row.target_event}_pit_max_abs_deviation", row.pit_max_abs_deviation
            )
        for row in report["allocation"].itertuples():
            if pd.notna(row.top1_hit_rate):
                mlflow.log_metric(f"{row.target_event}_top1_hit_rate", row.top1_hit_rate)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            coverage.to_csv(artifact_dir / "player_coverage.csv", index=False)
            report["team_coverage"].to_csv(artifact_dir / "team_coverage.csv", index=False)
            report["allocation"].to_csv(artifact_dir / "allocation.csv", index=False)
            skipped.to_csv(artifact_dir / "skipped_fixtures.csv", index=False)
            (artifact_dir / "report.json").write_text(
                json.dumps(
                    {
                        "player_coverage": coverage.to_dict(orient="records"),
                        "team_coverage": report["team_coverage"].to_dict(orient="records"),
                        "allocation": report["allocation"].to_dict(orient="records"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            mlflow.log_artifacts(str(artifact_dir), artifact_path="simulation_backtest")

    print(coverage.to_string(index=False))
    print(report["allocation"].to_string(index=False))
    if not skipped.empty:
        print(f"\nSkipped {len(skipped)} fixtures (see artifact for reasons)")


if __name__ == "__main__":
    main()
