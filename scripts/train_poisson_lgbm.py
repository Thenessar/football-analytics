"""Train Poisson LightGBM models from the gold player event feature table.

One exposure-offset Poisson LightGBM model is trained per target event
(business_logic.md §2/§13.3) against fct_football__player_event_features.
Training rows use actual games_minutes as the exposure offset; inference uses
expected_minutes via ExposurePoissonLightGBMModel.predict_mean(exposure=...).

Example:
    python scripts/train_poisson_lgbm.py \
        --feature-table football_analytics.gold.fct_football__player_event_features \
        --registered-model-prefix football_analytics.gold.player_event
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from football_analytics.ml_training import (
    DEFAULT_TARGET_COLUMNS,
    PoissonLightGBMConfig,
    add_model_interaction_features,
    load_tuned_configs,
    run_baseline_reference,
    run_rolling_origin_backtest,
    season_train_validation_split,
    temporal_train_validation_split,
    train_poisson_lightgbm_with_mlflow,
)


REQUIRED_ELO_FEATURE_COLUMNS = [
    "team_elo_general_pre",
    "opponent_elo_general_pre",
    "team_elo_attack_pre",
    "team_elo_defense_pre",
    "opponent_elo_attack_pre",
    "opponent_elo_defense_pre",
    "expected_goals_for_pre",
    "expected_goals_against_pre",
    "player_offensive_modifier_pre",
    "player_defensive_modifier_pre",
    "player_offensive_elo_pre",
    "player_defensive_elo_pre",
    "player_offensive_rating_pre",
    "player_defensive_rating_pre",
    "missed_fixture_count_pre",
    "team_lineup_attack_strength",
    "team_lineup_defense_strength",
]


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _feature_table_version(spark, table_name: str) -> str | None:
    """Returns the current Delta version of the feature table, if readable."""

    try:
        history = spark.sql(f"DESCRIBE HISTORY {table_name} LIMIT 1").collect()
        if history:
            return str(history[0]["version"])
    except Exception:
        pass
    return None


def build_training_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    require_elo: bool = True,
) -> pd.DataFrame:
    """Validates the dbt-materialized feature table used for model training.

    fct_football__player_event_features also carries inference rows for
    future fixtures (null labels, is_completed_fixture = false); training must
    only see completed rows with real exposure minutes.
    """

    frame = feature_frame
    if "is_completed_fixture" in frame.columns:
        frame = frame[frame["is_completed_fixture"].fillna(False).astype(bool)]
    if "games_minutes" in frame.columns:
        frame = frame[pd.to_numeric(frame["games_minutes"], errors="coerce").fillna(0) > 0]

    if not require_elo:
        return frame.copy()

    missing = [
        column for column in REQUIRED_ELO_FEATURE_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            "Feature table is missing dbt-materialized ELO columns: "
            + ", ".join(missing)
        )
    return add_model_interaction_features(frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-table",
        default="football_analytics.gold.fct_football__player_event_features",
        help="Fully-qualified Spark/Unity Catalog table containing training rows.",
    )
    parser.add_argument(
        "--targets",
        default=",".join(DEFAULT_TARGET_COLUMNS),
        help="Comma-separated count targets to train independently.",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Optional comma-separated feature override. Defaults to available built-ins.",
    )
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--run-name", default="hierarchical-elo-poisson-lightgbm")
    parser.add_argument(
        "--registered-model-prefix",
        default=None,
        help=(
            "Unity Catalog prefix for registered models; each target registers "
            "as <prefix>__<target> (e.g. football_analytics.gold.player_event"
            "__shots_total)."
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--validation-seasons",
        default=None,
        help=(
            "Comma-separated league seasons to validate on (train uses strictly "
            "earlier seasons). Overrides --validation-fraction."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Stop each target at its best validation iteration; 0 disables.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help=(
            "Rolling-origin backtest mode: train throwaway models per season "
            "fold, log cross-fold metric means/stds and a consolidated report "
            "artifact, register nothing (ml_upgrade_backlog.md L2)."
        ),
    )
    parser.add_argument(
        "--min-train-seasons",
        type=int,
        default=2,
        help="Backtest mode: earliest validation season must have this many prior seasons.",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help=(
            "Score the naive baselines (position-group rate, player L5) "
            "through the fold harness under run name 'baseline-reference'; "
            "trains and registers nothing (ml_upgrade_backlog.md L4)."
        ),
    )
    parser.add_argument(
        "--tuned-params-dir",
        default="config/lgbm_params",
        help=(
            "Directory of adopted per-target tuning results from "
            "scripts/tune_lgbm.py (ml_upgrade_backlog.md M3)."
        ),
    )
    parser.add_argument(
        "--ignore-tuned-params",
        action="store_true",
        help="Train every target with the shared CLI config only.",
    )
    return parser.parse_args()


def run_baselines_only_mode(args: "argparse.Namespace", training_frame: pd.DataFrame) -> None:
    """L4: log the skill-score denominators as an inspectable MLflow run."""

    import tempfile
    from pathlib import Path

    import mlflow

    try:
        mlflow.autolog(disable=True)
    except Exception:
        pass

    result = run_baseline_reference(
        training_frame,
        target_columns=_parse_csv(args.targets) or DEFAULT_TARGET_COLUMNS,
        min_train_seasons=args.min_train_seasons,
    )

    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="baseline-reference"):
        mlflow.log_param("mode", "baseline_reference")
        mlflow.log_param("min_train_seasons", args.min_train_seasons)
        mlflow.log_param("feature_table", args.feature_table)
        for row in result["summary"].itertuples():
            mlflow.log_metric(f"{row.target}_{row.baseline}_{row.metric}_mean", row.mean)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            result["report"].to_csv(artifact_dir / "baseline_report.csv", index=False)
            result["summary"].to_csv(artifact_dir / "baseline_summary.csv", index=False)
            mlflow.log_artifacts(str(artifact_dir), artifact_path="baselines")

    print(result["summary"].to_string(index=False))


def run_backtest_mode(args: "argparse.Namespace", training_frame: pd.DataFrame, config: PoissonLightGBMConfig) -> None:
    """Runs the L2 backtest and logs quota-safe aggregates to one MLflow run.

    Per-fold detail goes to artifacts (Working Agreement #4); only per-target
    cross-fold mean/std land as metrics. No models are logged or registered,
    so the per-model metric quota can never trip.
    """

    import tempfile
    from pathlib import Path

    import mlflow

    try:
        mlflow.autolog(disable=True)
    except Exception:
        pass

    result = run_rolling_origin_backtest(
        training_frame,
        target_columns=_parse_csv(args.targets) or DEFAULT_TARGET_COLUMNS,
        feature_columns=_parse_csv(args.features),
        config=config,
        min_train_seasons=args.min_train_seasons,
    )

    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=f"{args.run_name}-backtest"):
        mlflow.log_param("mode", "rolling_origin_backtest")
        mlflow.log_param("min_train_seasons", args.min_train_seasons)
        mlflow.log_param("feature_table", args.feature_table)
        for row in result["summary"].itertuples():
            mlflow.log_metric(f"{row.target}_{row.metric}_mean", row.mean)
            if row.folds > 1:
                mlflow.log_metric(f"{row.target}_{row.metric}_std", row.std)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            result["report"].to_csv(artifact_dir / "backtest_report.csv", index=False)
            result["summary"].to_csv(artifact_dir / "backtest_summary.csv", index=False)
            result["skipped"].to_csv(artifact_dir / "backtest_skipped.csv", index=False)
            (artifact_dir / "backtest_report.json").write_text(
                result["report"].to_json(orient="records"), encoding="utf-8"
            )
            mlflow.log_artifacts(str(artifact_dir), artifact_path="backtest")

    print(result["summary"].to_string(index=False))
    if not result["skipped"].empty:
        print("\nSkipped fold/target pairs:")
        print(result["skipped"].to_string(index=False))


def main() -> None:
    args = parse_args()

    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError("PySpark is required to load the Databricks feature table") from exc

    spark = SparkSession.builder.getOrCreate()

    if args.registered_model_prefix:
        # Registered models live in the Unity Catalog registry (see commit
        # "Configure MLflow to explicitly use Unity Catalog registry URI").
        import mlflow

        mlflow.set_registry_uri("databricks-uc")

    feature_frame = spark.table(args.feature_table).toPandas()
    training_frame = build_training_feature_frame(feature_frame)

    if args.baselines_only:
        run_baselines_only_mode(args, training_frame)
        return

    if args.backtest:
        config = PoissonLightGBMConfig(
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_child_samples=args.min_child_samples,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        run_backtest_mode(args, training_frame, config)
        return

    validation_seasons = _parse_csv(args.validation_seasons)
    if validation_seasons:
        train_df, validation_df = season_train_validation_split(
            training_frame,
            validation_seasons=[int(season) for season in validation_seasons],
        )
    else:
        train_df, validation_df = temporal_train_validation_split(
            training_frame,
            validation_fraction=args.validation_fraction,
        )

    config = PoissonLightGBMConfig(
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    per_target_configs = (
        {} if args.ignore_tuned_params else load_tuned_configs(args.tuned_params_dir)
    )
    if per_target_configs:
        print(f"Using tuned configs for: {sorted(per_target_configs)}")

    feature_table_version = _feature_table_version(spark, args.feature_table)
    result = train_poisson_lightgbm_with_mlflow(
        train_df,
        validation_df,
        target_columns=_parse_csv(args.targets) or DEFAULT_TARGET_COLUMNS,
        feature_columns=_parse_csv(args.features),
        config=config,
        per_target_configs=per_target_configs,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        registered_model_name_prefix=args.registered_model_prefix,
        run_tags={
            "feature_table_name": args.feature_table,
            "feature_table_version": feature_table_version or "unknown",
        },
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(json.dumps({"feature_columns": result["feature_columns"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
