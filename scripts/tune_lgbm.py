"""Per-target hyperparameter tuning under the rolling-origin harness (M3).

Optuna TPE minimizes cross-fold mean RPS per target; results are written to
version-controlled JSON files that `scripts/train_poisson_lgbm.py` loads by
default. The adoption gate is pre-committed: tuned params are marked
`adopted` only when they beat the default config on the same folds.

Requires the `tuning` extra: pip install football-analytics[tuning]

Example:
    python scripts/tune_lgbm.py \
        --feature-table football_analytics.gold.fct_football__player_event_features \
        --targets shots_total,passes_total --n-trials 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_analytics.ml_training import (
    DEFAULT_TARGET_COLUMNS,
    PoissonLightGBMConfig,
    tune_target_hyperparameters,
)
from train_poisson_lgbm import build_training_feature_frame


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-table",
        default="football_analytics.gold.fct_football__player_event_features",
    )
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGET_COLUMNS))
    parser.add_argument("--features", default=None)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--min-train-seasons", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="config/lgbm_params",
        help="Directory for per-target JSON results (version-controlled).",
    )
    parser.add_argument("--experiment-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    feature_frame = spark.table(args.feature_table).toPandas()
    training_frame = build_training_feature_frame(feature_frame)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for target in _parse_csv(args.targets) or DEFAULT_TARGET_COLUMNS:
        result = tune_target_hyperparameters(
            training_frame,
            target=target,
            feature_columns=_parse_csv(args.features),
            n_trials=args.n_trials,
            min_train_seasons=args.min_train_seasons,
            seed=args.seed,
            baseline_config=PoissonLightGBMConfig(),
        )
        (output_dir / f"{target}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        results.append(result)
        print(
            f"{target}: tuned RPS {result['cross_fold_rps_tuned']:.5f} vs "
            f"default {result['cross_fold_rps_default']:.5f} -> "
            f"{'ADOPTED' if result['adopted'] else 'kept default'}"
        )

    try:
        import mlflow

        try:
            mlflow.autolog(disable=True)
        except Exception:
            pass
        if args.experiment_name:
            mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_name="lgbm-tuning"):
            mlflow.log_param("mode", "hyperparameter_tuning")
            mlflow.log_param("n_trials", args.n_trials)
            mlflow.log_param("feature_table", args.feature_table)
            mlflow.log_dict({"results": results}, "tuning/study_summary.json")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
