"""Train hierarchical-ELO Poisson LightGBM models from the gold feature table.

Example:
    python scripts/train_poisson_lgbm.py \
        --feature-table football_analytics.gold.fct_football__player_shot_features \
        --registered-model-prefix football_analytics.gold.player_prop_poisson_lgbm
"""

from __future__ import annotations

import argparse
import json

from football_analytics.ml_training import (
    DEFAULT_TARGET_COLUMNS,
    PoissonLightGBMConfig,
    temporal_train_validation_split,
    train_poisson_lightgbm_with_mlflow,
)


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-table",
        default="football_analytics.gold.fct_football__player_shot_features",
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
    parser.add_argument("--registered-model-prefix", default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=80)
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--decay-alpha", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError("PySpark is required to load the Databricks feature table") from exc

    spark = SparkSession.builder.getOrCreate()
    feature_frame = spark.table(args.feature_table).toPandas()
    train_df, validation_df = temporal_train_validation_split(
        feature_frame,
        validation_fraction=args.validation_fraction,
    )

    config = PoissonLightGBMConfig(
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        num_boost_round=args.num_boost_round,
        decay_alpha=args.decay_alpha,
    )
    result = train_poisson_lightgbm_with_mlflow(
        train_df,
        validation_df,
        target_columns=_parse_csv(args.targets) or DEFAULT_TARGET_COLUMNS,
        feature_columns=_parse_csv(args.features),
        config=config,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        registered_model_name_prefix=args.registered_model_prefix,
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
