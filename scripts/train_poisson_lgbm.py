"""Train hierarchical-ELO Poisson LightGBM models from the gold feature table.

Example:
    python scripts/train_poisson_lgbm.py \
        --feature-table football_analytics.gold.fct_football__player_shot_features \
        --registered-model-prefix football_analytics.gold.player_prop_poisson_lgbm
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from football_analytics.elo import (
    assemble_hierarchical_feature_frame,
    build_player_elo_history,
    build_team_elo_history,
)
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


def _normalize_ranking_frame(rankings: pd.DataFrame | None) -> pd.DataFrame | None:
    if rankings is None:
        return None
    renamed = rankings.copy()
    column_map = {}
    if "Team" in renamed.columns and "team_name" not in renamed.columns:
        column_map["Team"] = "team_name"
    if "Raiting" in renamed.columns and "rating" not in renamed.columns:
        column_map["Raiting"] = "rating"
    if "Rating" in renamed.columns and "rating" not in renamed.columns:
        column_map["Rating"] = "rating"
    return renamed.rename(columns=column_map)


def build_training_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    fixtures_frame: pd.DataFrame | None = None,
    rankings_frame: pd.DataFrame | None = None,
    decay_alpha: float = 0.85,
    enable_elo: bool = True,
) -> pd.DataFrame:
    """Optionally enriches dbt features with chronological ELO features."""

    if not enable_elo:
        return feature_frame
    if fixtures_frame is None or fixtures_frame.empty:
        raise ValueError(
            "ELO is enabled but no fixture history was loaded. "
            "Pass --fixture-table or use --disable-elo explicitly."
        )

    rankings_frame = _normalize_ranking_frame(rankings_frame)
    team_elo_history = build_team_elo_history(fixtures_frame, rankings_frame)
    player_elo_history = build_player_elo_history(
        feature_frame,
        team_elo_history,
        decay_alpha=decay_alpha,
    )
    enriched = assemble_hierarchical_feature_frame(
        feature_frame,
        team_elo_history,
        player_elo_history,
    )

    elo_columns = [
        "team_elo_general_pre",
        "opponent_elo_general_pre",
        "team_elo_attack_pre",
        "team_elo_defense_pre",
        "opponent_elo_attack_pre",
        "opponent_elo_defense_pre",
        "player_offensive_modifier_pre",
        "player_defensive_modifier_pre",
    ]
    missing = [column for column in elo_columns if column not in enriched.columns]
    if missing:
        raise ValueError(f"ELO enrichment failed; missing columns: {', '.join(missing)}")
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-table",
        default="football_analytics.gold.fct_football__player_shot_features",
        help="Fully-qualified Spark/Unity Catalog table containing training rows.",
    )
    parser.add_argument(
        "--fixture-table",
        default="football_analytics.silver.stg_football__fixtures",
        help="Fully-qualified fixture table used to build chronological team ELO.",
    )
    parser.add_argument(
        "--ranking-table",
        default="football_analytics.gold.dim_football__rating_baseline",
        help="Optional FIFA ranking baseline table used to initialize team ELO.",
    )
    parser.add_argument(
        "--disable-elo",
        action="store_true",
        help="Train from the feature table without building or merging ELO features.",
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
    fixtures_frame = None
    rankings_frame = None
    if not args.disable_elo:
        fixtures_frame = spark.table(args.fixture_table).toPandas()
        if args.ranking_table:
            rankings_frame = spark.table(args.ranking_table).toPandas()
    training_frame = build_training_feature_frame(
        feature_frame,
        fixtures_frame=fixtures_frame,
        rankings_frame=rankings_frame,
        decay_alpha=args.decay_alpha,
        enable_elo=not args.disable_elo,
    )
    train_df, validation_df = temporal_train_validation_split(
        training_frame,
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
    print(json.dumps({"feature_columns": result["feature_columns"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
