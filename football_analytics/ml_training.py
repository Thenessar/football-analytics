"""Poisson LightGBM training helpers with MLflow tracking.

LightGBM, MLflow, SHAP, and Matplotlib are imported lazily so the core package
and unit tests remain usable in lightweight environments. Databricks jobs should
install the `modeling` optional dependency group before calling
`train_poisson_lightgbm_with_mlflow`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence
import inspect
import json
import math
import pickle
import tempfile

import numpy as np
import pandas as pd

# Canonical scoring primitives live in evaluation.py (ml_upgrade_backlog.md
# L1); poisson_log_loss is re-exported here for backward compatibility.
from football_analytics.evaluation import (  # noqa: F401
    dispersion_index,
    poisson_log_loss,
    ranked_probability_score,
    validation_metric_suite,
)


# All 11 target events from business_logic.md §2, trained independently
# against fct_football__player_event_features.
DEFAULT_TARGET_COLUMNS = [
    "offsides",
    "shots_total",
    "shots_on",
    "goals_total",
    "goals_assists",
    "goals_saves",
    "passes_total",
    "fouls_drawn",
    "fouls_committed",
    "cards_yellow",
    "cards_red",
]

# Targets that are structural zeros outside a position group (§13.3):
# goalkeepers are the only players who can record saves.
GOALKEEPER_ONLY_TARGETS = frozenset({"goals_saves"})

# games_minutes is intentionally NOT a feature: it is post-match information
# (§13.4). Minutes enter the model through the exposure offset instead —
# actual games_minutes at training time, expected_minutes at inference time.
DEFAULT_LIGHTGBM_FEATURES = [
    "is_starter",
    "is_starting",
    "was_substitute",
    "appearances_l5_count",
    "minutes_l5",
    "offsides_l5_p90",
    "shots_total_l5_p90",
    "shots_on_l5_p90",
    "goals_total_l5_p90",
    "goals_assists_l5_p90",
    "goals_saves_l5_p90",
    "passes_total_l5_p90",
    "fouls_drawn_l5_p90",
    "fouls_committed_l5_p90",
    "cards_yellow_l5_p90",
    "cards_red_l5_p90",
    "dribbles_attempts_l5_p90",
    "tackles_interceptions_l5_p90",
    "team_possession_l5_avg",
    "opponent_possession_l5_avg",
    "expected_possession_share",
    "team_passes_l5_per_match",
    "opponent_passes_allowed_l5",
    "team_shots_l5",
    "opponent_shots_allowed_l5",
    "team_fouls_l5",
    "opponent_fouls_drawn_allowed_l5",
    "elo_expected_score",
    "elo_possession_interaction",
    "formation_possession_profile",
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
    "game_importance_scalar",
    "game_importance_l5",
    "opponent_strength_adjustment",
    "defensive_containment_rating",
    "opponent_defensive_elo_l10",
    "team_elo_general_diff",
    "team_attack_vs_opp_defense",
    "team_defense_vs_opp_attack",
    "player_attack_vs_opp_defense",
    "player_defense_vs_opp_attack",
    "lineup_attack_vs_opp_defense",
    "lineup_defense_vs_opp_attack",
    "player_attack_delta_vs_team",
    "player_defense_delta_vs_team",
    "lineup_attack_delta_vs_team",
    "lineup_defense_delta_vs_team",
    "formation_row",
    "formation_column",
    "formation_win_rate_pre",
    "formation_matchup_win_rate_pre",
    "formation_count_pre",
    "formation_matchup_count_pre",
]


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    """Returns values once, keeping the first occurrence order."""

    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def add_model_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Adds leakage-safe matchup features from dbt-materialized ELO columns."""

    out = frame.copy()

    def has(*columns: str) -> bool:
        return all(column in out.columns for column in columns)

    if has("team_elo_general_pre", "opponent_elo_general_pre"):
        out["team_elo_general_diff"] = (
            out["team_elo_general_pre"] - out["opponent_elo_general_pre"]
        )
    if has("team_elo_attack_pre", "opponent_elo_defense_pre"):
        out["team_attack_vs_opp_defense"] = (
            out["team_elo_attack_pre"] - out["opponent_elo_defense_pre"]
        )
    if has("team_elo_defense_pre", "opponent_elo_attack_pre"):
        out["team_defense_vs_opp_attack"] = (
            out["team_elo_defense_pre"] - out["opponent_elo_attack_pre"]
        )
    if has("player_offensive_elo_pre", "opponent_elo_defense_pre"):
        out["player_attack_vs_opp_defense"] = (
            out["player_offensive_elo_pre"] - out["opponent_elo_defense_pre"]
        )
    if has("player_defensive_elo_pre", "opponent_elo_attack_pre"):
        out["player_defense_vs_opp_attack"] = (
            out["player_defensive_elo_pre"] - out["opponent_elo_attack_pre"]
        )
    if has("team_lineup_attack_strength", "opponent_elo_defense_pre"):
        out["lineup_attack_vs_opp_defense"] = (
            out["team_lineup_attack_strength"] - out["opponent_elo_defense_pre"]
        )
    if has("team_lineup_defense_strength", "opponent_elo_attack_pre"):
        out["lineup_defense_vs_opp_attack"] = (
            out["team_lineup_defense_strength"] - out["opponent_elo_attack_pre"]
        )
    if has("player_offensive_elo_pre", "team_elo_attack_pre"):
        out["player_attack_delta_vs_team"] = (
            out["player_offensive_elo_pre"] - out["team_elo_attack_pre"]
        )
    if has("player_defensive_elo_pre", "team_elo_defense_pre"):
        out["player_defense_delta_vs_team"] = (
            out["player_defensive_elo_pre"] - out["team_elo_defense_pre"]
        )
    if has("team_lineup_attack_strength", "team_elo_attack_pre"):
        out["lineup_attack_delta_vs_team"] = (
            out["team_lineup_attack_strength"] - out["team_elo_attack_pre"]
        )
    if has("team_lineup_defense_strength", "team_elo_defense_pre"):
        out["lineup_defense_delta_vs_team"] = (
            out["team_lineup_defense_strength"] - out["team_elo_defense_pre"]
        )

    return out


@dataclass
class PoissonLightGBMConfig:
    """Training parameters logged to MLflow and reused at inference."""

    learning_rate: float = 0.025
    num_leaves: int = 31
    min_child_samples: int = 80
    num_boost_round: int = 800
    early_stopping_rounds: int = 50
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 1
    lambda_l2: float = 0.0
    random_state: int = 42
    lightgbm_extra_params: Mapping[str, Any] = field(default_factory=dict)

    def to_lightgbm_params(self) -> Dict[str, Any]:
        params = {
            "objective": "poisson",
            "metric": "poisson",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "seed": self.random_state,
            "verbosity": -1,
        }
        params.update(dict(self.lightgbm_extra_params))
        return params

    def to_mlflow_params(self, feature_columns: Sequence[str]) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "feature_columns": json.dumps(list(feature_columns)),
        }


@dataclass
class ExposurePoissonLightGBMModel:
    """Serializable wrapper that applies the exposure offset at prediction time."""

    booster: Any
    feature_columns: list[str]
    target_column: str
    exposure_column: str = "games_minutes"
    goalkeeper_only: bool = False

    def predict_mean(self, frame: pd.DataFrame, exposure: Optional[Iterable[float]] = None) -> np.ndarray:
        X = frame[self.feature_columns]
        if exposure is None:
            exposure_values = exposure_from_minutes(frame[self.exposure_column])
        else:
            exposure_values = np.asarray(list(exposure), dtype=float)
        raw_rate = self.booster.predict(X, raw_score=True)
        return np.exp(raw_rate + np.log(np.clip(exposure_values, 1e-6, None)))

    def predict_distribution(
        self,
        frame: pd.DataFrame,
        *,
        thresholds: Sequence[int] = (1, 2),
    ) -> list[Dict[str, float]]:
        means = self.predict_mean(frame)
        return [count_threshold_probabilities(mean, thresholds=thresholds) for mean in means]


def exposure_from_minutes(minutes: Iterable[float]) -> np.ndarray:
    """Converts minutes played into positive 90-minute exposure units."""

    if isinstance(minutes, pd.DataFrame):
        if minutes.shape[1] != 1:
            raise ValueError(
                "Exposure input must be 1-dimensional; "
                f"received DataFrame with shape {minutes.shape}"
            )
        minutes = minutes.iloc[:, 0]
    values = pd.to_numeric(pd.Series(minutes), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return np.clip(values / 90.0, 1e-6, None)


def evaluate_count_predictions(
    y_true: Iterable[float],
    y_mean: Iterable[float],
) -> Dict[str, float]:
    y = np.asarray(list(y_true), dtype=float)
    mu = np.asarray(list(y_mean), dtype=float)
    return {
        "poisson_logloss": poisson_log_loss(y, mu),
        "mae": float(np.mean(np.abs(y - mu))),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
    }


def count_threshold_probabilities(
    mean_count: float,
    *,
    thresholds: Sequence[int] = (1, 2),
) -> Dict[str, float]:
    """Returns Poisson mean and P(count >= k) threshold probabilities."""

    mu = max(float(mean_count), 0.0)
    probabilities = {"mean": mu}
    for threshold in thresholds:
        if threshold <= 0:
            probabilities[f"p_ge_{threshold}"] = 1.0
            continue
        cumulative = 0.0
        for i in range(threshold):
            cumulative += math.exp(-mu) * (mu ** i) / math.factorial(i)
        probabilities[f"p_ge_{threshold}"] = float(np.clip(1.0 - cumulative, 0.0, 1.0))
    return probabilities


def filter_rows_for_target(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Restricts rows for position-gated targets (business_logic.md §13.3).

    goals_saves models train on goalkeepers only; non-goalkeepers are a
    structural zero handled outside the model at inference time (see
    docs/adr/0003-sparse-target-handling.md).
    """

    if target_column not in GOALKEEPER_ONLY_TARGETS:
        return frame
    if "is_goalkeeper" in frame.columns:
        mask = frame["is_goalkeeper"].fillna(False).astype(bool)
        return frame[mask]
    if "position_group" in frame.columns:
        return frame[frame["position_group"] == "G"]
    return frame


def _select_available_features(
    frame: pd.DataFrame,
    feature_columns: Optional[Sequence[str]],
) -> list[str]:
    requested = _unique_preserve_order(feature_columns or DEFAULT_LIGHTGBM_FEATURES)
    selected = [column for column in requested if column in frame.columns]
    if not selected:
        raise ValueError("No requested feature columns exist in the training frame")
    return selected


def _prepare_xy(
    frame: pd.DataFrame,
    *,
    target_column: str,
    feature_columns: Sequence[str],
    exposure_column: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    selected_features = _unique_preserve_order(feature_columns)
    required_columns = _unique_preserve_order([target_column, exposure_column, *selected_features])
    missing = [column for column in required_columns if column not in frame]
    if missing:
        raise ValueError(f"Training frame is missing columns: {', '.join(missing)}")

    subset = frame[required_columns].copy()
    subset = subset.loc[:, ~subset.columns.duplicated()].copy()
    subset = subset.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Nullable booleans (e.g. is_starting from Spark) arrive as object dtype;
    # LightGBM needs numeric inputs.
    X = subset[selected_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = subset[target_column].clip(lower=0.0).to_numpy(dtype=float)
    exposure = exposure_from_minutes(subset[exposure_column])
    return X, y, exposure


def _log_feature_importance_artifacts(
    *,
    booster: Any,
    feature_columns: Sequence[str],
    target_column: str,
    artifact_dir: Path,
) -> None:
    importance = pd.DataFrame({
        "feature": list(feature_columns),
        "importance_gain": booster.feature_importance(importance_type="gain"),
        "importance_split": booster.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    importance_path = artifact_dir / f"{target_column}_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    try:
        import matplotlib.pyplot as plt

        top = importance.head(30).sort_values("importance_gain", ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.25 * len(top))))
        ax.barh(top["feature"], top["importance_gain"])
        ax.set_title(f"{target_column} feature importance")
        ax.set_xlabel("Gain")
        fig.tight_layout()
        fig.savefig(artifact_dir / f"{target_column}_feature_importance.png", dpi=160)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        (artifact_dir / f"{target_column}_feature_importance_plot_skipped.txt").write_text(
            f"Feature importance plot skipped: {exc}",
            encoding="utf-8",
        )


def _log_shap_artifacts(
    *,
    model: ExposurePoissonLightGBMModel,
    training_frame: pd.DataFrame,
    target_column: str,
    artifact_dir: Path,
    sample_rows: int = 500,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import shap

        sample = training_frame[model.feature_columns].head(sample_rows)
        explainer = shap.TreeExplainer(model.booster)
        shap_values = explainer.shap_values(sample)
        shap.summary_plot(shap_values, sample, show=False, max_display=30)
        plt.tight_layout()
        plt.savefig(artifact_dir / f"{target_column}_shap_summary.png", dpi=160)
        plt.close()
    except Exception as exc:  # pragma: no cover - optional SHAP dependency
        (artifact_dir / f"{target_column}_shap_skipped.txt").write_text(
            f"SHAP summary skipped: {exc}",
            encoding="utf-8",
        )


def _fit_poisson_booster(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    train_exposure: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    valid_exposure: np.ndarray,
    config: "PoissonLightGBMConfig",
    log_period: Optional[int] = 100,
) -> Any:
    """Trains one exposure-offset Poisson booster (shared by training + backtest).

    The LightGBM Dataset ``init_score`` is set to log(exposure), so the
    booster learns a per-90 log intensity while labels stay raw match counts.
    """

    import lightgbm as lgb

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        init_score=np.log(train_exposure),
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        init_score=np.log(valid_exposure),
        reference=train_set,
        free_raw_data=False,
    )
    callbacks = []
    if log_period:
        callbacks.append(lgb.log_evaluation(period=log_period))
    if config.early_stopping_rounds and config.early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(config.early_stopping_rounds, verbose=False))
    return lgb.train(
        config.to_lightgbm_params(),
        train_set,
        num_boost_round=config.num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "validation"],
        callbacks=callbacks,
    )


def _lightgbm_log_model_kwargs(
    log_model_fn: Any,
    *,
    model_name: str,
    registered_model_name: str,
    input_example: pd.DataFrame,
    signature: Any,
) -> Dict[str, Any]:
    """Builds MLflow LightGBM logging kwargs across MLflow 2.x/3.x APIs."""

    parameters = inspect.signature(log_model_fn).parameters
    name_arg = "name" if "name" in parameters else "artifact_path"
    return {
        name_arg: model_name,
        "registered_model_name": registered_model_name,
        "input_example": input_example,
        "signature": signature,
    }


def train_poisson_lightgbm_with_mlflow(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
    feature_columns: Optional[Sequence[str]] = None,
    exposure_column: str = "games_minutes",
    config: Optional[PoissonLightGBMConfig] = None,
    experiment_name: Optional[str] = None,
    run_name: str = "hierarchical-elo-poisson-lightgbm",
    registered_model_name_prefix: Optional[str] = None,
    run_tags: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Trains one exposure-offset Poisson LightGBM model per target.

    The LightGBM Dataset `init_score` is set to log(exposure), causing the
    booster to learn a per-90 log intensity while metrics are evaluated on
    exposure-scaled match counts.

    Registered model names map 1:1 to the target event as
    `<registered_model_name_prefix>__<target>` (e.g. a prefix of
    `catalog.schema.player_event` registers `player_event__shots_total`).
    `run_tags` should carry feature-table lineage (`feature_table_name`,
    `feature_table_version`) so prediction rows can reference them (Epic H).
    """

    try:
        import lightgbm as lgb
        import mlflow
        from mlflow.models import infer_signature
    except ImportError as exc:  # pragma: no cover - exercised in Databricks
        raise RuntimeError(
            "LightGBM and MLflow are required for training. Install the "
            "`football-analytics[modeling]` optional dependencies."
        ) from exc

    cfg = config or PoissonLightGBMConfig()
    selected_features = _select_available_features(train_df, feature_columns)

    # Databricks enables MLflow autologging by default, which records the eval
    # metric at every boosting iteration against each logged model and blows
    # the free-tier cap of 1000 metrics per logged model. The Databricks
    # autologging layer only honors the GLOBAL disable (flavor-specific
    # mlflow.lightgbm.autolog(disable=True) gets overridden by it).
    # Everything needed is logged explicitly below.
    try:
        mlflow.autolog(disable=True)
    except Exception:
        pass
    try:
        mlflow.lightgbm.autolog(disable=True)
    except Exception:
        pass

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    trained: Dict[str, Any] = {
        "models": {},
        "metrics": {},
        "registered_models": {},
        "feature_columns": selected_features,
    }
    registration_queue: list[Dict[str, Any]] = []

    with mlflow.start_run(run_name=run_name) as training_run:
        training_run_id = training_run.info.run_id
        mlflow.log_params(cfg.to_mlflow_params(selected_features))
        mlflow.log_param("target_columns", json.dumps(list(target_columns)))
        mlflow.log_param("exposure_column", exposure_column)
        if run_tags:
            mlflow.set_tags({key: str(value) for key, value in dict(run_tags).items()})

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)

            for target in target_columns:
                target_train_df = filter_rows_for_target(train_df, target)
                target_valid_df = filter_rows_for_target(validation_df, target)
                if target_train_df.empty or target_valid_df.empty:
                    mlflow.log_param(
                        f"{target}_skipped",
                        "no rows after position-group filtering",
                    )
                    continue
                mlflow.log_param(f"{target}_train_rows", len(target_train_df))
                mlflow.log_param(f"{target}_validation_rows", len(target_valid_df))

                X_train, y_train, train_exposure = _prepare_xy(
                    target_train_df,
                    target_column=target,
                    feature_columns=selected_features,
                    exposure_column=exposure_column,
                )
                X_valid, y_valid, valid_exposure = _prepare_xy(
                    target_valid_df,
                    target_column=target,
                    feature_columns=selected_features,
                    exposure_column=exposure_column,
                )

                booster = _fit_poisson_booster(
                    X_train=X_train,
                    y_train=y_train,
                    train_exposure=train_exposure,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    valid_exposure=valid_exposure,
                    config=cfg,
                )
                # Each target keeps its best-validation iteration (LightGBM
                # predicts with best_iteration by default after early stop).
                mlflow.log_param(
                    f"{target}_best_iteration",
                    booster.best_iteration or cfg.num_boost_round,
                )
                model = ExposurePoissonLightGBMModel(
                    booster=booster,
                    feature_columns=selected_features,
                    target_column=target,
                    exposure_column=exposure_column,
                    goalkeeper_only=target in GOALKEEPER_ONLY_TARGETS,
                )

                train_mu = np.exp(
                    booster.predict(X_train, raw_score=True) + np.log(train_exposure)
                )
                valid_mu = np.exp(
                    booster.predict(X_valid, raw_score=True) + np.log(valid_exposure)
                )
                train_metrics = evaluate_count_predictions(y_train, train_mu)
                valid_metrics = evaluate_count_predictions(y_valid, valid_mu)

                for metric_name, metric_value in train_metrics.items():
                    mlflow.log_metric(f"{target}_train_{metric_name}", metric_value)
                for metric_name, metric_value in valid_metrics.items():
                    mlflow.log_metric(f"{target}_validation_{metric_name}", metric_value)

                # L-series metric suite (ml_upgrade_backlog.md L3): proper
                # scoring, calibration, dispersion, baseline skill, and
                # within-fixture ranking. Scalars go to MLflow metrics;
                # reliability tables go to artifacts (Working Agreement #4).
                suite = validation_metric_suite(
                    train_frame=target_train_df,
                    valid_frame=target_valid_df,
                    y_valid=y_valid,
                    mu_valid=valid_mu,
                    target=target,
                    train_exposure=train_exposure,
                    valid_exposure=valid_exposure,
                )
                for metric_name, metric_value in suite["metrics"].items():
                    if np.isfinite(metric_value):
                        mlflow.log_metric(
                            f"{target}_validation_{metric_name}", float(metric_value)
                        )
                for artifact_name, table in suite["artifacts"].items():
                    if not table.empty:
                        table.to_csv(
                            artifact_dir / f"{target}_{artifact_name}.csv", index=False
                        )

                # Chronological-evaluation harness (F4): report validation
                # metrics per position group so miscalibrated segments are
                # visible (position-specific models stay deferred, ADR 0003).
                # Basic metrics stay as MLflow metrics for continuity; the
                # extended per-group breakdown goes to one CSV artifact.
                if "position_group" in target_valid_df.columns:
                    group_values = target_valid_df["position_group"].fillna("unknown").to_numpy()
                    group_rows = []
                    for group in sorted(set(group_values)):
                        group_mask = group_values == group
                        if not group_mask.any():
                            continue
                        group_metrics = evaluate_count_predictions(
                            y_valid[group_mask], valid_mu[group_mask]
                        )
                        for metric_name, metric_value in group_metrics.items():
                            mlflow.log_metric(
                                f"{target}_validation_{metric_name}_pos_{group}",
                                metric_value,
                            )
                        group_rows.append({
                            "position_group": group,
                            "rows": int(group_mask.sum()),
                            **group_metrics,
                            "rps": ranked_probability_score(
                                y_valid[group_mask], valid_mu[group_mask]
                            ),
                            "dispersion_index": dispersion_index(
                                y_valid[group_mask], valid_mu[group_mask]
                            ),
                        })
                    if group_rows:
                        pd.DataFrame(group_rows).to_csv(
                            artifact_dir / f"{target}_position_group_metrics.csv",
                            index=False,
                        )

                _log_feature_importance_artifacts(
                    booster=booster,
                    feature_columns=selected_features,
                    target_column=target,
                    artifact_dir=artifact_dir,
                )
                _log_shap_artifacts(
                    model=model,
                    training_frame=target_train_df,
                    target_column=target,
                    artifact_dir=artifact_dir,
                )

                model_path = artifact_dir / f"{target}_poisson_lgbm.pkl"
                with model_path.open("wb") as handle:
                    pickle.dump(model, handle)

                if registered_model_name_prefix:
                    registered_model_name = f"{registered_model_name_prefix}__{target}"
                    mlflow.log_param(f"{target}_registered_model_name", registered_model_name)
                    input_example = X_train.head(min(5, len(X_train))).copy()
                    model_output_example = booster.predict(input_example)
                    signature = infer_signature(input_example, model_output_example)
                    registration_queue.append({
                        "target": target,
                        "booster": booster,
                        "registered_model_name": registered_model_name,
                        "input_example": input_example,
                        "signature": signature,
                    })
                trained["models"][target] = model
                trained["metrics"][target] = {
                    "train": train_metrics,
                    "validation": valid_metrics,
                    "validation_suite": suite["metrics"],
                }

            mlflow.log_artifacts(str(artifact_dir), artifact_path="model_artifacts")

    # Register models OUTSIDE the training run, one clean run per target.
    # Databricks MLflow links run metrics to the run's logged models and the
    # free tier caps 1000 metrics per logged model; a registration run that
    # logs no metrics can never trip that quota. Params link back to the
    # training run for lineage.
    for entry in registration_queue:
        with mlflow.start_run(run_name=f"register__{entry['target']}"):
            mlflow.log_param("training_run_id", training_run_id)
            mlflow.log_param("target_event", entry["target"])
            mlflow.lightgbm.log_model(
                entry["booster"],
                **_lightgbm_log_model_kwargs(
                    mlflow.lightgbm.log_model,
                    model_name=f"{entry['target']}_lightgbm_booster",
                    registered_model_name=entry["registered_model_name"],
                    input_example=entry["input_example"],
                    signature=entry["signature"],
                ),
            )
        trained["registered_models"][entry["target"]] = entry["registered_model_name"]

    return trained


def season_train_validation_split(
    frame: pd.DataFrame,
    *,
    validation_seasons: Sequence[int],
    season_column: str = "league_season",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological season split: train on seasons N-k..N-1, validate on N.

    Training rows are strictly earlier than the earliest validation season;
    rows from seasons after the validation set are dropped entirely so future
    data can never appear on either side.
    """

    seasons = {int(season) for season in validation_seasons}
    if not seasons:
        raise ValueError("validation_seasons must not be empty")
    if season_column not in frame:
        raise ValueError(f"Missing season column: {season_column}")

    season_values = pd.to_numeric(frame[season_column], errors="coerce")
    earliest_validation = min(seasons)
    train = frame[season_values < earliest_validation].copy()
    validation = frame[season_values.isin(seasons)].copy()
    if train.empty:
        raise ValueError("No training rows earlier than the validation seasons")
    if validation.empty:
        raise ValueError("No rows found for the requested validation seasons")
    return train, validation


def run_rolling_origin_backtest(
    frame: pd.DataFrame,
    *,
    target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
    feature_columns: Optional[Sequence[str]] = None,
    exposure_column: str = "games_minutes",
    config: Optional[PoissonLightGBMConfig] = None,
    season_column: str = "league_season",
    min_train_seasons: int = 2,
    min_train_rows: int = 50,
) -> Dict[str, pd.DataFrame]:
    """Expanding-window chronological backtest (ml_upgrade_backlog.md L2).

    Trains one throwaway booster per (validation season, target) — nothing is
    registered — and scores each with the full L-series metric suite. Returns:

    - ``report``: long frame, one row per season/target/metric.
    - ``summary``: per target/metric mean and std across folds — the numbers
      every Epic M adoption decision must cite.
    - ``skipped``: season/target pairs that could not be scored and why.
    """

    from football_analytics.evaluation import rolling_origin_folds, validation_metric_suite

    cfg = config or PoissonLightGBMConfig()
    selected_features = _select_available_features(frame, feature_columns)

    report_rows: list[Dict[str, Any]] = []
    skipped_rows: list[Dict[str, Any]] = []

    for season, train_df, valid_df in rolling_origin_folds(
        frame, season_column=season_column, min_train_seasons=min_train_seasons
    ):
        for target in target_columns:
            target_train = filter_rows_for_target(train_df, target)
            target_valid = filter_rows_for_target(valid_df, target)
            if len(target_train) < min_train_rows or target_valid.empty:
                skipped_rows.append({
                    "season": season,
                    "target": target,
                    "reason": f"insufficient rows (train={len(target_train)}, valid={len(target_valid)})",
                })
                continue

            X_train, y_train, train_exposure = _prepare_xy(
                target_train,
                target_column=target,
                feature_columns=selected_features,
                exposure_column=exposure_column,
            )
            if y_train.sum() <= 0:
                skipped_rows.append({
                    "season": season,
                    "target": target,
                    "reason": "no positive labels in training fold",
                })
                continue
            X_valid, y_valid, valid_exposure = _prepare_xy(
                target_valid,
                target_column=target,
                feature_columns=selected_features,
                exposure_column=exposure_column,
            )

            booster = _fit_poisson_booster(
                X_train=X_train,
                y_train=y_train,
                train_exposure=train_exposure,
                X_valid=X_valid,
                y_valid=y_valid,
                valid_exposure=valid_exposure,
                config=cfg,
                log_period=None,
            )
            mu_valid = np.exp(booster.predict(X_valid, raw_score=True) + np.log(valid_exposure))

            metrics = dict(evaluate_count_predictions(y_valid, mu_valid))
            suite = validation_metric_suite(
                train_frame=target_train,
                valid_frame=target_valid,
                y_valid=y_valid,
                mu_valid=mu_valid,
                target=target,
                train_exposure=train_exposure,
                valid_exposure=valid_exposure,
            )
            metrics.update(suite["metrics"])
            for metric_name, metric_value in metrics.items():
                report_rows.append({
                    "season": season,
                    "target": target,
                    "metric": metric_name,
                    "value": float(metric_value),
                })

    report = pd.DataFrame(report_rows, columns=["season", "target", "metric", "value"])
    if report.empty:
        summary = pd.DataFrame(columns=["target", "metric", "mean", "std", "folds"])
    else:
        summary = (
            report.groupby(["target", "metric"])["value"]
            .agg(mean="mean", std="std", folds="count")
            .reset_index()
        )
    return {
        "report": report,
        "summary": summary,
        "skipped": pd.DataFrame(skipped_rows, columns=["season", "target", "reason"]),
    }


def run_baseline_reference(
    frame: pd.DataFrame,
    *,
    target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
    exposure_column: str = "games_minutes",
    season_column: str = "league_season",
    min_train_seasons: int = 2,
) -> Dict[str, pd.DataFrame]:
    """Scores the naive baselines through the L2 fold harness (backlog L4).

    No models are trained: per fold and target the position-group per-90 rate
    baseline and the player-L5 baseline are evaluated with the same metrics
    as model runs, so every skill score has an inspectable denominator run.
    Returns ``report`` (season/target/baseline/metric/value) and ``summary``
    (target/baseline/metric mean+std across folds).
    """

    from football_analytics.evaluation import (
        dispersion_index,
        player_l5_baseline_means,
        position_group_baseline_means,
        position_group_rates,
        ranked_probability_score,
        rolling_origin_folds,
    )

    report_rows: list[Dict[str, Any]] = []
    for season, train_df, valid_df in rolling_origin_folds(
        frame, season_column=season_column, min_train_seasons=min_train_seasons
    ):
        for target in target_columns:
            target_train = filter_rows_for_target(train_df, target)
            target_valid = filter_rows_for_target(valid_df, target)
            if target_train.empty or target_valid.empty:
                continue
            y_train = pd.to_numeric(target_train[target], errors="coerce").fillna(0.0).to_numpy()
            if y_train.sum() <= 0:
                continue
            y_valid = pd.to_numeric(target_valid[target], errors="coerce").fillna(0.0).to_numpy()
            train_exposure = exposure_from_minutes(target_train[exposure_column])
            valid_exposure = exposure_from_minutes(target_valid[exposure_column])

            rates = position_group_rates(target_train, target, exposure=train_exposure)
            posgroup_mu = position_group_baseline_means(
                target_valid, rates, exposure=valid_exposure
            )
            player_mu = player_l5_baseline_means(
                target_valid, target, exposure=valid_exposure, fallback_means=posgroup_mu
            )
            for baseline_name, mu in (
                ("posgroup_rate", posgroup_mu),
                ("player_l5", player_mu),
            ):
                metrics = dict(evaluate_count_predictions(y_valid, mu))
                metrics["rps"] = ranked_probability_score(y_valid, mu)
                metrics["dispersion_index"] = dispersion_index(y_valid, mu)
                for metric_name, metric_value in metrics.items():
                    report_rows.append({
                        "season": season,
                        "target": target,
                        "baseline": baseline_name,
                        "metric": metric_name,
                        "value": float(metric_value),
                    })

    report = pd.DataFrame(
        report_rows, columns=["season", "target", "baseline", "metric", "value"]
    )
    if report.empty:
        summary = pd.DataFrame(columns=["target", "baseline", "metric", "mean", "std", "folds"])
    else:
        summary = (
            report.groupby(["target", "baseline", "metric"])["value"]
            .agg(mean="mean", std="std", folds="count")
            .reset_index()
        )
    return {"report": report, "summary": summary}


def temporal_train_validation_split(
    frame: pd.DataFrame,
    *,
    date_column: str = "fixture_date_utc",
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates a leakage-safe chronological train/validation split."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in the interval (0, 1)")
    if date_column not in frame:
        raise ValueError(f"Missing date column: {date_column}")

    ordered = frame.copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column], utc=True)
    ordered = ordered.sort_values([date_column, "fixture_id"]).reset_index(drop=True)
    split_idx = max(1, int(len(ordered) * (1.0 - validation_fraction)))
    split_idx = min(split_idx, len(ordered) - 1)
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()
