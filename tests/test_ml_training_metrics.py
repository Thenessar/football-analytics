"""Integration test: the L-series metric suite reaches MLflow runs (L3)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
mlflow = pytest.importorskip("mlflow")

from football_analytics.ml_training import (
    PoissonLightGBMConfig,
    train_poisson_lightgbm_with_mlflow,
)


def _frame(rows=300, seed=9):
    rng = np.random.default_rng(seed)
    strength = rng.uniform(0.0, 1.0, size=rows)
    minutes = rng.uniform(45.0, 95.0, size=rows)
    rate = 0.5 + 2.0 * strength
    return pd.DataFrame({
        "fixture_id": np.repeat(np.arange(rows // 20), 20),
        "team_id": np.tile([1, 2], rows // 2),
        "position_group": rng.choice(["F", "M", "D"], size=rows),
        "player_strength": strength,
        "games_minutes": minutes,
        "shots_total": rng.poisson(rate * minutes / 90.0).astype(float),
        "shots_total_l5_p90": rate,
    })


def test_training_logs_metric_suite_and_reliability_artifacts(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    # Pin the artifact root too — otherwise artifacts land in ./mlruns
    # relative to the working directory and pollute the repo.
    experiment_name = "metric-suite-test"
    mlflow.create_experiment(
        experiment_name, artifact_location=f"file:///{tmp_path.as_posix()}/artifacts"
    )

    result = train_poisson_lightgbm_with_mlflow(
        _frame(seed=1),
        _frame(seed=2),
        target_columns=["shots_total"],
        feature_columns=["player_strength"],
        config=PoissonLightGBMConfig(
            learning_rate=0.2,
            num_leaves=7,
            min_child_samples=10,
            num_boost_round=30,
            early_stopping_rounds=10,
        ),
        experiment_name=experiment_name,
        run_name="metric-suite-test",
    )

    suite = result["metrics"]["shots_total"]["validation_suite"]
    for key in (
        "rps", "dispersion_index", "brier_ge_1", "ece_ge_1",
        "skill_vs_posgroup_rate", "skill_vs_player_l5",
        "rank_spearman", "top1_hit_rate",
    ):
        assert key in suite

    run = mlflow.search_runs(search_all_experiments=True, output_format="list")[0]
    logged = run.data.metrics
    assert "shots_total_validation_rps" in logged
    assert "shots_total_validation_dispersion_index" in logged
    assert "shots_total_validation_skill_vs_posgroup_rate" in logged
    # Poisson data scored with a Poisson model: dispersion index near 1.
    assert 0.5 < logged["shots_total_validation_dispersion_index"] < 2.0

    client = mlflow.MlflowClient()
    artifacts = {
        artifact.path
        for artifact in client.list_artifacts(run.info.run_id, "model_artifacts")
    }
    assert "model_artifacts/shots_total_reliability_ge_1.csv" in artifacts
    assert "model_artifacts/shots_total_position_group_metrics.csv" in artifacts
