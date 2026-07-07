"""Self-contained pyfunc model artifact round-trip (ml_upgrade_backlog.md M2)."""

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")
mlflow = pytest.importorskip("mlflow")

from football_analytics.inference import load_event_model_artifact
from football_analytics.ml_training import (
    ExposurePoissonLightGBMModel,
    build_pyfunc_player_event_model,
)


@pytest.fixture()
def tracking(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path.as_posix()}/mlflow.db")
    experiment = "pyfunc-roundtrip"
    mlflow.create_experiment(
        experiment, artifact_location=f"file:///{tmp_path.as_posix()}/artifacts"
    )
    mlflow.set_experiment(experiment)


def _tiny_model(goalkeeper_only=False, alpha_player=0.3, alpha_team=0.7):
    rng = np.random.default_rng(4)
    X = pd.DataFrame({"strength": rng.uniform(0, 1, 400)})
    y = rng.poisson(0.5 + 2.0 * X["strength"])
    booster = lgb.train(
        {"objective": "poisson", "verbosity": -1, "min_child_samples": 10, "seed": 1},
        lgb.Dataset(X, label=y),
        num_boost_round=20,
    )
    return ExposurePoissonLightGBMModel(
        booster=booster,
        feature_columns=["strength"],
        target_column="shots_total",
        goalkeeper_only=goalkeeper_only,
        alpha_player=alpha_player,
        alpha_team=alpha_team,
    )


def test_pyfunc_roundtrip_preserves_exposure_semantics_and_metadata(tracking):
    model = _tiny_model()
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            name="player_event_model",
            python_model=build_pyfunc_player_event_model(model),
        )

    loaded = mlflow.pyfunc.load_model(info.model_uri)
    frame = pd.DataFrame({"strength": [0.1, 0.9], "expected_minutes": [45.0, 90.0]})
    predictions = np.asarray(loaded.predict(frame), dtype=float)

    raw = model.booster.predict(frame[["strength"]], raw_score=True)
    expected = np.exp(raw) * np.array([0.5, 1.0])
    assert predictions == pytest.approx(expected)

    inner = loaded.unwrap_python_model().inner
    assert inner.alpha_player == pytest.approx(0.3)
    assert inner.alpha_team == pytest.approx(0.7)
    assert inner.feature_columns == ["strength"]

    # Without a minutes column the output is a per-90 rate.
    rate = np.asarray(loaded.predict(frame[["strength"]]), dtype=float)
    assert rate == pytest.approx(np.exp(raw))


def test_pyfunc_goalkeeper_gating(tracking):
    model = _tiny_model(goalkeeper_only=True)
    with mlflow.start_run():
        info = mlflow.pyfunc.log_model(
            name="player_event_model",
            python_model=build_pyfunc_player_event_model(model),
        )
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    frame = pd.DataFrame({
        "strength": [0.5, 0.5],
        "expected_minutes": [90.0, 90.0],
        "is_goalkeeper": [True, False],
    })
    predictions = np.asarray(loaded.predict(frame), dtype=float)
    assert predictions[0] > 0.0
    assert predictions[1] == 0.0


def test_load_event_model_artifact_handles_both_generations(tracking):
    model = _tiny_model()
    with mlflow.start_run():
        pyfunc_info = mlflow.pyfunc.log_model(
            name="m2_model",
            python_model=build_pyfunc_player_event_model(model),
        )
        legacy_info = mlflow.lightgbm.log_model(model.booster, name="legacy_model")

    m2 = load_event_model_artifact(pyfunc_info.model_uri, target_event="goals_saves")
    assert m2["self_contained"]
    assert m2["alpha_player"] == pytest.approx(0.3)
    assert m2["feature_columns"] == ("strength",)
    # M2 artifacts carry gating explicitly (False here despite the target).
    assert m2["goalkeeper_only"] is False

    legacy = load_event_model_artifact(legacy_info.model_uri, target_event="goals_saves")
    assert not legacy["self_contained"]
    assert legacy["feature_columns"] == ("strength",)
    # Pre-M2 artifacts derive gating from the target constant.
    assert legacy["goalkeeper_only"] is True
    X = pd.DataFrame({"strength": [0.4]})
    assert legacy["booster"].predict(X, raw_score=True) == pytest.approx(
        model.booster.predict(X, raw_score=True)
    )
