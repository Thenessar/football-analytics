"""Simulation calibration backtest (ml_upgrade_backlog.md N6).

The synthetic world draws labels from the exact process the stub models
describe. Because a Poisson team total split multinomially yields Poisson
player marginals (thinning), the simulator's per-player distributions are
exactly Poisson(rate) here — so interval coverage must hit the nominal band
and the randomized PIT must be uniform. This validates the whole
backtest-scoring pipeline end to end.
"""

import numpy as np
import pandas as pd
import pytest

from football_analytics.inference import LoadedEventModel
from football_analytics.simulation import ALL_SIMULATION_TARGETS, SimulationConfig
from football_analytics.simulation_backtest import (
    score_simulation_backtest,
    simulate_completed_fixtures,
)


class SignalBooster:
    """Predicts log(rate90) from a per-target signal column."""

    def __init__(self, column):
        self.column = column

    def predict(self, X, raw_score=False):
        return np.log(np.clip(X[self.column].to_numpy(dtype=float), 1e-6, None))


def _models():
    return [
        LoadedEventModel(
            target_event=target,
            booster=SignalBooster(f"signal_{target}"),
            feature_columns=(f"signal_{target}",),
            model_name=f"catalog.gold.player_event__{target}",
            model_version="1",
            goalkeeper_only=(target == "goals_saves"),
        )
        for target in ALL_SIMULATION_TARGETS
    ]


def _feature_rows(n_fixtures=20, seed=61):
    """Completed fixtures whose labels follow the models' rates exactly."""

    rng = np.random.default_rng(seed)
    rows = []
    player_id = 0
    for fixture_index in range(n_fixtures):
        fixture_id = 1000 + fixture_index
        for team_id in (1, 2):
            team_shot_rates = []
            for slot in range(11):
                player_id += 1
                is_goalkeeper = slot == 0
                shots = 0.0 if is_goalkeeper else float(rng.uniform(0.5, 3.0))
                team_shot_rates.append(shots)
                rates = {
                    "shots_total": shots,
                    "shots_on": 0.4 * shots,
                    "goals_total": 0.12 * shots,
                    "passes_total": float(rng.uniform(25.0, 70.0)),
                    "offsides": 0.3,
                    "fouls_committed": 1.1,
                    "fouls_drawn": 1.1,
                    "cards_yellow": 0.15,
                    "cards_red": 0.01,
                    "goals_assists": 0.1,
                    "goals_saves": 3.0 if is_goalkeeper else 0.0,
                }
                labels = {
                    target: float(rng.poisson(rate))
                    for target, rate in rates.items()
                    if target not in ("cards_red", "goals_saves")
                }
                labels["cards_red"] = float(rng.random() < rates["cards_red"])
                # GK saves label drawn from the identity-consistent marginal:
                # Poisson(sum of opponent on-target rates x miss share).
                labels["goals_saves"] = (
                    float(rng.poisson(0.4 * 0.7 * 19.0)) if is_goalkeeper else 0.0
                )
                rows.append({
                    "fixture_id": fixture_id,
                    "fixture_date_utc": pd.Timestamp("2026-06-01T18:00:00Z"),
                    "team_id": team_id,
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "position_group": "G" if is_goalkeeper else "M",
                    "is_goalkeeper": is_goalkeeper,
                    "is_starting": True,
                    "expected_minutes": 90.0,
                    "p_plays": 1.0,
                    "expected_minutes_if_plays": 90.0,
                    **{f"signal_{target}": rate for target, rate in rates.items()},
                    **labels,
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def backtest_report():
    feature_rows = _feature_rows()
    results, skipped = simulate_completed_fixtures(
        feature_rows,
        _models(),
        config=SimulationConfig(n_sims=2000, seed=71),
    )
    assert skipped.empty
    assert len(results) == 20
    return score_simulation_backtest(results, feature_rows)


def test_backtest_report_covers_all_targets(backtest_report):
    for frame_name in ("player_coverage", "team_coverage", "allocation"):
        frame = backtest_report[frame_name]
        assert set(frame["target_event"]) == set(ALL_SIMULATION_TARGETS)


def test_player_intervals_and_pit_are_calibrated_in_a_true_model_world(backtest_report):
    coverage = backtest_report["player_coverage"].set_index("target_event")

    # 10-90% interval on discrete counts: realized coverage >= nominal band.
    for target in ("shots_total", "passes_total"):
        assert 0.78 <= coverage.loc[target, "coverage"] <= 0.99
    # Randomized PIT is exactly uniform under a true model; tolerance covers
    # 440-sample noise plus the 2000-draw empirical CDF.
    assert coverage.loc["passes_total", "pit_max_abs_deviation"] < 0.07
    assert coverage.loc["shots_total", "pit_max_abs_deviation"] < 0.07


def test_team_totals_and_allocation_are_scored(backtest_report):
    team = backtest_report["team_coverage"].set_index("target_event")
    assert 0.70 <= team.loc["passes_total", "coverage"] <= 1.0
    assert team.loc["shots_total", "team_fixtures_scored"] == 40

    allocation = backtest_report["allocation"].set_index("target_event")
    # Heterogeneous shot rates: the predicted leader must beat the 1/10
    # random-pick baseline comfortably.
    assert allocation.loc["shots_total", "top1_hit_rate"] > 0.2
    assert allocation.loc["shots_total", "top3_hit_rate"] > allocation.loc["shots_total", "top1_hit_rate"]


def test_backtest_skips_fixtures_without_full_lineups():
    feature_rows = _feature_rows(n_fixtures=2)
    short = feature_rows[
        ~((feature_rows["fixture_id"] == 1000) & (feature_rows["player_id"] % 11 == 0))
    ]
    results, skipped = simulate_completed_fixtures(
        short,
        _models(),
        config=SimulationConfig(n_sims=200, seed=73),
    )
    assert len(results) == 1
    assert skipped["fixture_id"].tolist() == [1000]


def test_backtest_skips_fixtures_without_expected_minutes():
    # First-real-run regression (FA-106): fixtures predating lineup ingestion
    # have null expected_minutes on every row; they must land in the skip log
    # with a reason instead of crashing input construction with a KeyError.
    import numpy as np

    feature_rows = _feature_rows(n_fixtures=2)
    feature_rows.loc[
        feature_rows["fixture_id"] == 1000, "expected_minutes"
    ] = np.nan

    results, skipped = simulate_completed_fixtures(
        feature_rows,
        _models(),
        config=SimulationConfig(n_sims=200, seed=73),
    )

    assert len(results) == 1
    assert skipped["fixture_id"].tolist() == [1000]
    assert "expected_minutes" in skipped["reason"].iloc[0]
