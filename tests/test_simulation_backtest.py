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


def _models(
    passes_alpha_player=0.0,
    passes_alpha_team=0.0,
    passes_alpha_by_position=None,
):
    return [
        LoadedEventModel(
            target_event=target,
            booster=SignalBooster(f"signal_{target}"),
            feature_columns=(f"signal_{target}",),
            model_name=f"catalog.gold.player_event__{target}",
            model_version="1",
            goalkeeper_only=(target == "goals_saves"),
            alpha_player=passes_alpha_player if target == "passes_total" else 0.0,
            alpha_team=passes_alpha_team if target == "passes_total" else 0.0,
            alpha_player_by_position=(
                dict(passes_alpha_by_position)
                if passes_alpha_by_position and target == "passes_total"
                else {}
            ),
        )
        for target in ALL_SIMULATION_TARGETS
    ]


def _feature_rows(
    n_fixtures=20,
    seed=61,
    passes_dispersion=0.0,
    passes_dispersion_by_group=None,
):
    """Completed fixtures whose labels follow the models' rates exactly.

    ``passes_dispersion > 0`` switches passes labels from Poisson(rate) to
    NB(rate, alpha) via the Poisson–Gamma mixture — the player-heterogeneous
    world the FA-108 dispersion layer must recover.
    ``passes_dispersion_by_group`` ({group: alpha}, FA-109) draws each
    player's passes label with his position group's alpha instead — the
    role-heterogeneous world a single pooled alpha cannot calibrate.
    Outfield slots split into defenders (slots 1-5, 'D') and forwards
    (slots 6-10, 'F'); slot 0 is the goalkeeper ('G').
    """

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
                position_group = "G" if is_goalkeeper else ("D" if slot <= 5 else "F")
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
                label_alpha = passes_dispersion
                if passes_dispersion_by_group:
                    label_alpha = float(
                        passes_dispersion_by_group.get(position_group, 0.0)
                    )
                if label_alpha > 0.0:
                    labels["passes_total"] = float(rng.poisson(
                        rates["passes_total"]
                        * rng.gamma(1.0 / label_alpha, label_alpha)
                    ))
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
                    "position_group": position_group,
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


def test_player_dispersion_layer_restores_passes_calibration_in_an_nb_world():
    # FA-108: real passes counts are NB-overdispersed per player, and the
    # plain multinomial allocation then under-covers exactly like the FA-106
    # holdout run (0.581 at 0.80 nominal, PIT deviation 0.189). In a synthetic
    # NB world the flag must restore nominal coverage and a uniform PIT while
    # the team-total calibration stays intact.
    alpha = 0.3
    feature_rows = _feature_rows(seed=67, passes_dispersion=alpha)
    # alpha_team ~ alpha * sum(w^2)/W^2: the team-level trace the player
    # heterogeneity leaves on 11-player totals (cf. ADR 0005 note on
    # alpha_team < alpha_player).
    models = _models(passes_alpha_player=alpha, passes_alpha_team=0.03)

    off_results, _ = simulate_completed_fixtures(
        feature_rows, models, config=SimulationConfig(n_sims=1500, seed=79)
    )
    on_results, _ = simulate_completed_fixtures(
        feature_rows,
        models,
        config=SimulationConfig(
            n_sims=1500, seed=79, player_dispersion_targets=("passes_total",)
        ),
    )
    off_cov = (
        score_simulation_backtest(off_results, feature_rows)["player_coverage"]
        .set_index("target_event")
    )
    on_report = score_simulation_backtest(on_results, feature_rows)
    on_cov = on_report["player_coverage"].set_index("target_event")

    # Off reproduces the FA-106 symptom: far-too-narrow player intervals.
    assert off_cov.loc["passes_total", "coverage"] < 0.60
    assert off_cov.loc["passes_total", "pit_max_abs_deviation"] > 0.12

    # On restores the nominal band and PIT uniformity...
    assert 0.74 <= on_cov.loc["passes_total", "coverage"] <= 0.93
    assert on_cov.loc["passes_total", "pit_max_abs_deviation"] < 0.08
    # ...without disturbing an already-calibrated target.
    assert 0.78 <= on_cov.loc["shots_total", "coverage"] <= 0.99

    # Team passes totals stay calibrated with the layer on — the NB team
    # draw with the measured alpha_team is untouched by the allocation layer.
    team_on = on_report["team_coverage"].set_index("target_event")
    assert team_on.loc["passes_total", "coverage"] >= 0.65


def test_position_group_dispersion_beats_pooled_alpha_in_a_role_heterogeneous_world():
    # FA-109: when defenders' and forwards' passes variance differ
    # structurally (as the real N6 residual suggests), one pooled alpha
    # over-disperses the tight group and under-disperses the loose one —
    # coverage can look fine while the PIT stays visibly non-uniform. The
    # per-position refinement must restore uniformity.
    # This contrast reproduces the real N6 signature almost exactly: the
    # pooled arm lands at PIT deviation ~0.14 (holdout measured 0.139).
    alpha_by_group = {"D": 0.01, "F": 1.2, "G": 0.01}
    pooled_alpha = 0.5
    feature_rows = _feature_rows(
        seed=83, passes_dispersion_by_group=alpha_by_group
    )
    models = _models(
        passes_alpha_player=pooled_alpha,
        passes_alpha_team=0.03,
        passes_alpha_by_position=alpha_by_group,
    )

    pooled_results, _ = simulate_completed_fixtures(
        feature_rows,
        models,
        config=SimulationConfig(
            n_sims=1500, seed=89, player_dispersion_targets=("passes_total",)
        ),
    )
    by_position_results, _ = simulate_completed_fixtures(
        feature_rows,
        models,
        config=SimulationConfig(
            n_sims=1500,
            seed=89,
            player_dispersion_targets=("passes_total",),
            player_dispersion_by_position=True,
        ),
    )

    pooled_cov = (
        score_simulation_backtest(pooled_results, feature_rows)["player_coverage"]
        .set_index("target_event")
    )
    by_position_cov = (
        score_simulation_backtest(by_position_results, feature_rows)["player_coverage"]
        .set_index("target_event")
    )

    # Pooled alpha leaves the PIT visibly non-uniform in this world...
    assert pooled_cov.loc["passes_total", "pit_max_abs_deviation"] > 0.10
    # ...while the per-position refinement restores calibration.
    assert by_position_cov.loc["passes_total", "pit_max_abs_deviation"] < 0.08
    assert 0.74 <= by_position_cov.loc["passes_total", "coverage"] <= 0.95
    # A strict improvement, and unrelated targets stay calibrated.
    assert (
        by_position_cov.loc["passes_total", "pit_max_abs_deviation"]
        < pooled_cov.loc["passes_total", "pit_max_abs_deviation"] - 0.03
    )
    assert 0.78 <= by_position_cov.loc["shots_total", "coverage"] <= 0.99


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
