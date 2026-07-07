import math

import numpy as np
import pandas as pd
import pytest

from football_analytics.evaluation import (
    dispersion_index,
    player_l5_baseline_means,
    poisson_log_loss,
    position_group_baseline_means,
    position_group_rates,
    prob_at_least,
    ranked_probability_score,
    rolling_origin_folds,
    skill_score,
    threshold_calibration,
    validation_metric_suite,
    within_fixture_ranking,
)


def test_poisson_log_loss_matches_hand_computed_value():
    # y=1, mu=2: 2 - 1*ln(2) + ln(1!) = 2 - ln 2
    assert poisson_log_loss([1.0], [2.0]) == pytest.approx(2.0 - math.log(2.0))
    # y=2, mu=1: 1 - 2*ln(1) + ln(2) = 1 + ln 2
    assert poisson_log_loss([2.0], [1.0]) == pytest.approx(1.0 + math.log(2.0))


def test_prob_at_least_matches_poisson_survival():
    mu = 1.5
    p_ge_1 = 1.0 - math.exp(-mu)
    p_ge_2 = 1.0 - math.exp(-mu) * (1.0 + mu)
    assert prob_at_least([mu], 1)[0] == pytest.approx(p_ge_1)
    assert prob_at_least([mu], 2)[0] == pytest.approx(p_ge_2)
    assert prob_at_least([mu], 0)[0] == 1.0


def test_prob_at_least_nb_widens_the_tail():
    # Overdispersion shifts mass outward: P(Y >= 3) grows with alpha.
    poisson_tail = prob_at_least([1.0], 3, alpha=0.0)[0]
    nb_tail = prob_at_least([1.0], 3, alpha=1.0)[0]
    assert nb_tail > poisson_tail
    # NB survival for mu=1, alpha=1 (r=1, geometric with p=1/2):
    # P(Y >= 3) = (1/2)^3
    assert nb_tail == pytest.approx(0.125, abs=1e-9)


def test_rps_degenerate_distribution_hand_computed():
    # mu -> 0 concentrates all mass at 0: F(k) = 1 for all k.
    # y = 2 pays (1-0)^2 for k = 0 and 1 -> rps = 2; y = 0 pays nothing.
    assert ranked_probability_score([2.0], [1e-12]) == pytest.approx(2.0, abs=1e-6)
    assert ranked_probability_score([0.0], [1e-12]) == pytest.approx(0.0, abs=1e-6)


def test_rps_rewards_the_true_distribution_and_alpha_zero_is_poisson():
    rng = np.random.default_rng(7)
    y = rng.poisson(2.0, size=4000).astype(float)
    good = ranked_probability_score(y, np.full(len(y), 2.0))
    biased = ranked_probability_score(y, np.full(len(y), 4.0))
    assert good < biased
    with_alpha_zero = ranked_probability_score(y, np.full(len(y), 2.0), alpha=0.0)
    assert with_alpha_zero == pytest.approx(good)


def test_threshold_calibration_hand_computed():
    result = threshold_calibration([True, False], [0.8, 0.2], n_bins=1)
    assert result["brier"] == pytest.approx(((0.8 - 1) ** 2 + (0.2 - 0) ** 2) / 2)
    # One bin: ECE = |mean(p) - mean(y)| = |0.5 - 0.5| = 0
    assert result["ece"] == pytest.approx(0.0)
    table = result["reliability"]
    assert table["count"].sum() == 2

    skewed = threshold_calibration([False, False], [0.9, 0.9], n_bins=1)
    assert skewed["ece"] == pytest.approx(0.9)


def test_dispersion_index_hand_computed_and_poisson_reference():
    # y=[0,2], mu=[1,1]: ((0-1)^2 + (2-1)^2) / 2 = 1.0
    assert dispersion_index([0.0, 2.0], [1.0, 1.0]) == pytest.approx(1.0)
    rng = np.random.default_rng(11)
    y = rng.poisson(3.0, size=20000).astype(float)
    assert dispersion_index(y, np.full(len(y), 3.0)) == pytest.approx(1.0, abs=0.05)
    # NB draws (alpha=1) against a Poisson mean must look overdispersed.
    nb = rng.negative_binomial(1, 1.0 / (1.0 + 3.0), size=20000).astype(float)
    assert dispersion_index(nb, np.full(len(nb), 3.0)) > 1.5


def test_within_fixture_ranking_perfect_and_reversed():
    frame = pd.DataFrame({
        "fixture_id": [1] * 3 + [2] * 3,
        "team_id": [10] * 3 + [10] * 3,
        "pred": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
        "actual": [5.0, 3.0, 1.0, 5.0, 3.0, 1.0],
    })
    result = within_fixture_ranking(frame, pred_col="pred", actual_col="actual")
    # fixture 1 perfectly ordered (+1), fixture 2 reversed (-1)
    assert result["spearman_mean"] == pytest.approx(0.0)
    assert result["top1_hit_rate"] == pytest.approx(0.5)
    assert result["groups_scored"] == 2


def test_within_fixture_ranking_skips_all_zero_groups():
    frame = pd.DataFrame({
        "fixture_id": [1, 1, 2, 2],
        "team_id": [10, 10, 10, 10],
        "pred": [1.0, 2.0, 1.0, 2.0],
        "actual": [0.0, 0.0, 0.0, 3.0],
    })
    result = within_fixture_ranking(frame, pred_col="pred", actual_col="actual")
    assert result["groups_skipped_all_zero"] == 1
    assert result["groups_scored"] == 1
    assert result["top1_hit_rate"] == pytest.approx(1.0)


def test_position_group_baselines_and_player_l5_fallback():
    train = pd.DataFrame({
        "position_group": ["F", "F", "D", "D"],
        "shots_total": [4.0, 2.0, 1.0, 1.0],
    })
    rates = position_group_rates(train, "shots_total", exposure=[1.0, 1.0, 1.0, 1.0])
    assert rates["F"] == pytest.approx(3.0)
    assert rates["D"] == pytest.approx(1.0)
    assert rates["__global__"] == pytest.approx(2.0)

    eval_frame = pd.DataFrame({
        "position_group": ["F", "D", "M"],
        "shots_total_l5_p90": [2.5, 0.0, None],
    })
    posgroup_mu = position_group_baseline_means(eval_frame, rates, exposure=[1.0, 1.0, 1.0])
    assert posgroup_mu == pytest.approx([3.0, 1.0, 2.0])  # M falls back to global

    player_mu = player_l5_baseline_means(
        eval_frame, "shots_total", exposure=[1.0, 1.0, 1.0], fallback_means=posgroup_mu
    )
    # Player 0 has history (2.5/90 rate); players 1-2 fall back to posgroup.
    assert player_mu == pytest.approx([2.5, 1.0, 2.0])


def test_skill_score():
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)
    assert skill_score(1.0, 0.5) == pytest.approx(-1.0)
    assert math.isnan(skill_score(1.0, 0.0))


def test_rolling_origin_folds_expand_chronologically():
    frame = pd.DataFrame({
        "league_season": [2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
        "value": range(8),
    })
    folds = list(rolling_origin_folds(frame, min_train_seasons=2))
    assert [season for season, _, _ in folds] == [2023, 2024]

    season, train, valid = folds[0]
    assert set(train["league_season"]) == {2021, 2022}
    assert set(valid["league_season"]) == {2023}

    season, train, valid = folds[1]
    assert set(train["league_season"]) == {2021, 2022, 2023}
    assert set(valid["league_season"]) == {2024}


def test_validation_metric_suite_returns_scalars_and_artifacts():
    rng = np.random.default_rng(3)
    n_train, n_valid = 400, 200
    train = pd.DataFrame({
        "position_group": rng.choice(["F", "M", "D"], size=n_train),
        "shots_total": rng.poisson(1.2, size=n_train).astype(float),
    })
    valid = pd.DataFrame({
        "fixture_id": np.repeat(np.arange(n_valid // 10), 10),
        "team_id": 1,
        "position_group": rng.choice(["F", "M", "D"], size=n_valid),
        "shots_total_l5_p90": rng.uniform(0, 2, size=n_valid),
    })
    y_valid = rng.poisson(1.2, size=n_valid).astype(float)
    mu_valid = np.full(n_valid, 1.2)

    suite = validation_metric_suite(
        train_frame=train,
        valid_frame=valid,
        y_valid=y_valid,
        mu_valid=mu_valid,
        target="shots_total",
        train_exposure=np.ones(n_train),
        valid_exposure=np.ones(n_valid),
    )

    metrics = suite["metrics"]
    expected_keys = {
        "rps", "dispersion_index", "brier_ge_1", "ece_ge_1", "brier_ge_2",
        "ece_ge_2", "skill_vs_posgroup_rate", "skill_vs_player_l5",
        "rank_spearman", "top1_hit_rate", "top3_hit_rate",
    }
    assert expected_keys.issubset(metrics.keys())
    for key in expected_keys:
        assert isinstance(metrics[key], float)
    # The true-rate prediction should not lose badly to the global baseline.
    assert metrics["skill_vs_posgroup_rate"] > -0.05
    assert not suite["artifacts"]["reliability_ge_1"].empty
