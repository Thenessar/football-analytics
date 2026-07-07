"""Two-stage negative-binomial dispersion (ml_upgrade_backlog.md M1)."""

import math

import numpy as np
import pandas as pd
import pytest

from football_analytics.evaluation import (
    estimate_nb_alpha,
    estimate_team_total_nb_alpha,
)
from football_analytics.ml_training import count_threshold_probabilities


def test_alpha_is_zero_on_poisson_data():
    rng = np.random.default_rng(21)
    mu = rng.uniform(0.5, 4.0, size=30000)
    y = rng.poisson(mu)
    assert estimate_nb_alpha(y, mu) < 0.02


def test_alpha_recovers_known_nb_dispersion():
    rng = np.random.default_rng(22)
    alpha_true = 0.6
    mu = rng.uniform(0.5, 4.0, size=40000)
    r = 1.0 / alpha_true
    y = rng.negative_binomial(r, r / (r + mu))
    alpha_hat = estimate_nb_alpha(y, mu)
    assert alpha_hat == pytest.approx(alpha_true, rel=0.15)


def test_alpha_never_negative_and_empty_safe():
    # Underdispersed data clips to zero rather than going negative.
    y = np.full(1000, 2.0)
    mu = np.full(1000, 2.0)
    assert estimate_nb_alpha(y, mu) == 0.0
    assert estimate_nb_alpha([], []) == 0.0


def test_team_total_alpha_fits_on_group_sums():
    rng = np.random.default_rng(23)
    n_teams, players = 800, 11
    frame = pd.DataFrame({
        "fixture_id": np.repeat(np.arange(n_teams), players),
        "team_id": 1,
    })
    mu = np.full(len(frame), 0.9)
    # Correlated players: a shared per-team gamma factor creates
    # overdispersion that only shows at the team-total level.
    team_factor = np.repeat(rng.gamma(2.0, 0.5, size=n_teams), players)
    y = rng.poisson(mu * team_factor)

    alpha_team = estimate_team_total_nb_alpha(frame, y_true=y, mu=mu)
    alpha_player = estimate_nb_alpha(y, mu)
    assert alpha_team > alpha_player
    # Gamma(shape=2) mixing implies alpha = 1/shape = 0.5 at the total level.
    assert alpha_team == pytest.approx(0.5, rel=0.25)


def test_threshold_probabilities_alpha_zero_is_bit_identical_poisson():
    for mu in (0.05, 0.7, 2.3):
        legacy = {}
        for threshold in (1, 2, 3):
            cumulative = sum(
                math.exp(-mu) * mu ** i / math.factorial(i) for i in range(threshold)
            )
            legacy[f"p_ge_{threshold}"] = float(np.clip(1.0 - cumulative, 0.0, 1.0))
        result = count_threshold_probabilities(mu, thresholds=(1, 2, 3), alpha=0.0)
        for key, value in legacy.items():
            assert result[key] == value


def test_threshold_probabilities_nb_matches_exact_reference():
    # mu=2, alpha=0.5 -> r=2, p=1/2: P(0)=1/4, P(1)=1/4 -> exact tails.
    result = count_threshold_probabilities(2.0, thresholds=(1, 2), alpha=0.5)
    assert result["p_ge_1"] == pytest.approx(0.75, abs=1e-9)
    assert result["p_ge_2"] == pytest.approx(0.5, abs=1e-9)
    # Overdispersion fattens the DEEP tail (the crossover point sits past the
    # mean — at k=3 with mu=2 Poisson is still higher, which is correct).
    # Exact NB(r=2, p=1/2) deep tail: P(Y >= 6) = 1/16.
    poisson = count_threshold_probabilities(2.0, thresholds=(6,), alpha=0.0)
    nb = count_threshold_probabilities(2.0, thresholds=(6,), alpha=0.5)
    assert nb["p_ge_6"] == pytest.approx(0.0625, abs=1e-9)
    assert nb["p_ge_6"] > poisson["p_ge_6"]
