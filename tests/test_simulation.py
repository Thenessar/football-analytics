"""Fixture simulation engine tests (ml_upgrade_backlog.md Epic N)."""

import numpy as np
import pandas as pd
import pytest

from football_analytics.simulation import (
    ALL_SIMULATION_TARGETS,
    SimulationConfig,
    SimulationInputError,
    allocate_event_totals,
    build_simulation_inputs,
    deterministic_sim_set_id,
    multinomial_allocate,
    sample_dispersed_allocation_weights,
    sample_minutes,
    sample_nb_totals,
    simulate_fixture,
)


def _active_predictions(
    players_per_team=13,
    starters_per_team=11,
    teams=(1, 2),
    fixture_id=100,
    bench_p_plays=0.4,
    bench_if_plays=25.0,
):
    """Synthetic active prediction set: one row per player/target."""

    rows = []
    player_id = 0
    for team_id in teams:
        for slot in range(players_per_team):
            player_id += 1
            is_starting = slot < starters_per_team
            p_plays = 1.0 if is_starting else bench_p_plays
            if_plays = 90.0 if is_starting else bench_if_plays
            expected_minutes = p_plays * if_plays
            is_goalkeeper = slot == 0
            for target in ALL_SIMULATION_TARGETS:
                if target == "goals_saves":
                    rate90 = 3.0 if is_goalkeeper else 0.0
                elif target == "passes_total":
                    rate90 = 40.0
                elif target == "shots_total":
                    rate90 = 0.0 if is_goalkeeper else 1.5
                elif target == "shots_on":
                    rate90 = 0.0 if is_goalkeeper else 0.6
                elif target == "goals_total":
                    rate90 = 0.0 if is_goalkeeper else 0.15
                elif target == "goals_assists":
                    rate90 = 0.0 if is_goalkeeper else 0.1
                elif target in ("fouls_committed", "fouls_drawn"):
                    rate90 = 1.0
                elif target == "cards_yellow":
                    rate90 = 0.2
                elif target == "cards_red":
                    rate90 = 0.01
                else:  # offsides
                    rate90 = 0.3
                rows.append({
                    "fixture_id": fixture_id,
                    "prediction_set_id": "pred-set-1",
                    "team_id": team_id,
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "position_group": "G" if is_goalkeeper else "M",
                    "is_starting": is_starting,
                    "target_event": target,
                    "expected_minutes": expected_minutes,
                    "p_plays": p_plays,
                    "expected_minutes_if_plays": if_plays,
                    "predicted_mean": rate90 * expected_minutes / 90.0,
                })
    return pd.DataFrame(rows)


def test_inputs_recover_per90_rates_and_decomposition():
    inputs = build_simulation_inputs(_active_predictions())

    assert inputs.fixture_id == 100
    assert inputs.team_ids == (1, 2)
    assert len(inputs.players) == 26

    starters = inputs.players[inputs.players["is_starting"]]
    bench = inputs.players[~inputs.players["is_starting"]]
    # Per-90 intensity is recovered exactly: mean / exposure round-trips.
    outfield_starters = starters[starters["position_group"] != "G"]
    assert outfield_starters["rate90_shots_total"].to_numpy() == pytest.approx(1.5)
    # Bench rows carry the decomposition, not just the blended mean.
    assert bench["p_plays"].to_numpy() == pytest.approx(0.4)
    assert bench["expected_minutes_if_plays"].to_numpy() == pytest.approx(25.0)
    # Bench per-90 rate must round-trip through the blended expected minutes.
    outfield_bench = bench[bench["position_group"] != "G"]
    assert outfield_bench["rate90_shots_total"].to_numpy() == pytest.approx(1.5)


def test_inputs_validation_errors_are_specific():
    with pytest.raises(SimulationInputError, match="No active prediction rows"):
        build_simulation_inputs(_active_predictions().iloc[0:0])

    two_fixtures = pd.concat([
        _active_predictions(fixture_id=100),
        _active_predictions(fixture_id=200),
    ])
    with pytest.raises(SimulationInputError, match="exactly one fixture"):
        build_simulation_inputs(two_fixtures)

    missing_target = _active_predictions()
    missing_target = missing_target[missing_target["target_event"] != "shots_on"]
    with pytest.raises(SimulationInputError, match="shots_on"):
        build_simulation_inputs(missing_target)

    short_team = _active_predictions(players_per_team=10, starters_per_team=10)
    with pytest.raises(SimulationInputError, match="only 10 predicted players"):
        build_simulation_inputs(short_team)

    # All-null predicted means (predictions built without expected_minutes):
    # pivot_table drops the all-NaN columns, which must surface as the
    # governed input error, never a raw KeyError (first-real-run regression).
    null_means = _active_predictions()
    null_means["predicted_mean"] = np.nan
    with pytest.raises(SimulationInputError, match="predicted_mean is null"):
        build_simulation_inputs(null_means)


def test_inputs_degrade_gracefully_without_decomposition_columns():
    predictions = _active_predictions().drop(
        columns=["p_plays", "expected_minutes_if_plays"]
    )
    inputs = build_simulation_inputs(predictions)
    bench = inputs.players[~inputs.players["is_starting"]]
    # Pre-K5 rows: deterministic minutes, no participation sampling.
    assert (bench["p_plays"] == 1.0).all()
    assert bench["expected_minutes_if_plays"].to_numpy() == pytest.approx(
        0.4 * 25.0
    )


def test_sim_set_id_deterministic_and_config_sensitive():
    config = SimulationConfig(n_sims=100, seed=1)
    first = deterministic_sim_set_id(100, prediction_set_id="p1", config=config)
    second = deterministic_sim_set_id(100, prediction_set_id="p1", config=config)
    assert first == second

    other_seed = deterministic_sim_set_id(
        100, prediction_set_id="p1", config=SimulationConfig(n_sims=100, seed=2)
    )
    assert other_seed != first
    other_set = deterministic_sim_set_id(
        100, prediction_set_id="p2", config=config
    )
    assert other_set != first


def test_nb_totals_match_target_mean_and_variance():
    rng = np.random.default_rng(31)
    means = np.full(60000, 12.0)
    alpha = 0.15

    totals = np.asarray(sample_nb_totals(rng, means, alpha), dtype=float)
    assert totals.mean() == pytest.approx(12.0, rel=0.02)
    expected_var = 12.0 + alpha * 12.0 ** 2
    assert totals.var() == pytest.approx(expected_var, rel=0.05)

    poisson_totals = np.asarray(
        sample_nb_totals(np.random.default_rng(32), means, 0.0), dtype=float
    )
    assert poisson_totals.var() == pytest.approx(12.0, rel=0.05)


def test_multinomial_allocation_conserves_totals_and_preserves_means():
    rng = np.random.default_rng(33)
    n_sims = 50000
    totals = rng.poisson(10.0, size=n_sims)
    weights = np.tile(np.array([[4.0], [3.0], [2.0], [1.0]]), (1, n_sims))

    counts = multinomial_allocate(rng, totals, weights)

    # Conservation: every simulation's counts add to its total exactly.
    np.testing.assert_array_equal(counts.sum(axis=0), totals)
    # Mean preservation: E[count_p] = E[T] * w_p / sum(w).
    shares = counts.mean(axis=1) / 10.0
    assert shares == pytest.approx([0.4, 0.3, 0.2, 0.1], abs=0.01)


def test_allocation_repairs_zero_weight_and_empty_pitch_edges():
    rng = np.random.default_rng(34)
    totals = np.array([6, 6], dtype=np.int64)
    weights = np.zeros((3, 2))
    on_pitch = np.array([
        [True, False],
        [True, False],
        [False, False],
    ])

    counts, repaired = allocate_event_totals(rng, totals, weights, on_pitch)

    # Sim 0: zero weights but two players on pitch -> uniform split of 6.
    assert counts[:, 0].sum() == 6
    assert counts[2, 0] == 0
    # Sim 1: nobody on pitch -> the total itself is forced to zero.
    assert repaired[1] == 0
    assert counts[:, 1].sum() == 0


def test_inputs_clamp_player_dispersions_like_team_dispersions():
    inputs = build_simulation_inputs(
        _active_predictions(),
        alpha_team={"passes_total": -0.2, "shots_total": 0.4},
        alpha_player={"passes_total": -0.5, "shots_total": 0.3},
    )
    assert inputs.alpha_team == {"passes_total": 0.0, "shots_total": 0.4}
    assert inputs.alpha_player == {"passes_total": 0.0, "shots_total": 0.3}


def test_dispersed_allocation_weights_are_mean_preserving_gamma_noise():
    rng = np.random.default_rng(51)
    weights = np.tile(np.array([[4.0], [0.0], [2.0]]), (1, 100_000))

    # alpha <= 0 is the exact no-op: same object back, no randomness drawn.
    assert sample_dispersed_allocation_weights(rng, weights, 0.0) is weights

    dispersed = sample_dispersed_allocation_weights(rng, weights, 0.25)
    # Multipliers are mean-1 with variance alpha: E[v] = w, Var[v] = alpha*w^2.
    assert dispersed[0].mean() == pytest.approx(4.0, rel=0.02)
    assert dispersed[0].var() == pytest.approx(0.25 * 16.0, rel=0.05)
    assert dispersed[2].mean() == pytest.approx(2.0, rel=0.02)
    # Zero weights stay exactly zero (other team / off pitch).
    assert (dispersed[1] == 0.0).all()


def test_player_dispersion_is_config_gated_deterministic_and_digest_tracked():
    predictions = _active_predictions()
    with_alpha = build_simulation_inputs(
        predictions, alpha_player={"passes_total": 0.4}
    )
    plain = build_simulation_inputs(predictions)
    base_config = SimulationConfig(n_sims=300, seed=5)
    flagged = SimulationConfig(
        n_sims=300, seed=5, player_dispersion_targets=("passes_total",)
    )

    # Flag off: a fitted alpha_player on the inputs changes nothing, bitwise.
    baseline = simulate_fixture(plain, base_config)
    gated = simulate_fixture(with_alpha, base_config)
    for target in ALL_SIMULATION_TARGETS:
        np.testing.assert_array_equal(gated.counts[target], baseline.counts[target])

    # Flag on without a fitted alpha: still bit-identical (no draws consumed).
    unfitted = simulate_fixture(plain, flagged)
    for target in ALL_SIMULATION_TARGETS:
        np.testing.assert_array_equal(unfitted.counts[target], baseline.counts[target])

    # Flag on with a fitted alpha: passes draws actually change, and the run
    # is deterministic under the seed.
    first = simulate_fixture(with_alpha, flagged)
    second = simulate_fixture(with_alpha, flagged)
    for target in ALL_SIMULATION_TARGETS:
        np.testing.assert_array_equal(first.counts[target], second.counts[target])
    assert (first.counts["passes_total"] != baseline.counts["passes_total"]).any()

    # The gate participates in the config digest (dispersed sim sets replace
    # v1 sets), order-insensitively — same behavior means same digest.
    assert flagged.digest() != base_config.digest()
    assert SimulationConfig(
        player_dispersion_targets=("passes_total", "shots_total")
    ).digest() == SimulationConfig(
        player_dispersion_targets=("shots_total", "passes_total")
    ).digest()


def test_player_dispersion_adds_nb_variance_without_moving_means_or_totals():
    alpha = 0.1
    inputs = build_simulation_inputs(
        _active_predictions(), alpha_player={"passes_total": alpha}
    )
    base = simulate_fixture(inputs, SimulationConfig(n_sims=30000, seed=29))
    dispersed = simulate_fixture(
        inputs,
        SimulationConfig(
            n_sims=30000, seed=29, player_dispersion_targets=("passes_total",)
        ),
    )

    # shots_total draws before passes_total in the fixed order, so its rng
    # stream — and therefore its counts — stay bit-identical with the layer on.
    np.testing.assert_array_equal(
        dispersed.counts["shots_total"], base.counts["shots_total"]
    )

    starters = inputs.players["is_starting"].to_numpy()
    mu = 40.0  # rate90 40 x 90 minutes

    # Player means are preserved (shares are mean-1 perturbed)...
    base_means = base.counts["passes_total"][starters].mean(axis=1)
    dispersed_means = dispersed.counts["passes_total"][starters].mean(axis=1)
    assert dispersed_means == pytest.approx(base_means, rel=0.03)

    # ...while player variance moves from the compressed multinomial
    # conditional level to the NB target mu + alpha*mu^2 (finite-team
    # normalization keeps the realized inflation slightly below the naive
    # target — the (1-share)^2 factor).
    nb_variance = mu + alpha * mu**2
    base_var = float(base.counts["passes_total"][starters].var(axis=1).mean())
    dispersed_var = float(
        dispersed.counts["passes_total"][starters].var(axis=1).mean()
    )
    assert base_var < 0.35 * nb_variance
    assert 0.80 * nb_variance < dispersed_var < 1.05 * nb_variance

    # Team totals keep their distribution — the layer only spreads the
    # allocation, it never touches the NB total draw.
    for team_id in inputs.team_ids:
        base_totals = base.team_totals("passes_total", team_id)
        dispersed_totals = dispersed.team_totals("passes_total", team_id)
        assert dispersed_totals.mean() == pytest.approx(base_totals.mean(), rel=0.02)
        assert dispersed_totals.var() == pytest.approx(base_totals.var(), rel=0.06)


def test_inputs_normalize_position_group_alphas():
    inputs = build_simulation_inputs(
        _active_predictions(),
        alpha_player={"passes_total": 0.3},
        alpha_player_by_position={"passes_total": {"g": -0.5, "d": 0.2, "M": 0.6}},
    )
    assert inputs.alpha_player_by_position == {
        "passes_total": {"G": 0.0, "D": 0.2, "M": 0.6}
    }
    assert build_simulation_inputs(_active_predictions()).alpha_player_by_position == {}


def test_dispersed_weights_accept_per_player_alpha_vector():
    rng = np.random.default_rng(53)
    weights = np.tile(np.array([[4.0], [2.0], [3.0]]), (1, 100_000))
    alphas = np.array([0.5, 0.0, 0.1])

    dispersed = sample_dispersed_allocation_weights(rng, weights, alphas)

    # Row means preserved; row variances follow each player's own alpha.
    assert dispersed[0].mean() == pytest.approx(4.0, rel=0.02)
    assert dispersed[0].var() == pytest.approx(0.5 * 16.0, rel=0.05)
    assert dispersed[2].var() == pytest.approx(0.1 * 9.0, rel=0.05)
    # Zero-alpha rows keep their weights exactly (multiplier pinned to 1).
    assert (dispersed[1] == 2.0).all()

    # All-zero vector is the exact no-op; mismatched length is a hard error.
    assert sample_dispersed_allocation_weights(
        rng, weights, np.zeros(3)
    ) is weights
    with pytest.raises(ValueError, match="alpha vector length"):
        sample_dispersed_allocation_weights(rng, weights, np.array([0.1, 0.2]))


def test_position_dispersion_is_gated_deterministic_and_digest_tracked():
    # FA-109: the per-position refinement only acts when BOTH the target is
    # dispersed and the by-position flag is on; without fitted group alphas
    # it falls back to the pooled scalar bit-for-bit.
    predictions = _active_predictions()
    pooled_inputs = build_simulation_inputs(
        predictions, alpha_player={"passes_total": 0.3}
    )
    by_position_inputs = build_simulation_inputs(
        predictions,
        alpha_player={"passes_total": 0.3},
        alpha_player_by_position={"passes_total": {"G": 0.0, "M": 0.7}},
    )
    pooled_config = SimulationConfig(
        n_sims=400, seed=11, player_dispersion_targets=("passes_total",)
    )
    by_position_config = SimulationConfig(
        n_sims=400,
        seed=11,
        player_dispersion_targets=("passes_total",),
        player_dispersion_by_position=True,
    )

    pooled = simulate_fixture(pooled_inputs, pooled_config)

    # Flag off: by-position data on the inputs changes nothing, bitwise.
    gated = simulate_fixture(by_position_inputs, pooled_config)
    for target in ALL_SIMULATION_TARGETS:
        np.testing.assert_array_equal(gated.counts[target], pooled.counts[target])

    # Flag on without fitted group alphas: pooled fallback, bit-identical.
    fallback = simulate_fixture(pooled_inputs, by_position_config)
    for target in ALL_SIMULATION_TARGETS:
        np.testing.assert_array_equal(fallback.counts[target], pooled.counts[target])

    # Flag on with fitted group alphas: passes draws change; deterministic.
    first = simulate_fixture(by_position_inputs, by_position_config)
    second = simulate_fixture(by_position_inputs, by_position_config)
    for target in ALL_SIMULATION_TARGETS:
        np.testing.assert_array_equal(first.counts[target], second.counts[target])
    assert (first.counts["passes_total"] != pooled.counts["passes_total"]).any()

    # Digest-tracked so by-position sim sets supersede pooled ones.
    assert by_position_config.digest() != pooled_config.digest()


def test_position_dispersion_applies_group_alphas_per_player():
    # Outfielders ("M", alpha 0.5) must gain far more passes variance than
    # under the pooled alpha (0.05), while means stay put — the group alpha
    # really lands on the right players.
    predictions = _active_predictions()
    mu = 40.0  # rate90 40 x 90 minutes
    pooled_inputs = build_simulation_inputs(
        predictions, alpha_player={"passes_total": 0.05}
    )
    by_position_inputs = build_simulation_inputs(
        predictions,
        alpha_player={"passes_total": 0.05},
        alpha_player_by_position={"passes_total": {"M": 0.5, "G": 0.05}},
    )
    pooled = simulate_fixture(
        pooled_inputs,
        SimulationConfig(
            n_sims=30000, seed=31, player_dispersion_targets=("passes_total",)
        ),
    )
    by_position = simulate_fixture(
        by_position_inputs,
        SimulationConfig(
            n_sims=30000,
            seed=31,
            player_dispersion_targets=("passes_total",),
            player_dispersion_by_position=True,
        ),
    )

    mask = (
        by_position.inputs.players["is_starting"]
        & (by_position.inputs.players["position_group"] == "M")
    ).to_numpy()

    pooled_means = pooled.counts["passes_total"][mask].mean(axis=1)
    by_position_means = by_position.counts["passes_total"][mask].mean(axis=1)
    assert by_position_means == pytest.approx(pooled_means, rel=0.03)

    pooled_var = float(pooled.counts["passes_total"][mask].var(axis=1).mean())
    by_position_var = float(
        by_position.counts["passes_total"][mask].var(axis=1).mean()
    )
    assert pooled_var < 0.35 * (mu + 0.5 * mu**2)
    assert 0.80 * (mu + 0.5 * mu**2) < by_position_var < 1.05 * (mu + 0.5 * mu**2)


@pytest.mark.parametrize(
    "dispersion_targets",
    [(), ("passes_total", "shots_total")],
    ids=["v1_allocation", "player_dispersion_on"],
)
def test_simulated_game_satisfies_every_structural_invariant(dispersion_targets):
    inputs = build_simulation_inputs(
        _active_predictions(),
        alpha_team={"shots_total": 0.2, "fouls_committed": 0.1},
        alpha_player={"shots_total": 0.15, "passes_total": 0.25},
    )
    config = SimulationConfig(
        n_sims=1000, seed=17, player_dispersion_targets=dispersion_targets
    )
    result = simulate_fixture(inputs, config)

    counts = result.counts
    assert set(counts) == set(ALL_SIMULATION_TARGETS)
    team_a, team_b = inputs.team_ids
    mask_a = result.team_mask(team_a)
    mask_b = result.team_mask(team_b)

    # Shot chain per player, every draw.
    assert (counts["goals_total"] <= counts["shots_on"]).all()
    assert (counts["shots_on"] <= counts["shots_total"]).all()

    # Fouls mirror: committed by A == drawn by B, per simulation, exactly.
    np.testing.assert_array_equal(
        counts["fouls_committed"][mask_a].sum(axis=0),
        counts["fouls_drawn"][mask_b].sum(axis=0),
    )
    np.testing.assert_array_equal(
        counts["fouls_committed"][mask_b].sum(axis=0),
        counts["fouls_drawn"][mask_a].sum(axis=0),
    )

    # Saves identity: starting GK of A saves B's on-target misses; everyone
    # else records zero saves.
    players = inputs.players
    gk_a = players[(players["team_id"] == team_a) & (players["position_group"] == "G")].index[0]
    expected_saves = np.maximum(
        counts["shots_on"][mask_b].sum(axis=0) - counts["goals_total"][mask_b].sum(axis=0),
        0,
    )
    np.testing.assert_array_equal(counts["goals_saves"][gk_a], expected_saves)
    non_gk = np.ones(len(players), dtype=bool)
    gk_b = players[(players["team_id"] == team_b) & (players["position_group"] == "G")].index[0]
    non_gk[[gk_a, gk_b]] = False
    assert (counts["goals_saves"][non_gk] == 0).all()

    # Assists never exceed team goals; cards respect their caps.
    assert (
        counts["goals_assists"][mask_a].sum(axis=0)
        <= counts["goals_total"][mask_a].sum(axis=0)
    ).all()
    assert counts["cards_yellow"].max() <= config.max_yellow_per_player
    assert counts["cards_red"].max() <= 1

    # Players who stayed on the bench in a simulation record nothing.
    off_pitch = result.minutes == 0.0
    for target in ALL_SIMULATION_TARGETS:
        assert (counts[target][off_pitch] == 0).all()


def test_team_goal_anchor_scales_intensities_and_preserves_shares():
    from football_analytics.simulation import apply_team_goal_anchor

    inputs = build_simulation_inputs(
        _active_predictions(),
        expected_team_goals={1: 3.0, 2: 0.5},
    )
    config = SimulationConfig(n_sims=100, seed=1, team_goal_anchor_weight=1.0)
    anchored = apply_team_goal_anchor(inputs, config)

    # w = 1: each team's expected goal total lands exactly on its Elo anchor.
    for team_id, anchor in ((1, 3.0), (2, 0.5)):
        mask = anchored.players["team_id"] == team_id
        exposure = anchored.players.loc[mask, "expected_minutes"] / 90.0
        total = float(
            (anchored.players.loc[mask, "rate90_goals_total"] * exposure).sum()
        )
        assert total == pytest.approx(anchor)

    # Shares preserved: outfield teammates keep identical (uniformly scaled)
    # intensities; every other target family is untouched.
    team_one_outfield = anchored.players[
        (anchored.players["team_id"] == 1)
        & (anchored.players["position_group"] != "G")
    ]
    assert team_one_outfield["rate90_goals_total"].nunique() == 1
    pd.testing.assert_series_equal(
        anchored.players["rate90_shots_total"],
        inputs.players["rate90_shots_total"],
    )
    pd.testing.assert_series_equal(
        anchored.players["rate90_shots_on"],
        inputs.players["rate90_shots_on"],
    )

    # The original inputs stay untouched (pure function).
    original_total = float(
        (
            inputs.players.loc[inputs.players["team_id"] == 1, "rate90_goals_total"]
            * inputs.players.loc[inputs.players["team_id"] == 1, "expected_minutes"]
            / 90.0
        ).sum()
    )
    assert original_total != pytest.approx(3.0)


def test_team_goal_anchor_blends_and_noops_without_weight_or_anchor():
    from football_analytics.simulation import apply_team_goal_anchor

    predictions = _active_predictions()
    inputs = build_simulation_inputs(
        predictions, expected_team_goals={1: 3.0, 2: 0.5}
    )

    # w = 0 (default) is an exact no-op, as is a missing anchor dict.
    assert apply_team_goal_anchor(
        inputs, SimulationConfig(team_goal_anchor_weight=0.0)
    ) is inputs
    without_anchor = build_simulation_inputs(predictions)
    assert apply_team_goal_anchor(
        without_anchor, SimulationConfig(team_goal_anchor_weight=1.0)
    ) is without_anchor

    # Intermediate weights blend the anchor with the player-sum total.
    mask = inputs.players["team_id"] == 1
    exposure = inputs.players.loc[mask, "expected_minutes"] / 90.0
    original_total = float(
        (inputs.players.loc[mask, "rate90_goals_total"] * exposure).sum()
    )
    half = apply_team_goal_anchor(
        inputs, SimulationConfig(team_goal_anchor_weight=0.5)
    )
    blended_total = float(
        (half.players.loc[mask, "rate90_goals_total"] * exposure).sum()
    )
    assert blended_total == pytest.approx(0.5 * 3.0 + 0.5 * original_total)

    # The anchor weight participates in the config digest, so anchored sim
    # sets replace rather than collide with unanchored ones.
    assert deterministic_sim_set_id(
        100, prediction_set_id="p1", config=SimulationConfig(team_goal_anchor_weight=0.5)
    ) != deterministic_sim_set_id(
        100, prediction_set_id="p1", config=SimulationConfig(team_goal_anchor_weight=0.0)
    )


def test_anchored_simulation_differentiates_mismatched_fixtures():
    # E-6 acceptance: with the anchor on, team goal means track the Elo
    # expected-goals ordering and a favorite-vs-minnow game looks visibly
    # different from an even game — while determinism under seed holds.
    predictions = _active_predictions()
    config = SimulationConfig(n_sims=4000, seed=11, team_goal_anchor_weight=1.0)

    even = simulate_fixture(
        build_simulation_inputs(predictions, expected_team_goals={1: 1.3, 2: 1.3}),
        config,
    )
    mismatch = simulate_fixture(
        build_simulation_inputs(predictions, expected_team_goals={1: 3.0, 2: 0.5}),
        config,
    )

    favorite_goals = float(mismatch.team_totals("goals_total", 1).mean())
    underdog_goals = float(mismatch.team_totals("goals_total", 2).mean())
    even_goals = float(even.team_totals("goals_total", 1).mean())
    assert favorite_goals > even_goals > underdog_goals
    assert favorite_goals == pytest.approx(3.0, rel=0.10)
    assert underdog_goals == pytest.approx(0.5, rel=0.20)

    favorite_win_rate = float(
        (
            mismatch.team_totals("goals_total", 1)
            > mismatch.team_totals("goals_total", 2)
        ).mean()
    )
    even_win_rate = float(
        (even.team_totals("goals_total", 1) > even.team_totals("goals_total", 2)).mean()
    )
    assert favorite_win_rate > even_win_rate + 0.2

    # Structural chain intact and deterministic under the same seed.
    assert (mismatch.counts["goals_total"] <= mismatch.counts["shots_on"]).all()
    repeat = simulate_fixture(
        build_simulation_inputs(predictions, expected_team_goals={1: 3.0, 2: 0.5}),
        config,
    )
    np.testing.assert_array_equal(
        repeat.counts["goals_total"], mismatch.counts["goals_total"]
    )


def test_simulation_is_deterministic_under_seed():
    inputs = build_simulation_inputs(_active_predictions())
    config = SimulationConfig(n_sims=200, seed=23)
    first = simulate_fixture(inputs, config)
    second = simulate_fixture(inputs, config)
    for target in ALL_SIMULATION_TARGETS:
        np.testing.assert_array_equal(first.counts[target], second.counts[target])


def test_simulation_means_track_predicted_means():
    inputs = build_simulation_inputs(_active_predictions())
    result = simulate_fixture(inputs, SimulationConfig(n_sims=30000, seed=29))

    players = inputs.players
    starters = players["is_starting"].to_numpy()
    outfield = starters & (players["position_group"] != "G").to_numpy()

    # Starter outfielders: predicted mean = rate90 * 90/90 = rate90.
    simulated = result.counts["shots_total"][outfield].mean(axis=1)
    assert simulated == pytest.approx(1.5, rel=0.05)

    # Bench players: unconditional mean = p_plays * if_plays/90 * rate90.
    bench = ~starters & (players["position_group"] != "G").to_numpy()
    bench_mean = result.counts["shots_total"][bench].mean(axis=1)
    assert bench_mean == pytest.approx(0.4 * (25.0 / 90.0) * 1.5, rel=0.10)


def test_summarize_produces_player_team_and_scoreline_rows():
    import json

    from football_analytics.simulation import (
        SIMULATION_ENGINE_VERSION,
        summarize_simulation,
    )

    inputs = build_simulation_inputs(_active_predictions())
    config = SimulationConfig(n_sims=500, seed=41)
    result = simulate_fixture(inputs, config)
    summary = summarize_simulation(result, sim_set_id="sim-set-1")

    players, teams = 26, 2
    targets = len(ALL_SIMULATION_TARGETS)
    assert len(summary) == (players + teams) * targets + 1

    # Grain is unique.
    grain = ["fixture_id", "sim_set_id", "entity_type", "entity_id", "target_event"]
    assert not summary.duplicated(subset=grain).any()
    assert (summary["engine_version"] == SIMULATION_ENGINE_VERSION).all()
    assert summary["is_active_simulation"].all()
    assert (summary["prediction_set_id"] == "pred-set-1").all()

    player_rows = summary[summary["entity_type"] == "player"]
    assert (player_rows["p_ge_1"] >= player_rows["p_ge_2"]).all()
    assert (player_rows["p_ge_2"] >= player_rows["p_ge_3"]).all()
    assert (player_rows["p05"] <= player_rows["p95"]).all()

    # Team mean equals the sum of its players' means (allocation identity).
    for target in ("shots_total", "passes_total"):
        team_rows = summary[(summary["entity_type"] == "team") & (summary["target_event"] == target)]
        for row in team_rows.itertuples():
            player_sum = player_rows[
                (player_rows["team_id"] == row.team_id)
                & (player_rows["target_event"] == target)
            ]["sim_mean"].sum()
            assert row.sim_mean == pytest.approx(player_sum, rel=1e-9)

    scoreline = summary[summary["entity_type"] == "fixture"]
    assert len(scoreline) == 1
    matrix = json.loads(scoreline["score_matrix_json"].iloc[0])
    assert matrix["team_a_id"] == 1 and matrix["team_b_id"] == 2
    assert sum(matrix["matrix"].values()) == pytest.approx(1.0, abs=1e-6)


class _FakeWriter:
    def __init__(self, log):
        self._log = log

    def format(self, _fmt):
        return self

    def mode(self, mode):
        self._log["mode"] = mode
        return self

    def option(self, key, value):
        self._log.setdefault("options", {})[key] = value
        return self

    def saveAsTable(self, table):
        self._log["table"] = table


class _FakeColumn:
    def cast(self, _data_type):
        return self


class _FakeFrame:
    def __init__(self, log, columns):
        self.write = _FakeWriter(log)
        self.columns = list(columns)
        self.casts = log.setdefault("casts", [])

    def __getitem__(self, _name):
        return _FakeColumn()

    def withColumn(self, name, _column):
        self.casts.append(name)
        return self


class _FakeSpark:
    def __init__(self):
        self.sql_statements = []
        self.write_log = {}

    def sql(self, statement):
        self.sql_statements.append(" ".join(statement.split()))

    def createDataFrame(self, records):
        return _FakeFrame(self.write_log, records.columns)


def test_simulation_write_appends_and_flips_older_active_sets():
    from football_analytics.simulation import (
        SIMULATION_ENGINE_VERSION,
        summarize_simulation,
        write_fixture_simulation,
    )

    inputs = build_simulation_inputs(_active_predictions())
    result = simulate_fixture(inputs, SimulationConfig(n_sims=50, seed=43))
    records = summarize_simulation(result, sim_set_id="sim-set-1")
    spark = _FakeSpark()

    summary = write_fixture_simulation(
        spark, records, simulation_table="catalog.gold.sim_football__fixture_simulation"
    )

    assert summary["written"] == len(records)
    assert summary["deactivated_sets"] == 1
    assert spark.write_log["mode"] == "append"
    assert spark.write_log["options"].get("mergeSchema") == "true"

    create = spark.sql_statements[0]
    assert "CREATE TABLE IF NOT EXISTS" in create
    for column in ("sim_set_id", "entity_type", "score_matrix_json", "is_active_simulation"):
        assert column in create

    delete = spark.sql_statements[1]
    assert delete.startswith("DELETE FROM")
    assert "sim_set_id IN ('sim-set-1')" in delete

    updates = [s for s in spark.sql_statements if s.startswith("UPDATE")]
    assert len(updates) == 1
    assert "SET is_active_simulation = false" in updates[0]
    assert "fixture_id = 100" in updates[0]
    assert f"engine_version = '{SIMULATION_ENGINE_VERSION}'" in updates[0]
    assert "sim_set_id != 'sim-set-1'" in updates[0]


def test_minutes_sampling_is_seeded_and_matches_p_plays():
    inputs = build_simulation_inputs(_active_predictions())
    config = SimulationConfig(n_sims=4000, seed=11)

    minutes_a = sample_minutes(inputs, config, np.random.default_rng(config.seed))
    minutes_b = sample_minutes(inputs, config, np.random.default_rng(config.seed))
    np.testing.assert_array_equal(minutes_a, minutes_b)

    starters = inputs.players["is_starting"].to_numpy()
    # Starters always play their conditional expectation.
    assert (minutes_a[starters] == 90.0).all()
    # Bench players play 0 or if_plays, at the p_plays participation rate.
    bench_minutes = minutes_a[~starters]
    assert set(np.unique(bench_minutes)) == {0.0, 25.0}
    participation = (bench_minutes > 0).mean()
    assert participation == pytest.approx(0.4, abs=0.02)
