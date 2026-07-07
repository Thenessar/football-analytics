"""Fixture simulation engine tests (ml_upgrade_backlog.md Epic N)."""

import numpy as np
import pandas as pd
import pytest

from football_analytics.simulation import (
    ALL_SIMULATION_TARGETS,
    SimulationConfig,
    SimulationInputError,
    build_simulation_inputs,
    deterministic_sim_set_id,
    sample_minutes,
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
