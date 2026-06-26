import json

import pandas as pd

from football_analytics.modeling import (
    add_recency_weights,
    calculate_common_opponent_modifiers,
    map_position,
    run_player_monte_carlo,
    run_squad_simulation,
)


def test_map_position():
    assert map_position("forward") == "F"
    assert map_position("ATTACKER") == "F"
    assert map_position("M") == "M"
    assert map_position("D") == "D"
    assert map_position("g") == "G"
    assert map_position("unknown") == "M"


def test_common_opponent_bridge():
    mod_away, mod_home = calculate_common_opponent_modifiers(
        "Germany",
        "Ecuador",
        "world_cup_2026_completed_data.json",
    )
    assert 0.5 <= mod_away <= 1.5
    assert 0.5 <= mod_home <= 1.5


def test_common_opponent_modifiers_are_independent_and_asymmetric(tmp_path):
    def match(home, away, home_shots, away_shots):
        def team_payload(team, shots):
            return {
                "team": {"name": team},
                "players": [{
                    "player": {"id": 1, "name": f"{team} Player"},
                    "statistics": [{
                        "shots": {"total": shots},
                    }],
                }],
            }

        return {
            "match_info": {
                "home_team": home,
                "away_team": away,
            },
            "player_statistics": [
                team_payload(home, home_shots),
                team_payload(away, away_shots),
            ],
        }

    database = {
        "1": match("Team A", "Opponent X", 10, 4),
        "2": match("Team A", "Opponent Y", 10, 6),
        "3": match("Team B", "Opponent X", 10, 10),
        "4": match("Team B", "Opponent Y", 10, 5),
        "5": match("Team C", "Opponent X", 10, 8),
        "6": match("Team C", "Opponent Y", 10, 7),
    }
    database_path = tmp_path / "asymmetric.json"
    database_path.write_text(json.dumps(database), encoding="utf-8")

    away_factor, home_factor = calculate_common_opponent_modifiers(
        "Team A",
        "Team B",
        str(database_path),
    )

    assert abs(home_factor - (10 / 15)) < 1e-6
    assert abs(away_factor - (15 / 12.5)) < 1e-6
    assert round((home_factor - 1) * 100) == -33
    assert round((away_factor - 1) * 100) == 20


def test_bayesian_smoothing_and_monte_carlo():
    mock_history = pd.DataFrame([{
        "games_minutes": 90,
        "shots_total": 5,
        "shots_on": 5,
        "goals_total": 5,
    }])

    res = run_player_monte_carlo(
        player_name="Super Striker",
        position="F",
        player_history=mock_history,
        opp_def_factor=1.0,
        sims=1000,
    )

    assert res["player_name"] == "Super Striker"
    assert abs(res["smoothing_weight"] - 0.2) < 1e-6
    assert abs(res["adjusted_lambda"] - 3.08) < 0.05
    assert abs(res["sot_pct"] - 0.536) < 0.05
    assert abs(res["conversion"] - 0.328) < 0.05
    assert len(res["sim_shots"]) == 1000
    assert len(res["sim_sot"]) == 1000
    assert len(res["sim_goals"]) == 1000
    assert len(res["sim_missed"]) == 1000
    assert 0.0 <= res["any_shot_probability"] <= 1.0


def test_recency_decay_penalizes_old_shooting_volume():
    history = pd.DataFrame([
        {
            "fixture_date": "2023-06-26T00:00:00+00:00",
            "games_minutes": 90,
            "shots_total": 9,
            "shots_on": 6,
            "goals_total": 3,
        },
        {
            "fixture_date": "2026-06-12T00:00:00+00:00",
            "games_minutes": 90,
            "shots_total": 0,
            "shots_on": 0,
            "goals_total": 0,
        },
        {
            "fixture_date": "2026-06-19T00:00:00+00:00",
            "games_minutes": 90,
            "shots_total": 1,
            "shots_on": 0,
            "goals_total": 0,
        },
        {
            "fixture_date": "2026-06-25T00:00:00+00:00",
            "games_minutes": 90,
            "shots_total": 0,
            "shots_on": 0,
            "goals_total": 0,
        },
    ])

    weighted = add_recency_weights(history, "2026-06-26", half_life_days=365)
    assert weighted.iloc[0]["recency_weight"] < 0.13
    assert weighted.iloc[-1]["recency_weight"] > 0.99

    decayed = run_player_monte_carlo(
        player_name="Aging Finisher",
        position="F",
        player_history=history,
        opp_def_factor=1.0,
        sims=100,
        target_date="2026-06-26",
        recency_half_life_days=365,
    )
    unweighted = run_player_monte_carlo(
        player_name="Aging Finisher",
        position="F",
        player_history=history.drop(columns=["fixture_date"]),
        opp_def_factor=1.0,
        sims=100,
    )

    assert decayed["adjusted_lambda"] < unweighted["adjusted_lambda"]
    assert decayed["minutes_played"] < unweighted["minutes_played"]


def test_squad_history_join_is_accent_insensitive():
    history = pd.DataFrame([{
        "player_name": "Pervis Estupiñán",
        "games_minutes": 90,
        "shots_total": 2,
        "shots_on": 1,
        "goals_total": 0,
    }])
    results = run_squad_simulation(
        [{"name": "Pervis Estupinan", "position": "D", "team": "Ecuador"}],
        history,
        opp_def_factor=1.0,
        sims=20,
    )
    assert results[0]["minutes_played"] == 90
    assert results[0]["team"] == "Ecuador"
    assert "any_shot_probability" in results[0]
