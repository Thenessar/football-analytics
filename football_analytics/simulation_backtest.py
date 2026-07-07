"""Historical backtest of simulation calibration (ml_upgrade_backlog.md N6).

Replays completed fixtures as if they were upcoming: pseudo-pre-match
predictions are built from the feature mart's training rows using the
registered models with ``expected_minutes`` exposure (never actual minutes),
each fixture is simulated, and the simulated distributions are scored
against realized labels: central-interval coverage, randomized PIT
uniformity, team-total coverage, and leader-allocation hit rates.

Pure pandas/numpy given models and feature rows — the Spark/MLflow wiring
lives in ``scripts/backtest_simulation.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from football_analytics.evaluation import (
    central_interval_coverage,
    pit_uniformity_summary,
    randomized_pit,
)
from football_analytics.inference import (
    LoadedEventModel,
    build_prediction_records,
    deterministic_prediction_set_id,
    fixture_has_confirmed_starting_xi,
    lineup_signature,
)
from football_analytics.simulation import (
    ALL_SIMULATION_TARGETS,
    SimulationConfig,
    SimulationInputError,
    SimulationResult,
    build_simulation_inputs,
    simulate_fixture,
)

LINEUP_SOURCE_BACKTEST = "offline_sample"


def simulate_completed_fixtures(
    feature_rows: pd.DataFrame,
    models: Sequence[LoadedEventModel],
    *,
    config: Optional[SimulationConfig] = None,
) -> Tuple[List[SimulationResult], pd.DataFrame]:
    """Simulates every fixture in ``feature_rows`` as-of pre-match.

    ``feature_rows`` are completed-fixture training rows from the feature
    mart (labels populated, ``expected_minutes`` present). Team-total
    dispersion comes straight from the models' ``alpha_team`` (M1). Returns
    the per-fixture results plus a skip log.
    """

    cfg = config or SimulationConfig()
    alpha_team = {model.target_event: float(model.alpha_team) for model in models}
    model_versions = {model.target_event: model.model_version for model in models}

    results: List[SimulationResult] = []
    skipped: List[Dict[str, object]] = []
    for fixture_id, fixture_rows in feature_rows.groupby("fixture_id"):
        if not fixture_has_confirmed_starting_xi(fixture_rows):
            skipped.append({
                "fixture_id": int(fixture_id),
                "reason": "no confirmed starting XI in feature rows",
            })
            continue
        prediction_set_id = deterministic_prediction_set_id(
            int(fixture_id),
            lineup_digest=lineup_signature(fixture_rows),
            model_versions=model_versions,
        )
        predictions = build_prediction_records(
            fixture_rows,
            models,
            prediction_set_id=prediction_set_id,
            prediction_run_id="simulation-backtest",
            lineup_source=LINEUP_SOURCE_BACKTEST,
            feature_table_name="backtest",
        )
        try:
            inputs = build_simulation_inputs(predictions, alpha_team=alpha_team)
        except SimulationInputError as error:
            skipped.append({"fixture_id": int(fixture_id), "reason": str(error)})
            continue
        results.append(simulate_fixture(inputs, cfg))
    return results, pd.DataFrame(skipped, columns=["fixture_id", "reason"])


def _actual_lookup(feature_rows: pd.DataFrame) -> pd.DataFrame:
    labels = feature_rows[
        ["fixture_id", "team_id", "player_id", *ALL_SIMULATION_TARGETS]
    ].copy()
    for target in ALL_SIMULATION_TARGETS:
        labels[target] = pd.to_numeric(labels[target], errors="coerce").fillna(0.0)
    return labels.set_index(["fixture_id", "team_id", "player_id"])


def score_simulation_backtest(
    results: Sequence[SimulationResult],
    feature_rows: pd.DataFrame,
    *,
    interval: Tuple[float, float] = (0.10, 0.90),
    pit_seed: int = 99,
) -> Dict[str, pd.DataFrame]:
    """Scores simulated distributions against realized labels.

    Returns per-target frames: ``player_coverage`` (interval coverage + PIT
    uniformity), ``team_coverage`` (team totals), and ``allocation``
    (top-1/top-3 leader hit rates from simulated means). Interval endpoints
    are quantiles of the simulated draws.
    """

    labels = _actual_lookup(feature_rows)
    lower_q, upper_q = interval
    rng = np.random.default_rng(pit_seed)

    coverage_rows = []
    team_rows = []
    allocation_rows = []
    for target in ALL_SIMULATION_TARGETS:
        actuals: List[float] = []
        lowers: List[float] = []
        uppers: List[float] = []
        cdf_at: List[float] = []
        cdf_below: List[float] = []
        team_actuals: List[float] = []
        team_lowers: List[float] = []
        team_uppers: List[float] = []
        top1_hits: List[float] = []
        top3_hits: List[float] = []

        for result in results:
            draws = result.counts[target]
            players = result.inputs.players
            fixture_id = result.inputs.fixture_id
            sim_means = draws.mean(axis=1)

            player_actuals = np.zeros(len(players))
            known = np.zeros(len(players), dtype=bool)
            for position, row in enumerate(players.itertuples()):
                key = (fixture_id, int(row.team_id), int(row.player_id))
                if key in labels.index:
                    player_actuals[position] = float(labels.loc[key, target])
                    known[position] = True
            if not known.any():
                continue

            quantiles = np.quantile(draws, [lower_q, upper_q], axis=1)
            actuals.extend(player_actuals[known])
            lowers.extend(quantiles[0][known])
            uppers.extend(quantiles[1][known])
            for position in np.flatnonzero(known):
                y = player_actuals[position]
                cdf_at.append(float((draws[position] <= y).mean()))
                cdf_below.append(float((draws[position] <= y - 1).mean()))

            for team_id in result.inputs.team_ids:
                mask = result.team_mask(team_id) & known
                if not mask.any():
                    continue
                team_actual = player_actuals[mask].sum()
                team_draws = draws[result.team_mask(team_id)].sum(axis=0)
                team_actuals.append(team_actual)
                team_lowers.append(float(np.quantile(team_draws, lower_q)))
                team_uppers.append(float(np.quantile(team_draws, upper_q)))

                team_mask = result.team_mask(team_id)
                masked_actuals = np.where(known & team_mask, player_actuals, -np.inf)
                if (masked_actuals > 0).any():
                    leader = int(np.argmax(np.where(team_mask, sim_means, -np.inf)))
                    best = masked_actuals.max()
                    top1_hits.append(float(player_actuals[leader] == best))
                    positive = np.sort(masked_actuals[masked_actuals > -np.inf])
                    third_best = positive[-min(3, len(positive))]
                    top3_hits.append(float(player_actuals[leader] >= third_best))

        pit = randomized_pit(actuals, cdf_at, cdf_below, rng)
        pit_summary = pit_uniformity_summary(pit)
        coverage_rows.append({
            "target_event": target,
            "players_scored": len(actuals),
            "interval_lower_q": lower_q,
            "interval_upper_q": upper_q,
            "coverage": central_interval_coverage(actuals, lowers, uppers),
            "pit_max_abs_deviation": pit_summary["max_abs_deviation"],
        })
        team_rows.append({
            "target_event": target,
            "team_fixtures_scored": len(team_actuals),
            "coverage": central_interval_coverage(team_actuals, team_lowers, team_uppers),
            "mean_actual": float(np.mean(team_actuals)) if team_actuals else float("nan"),
        })
        allocation_rows.append({
            "target_event": target,
            "team_fixtures_scored": len(top1_hits),
            "top1_hit_rate": float(np.mean(top1_hits)) if top1_hits else float("nan"),
            "top3_hit_rate": float(np.mean(top3_hits)) if top3_hits else float("nan"),
        })

    return {
        "player_coverage": pd.DataFrame(coverage_rows),
        "team_coverage": pd.DataFrame(team_rows),
        "allocation": pd.DataFrame(allocation_rows),
    }
