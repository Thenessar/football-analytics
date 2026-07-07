"""Monte Carlo fixture simulation (ml_upgrade_backlog.md Epic N, ADR 0005).

Produces a coherent full-game view from the ACTIVE prediction set: per-player
counts for all 11 target events plus team totals and a scoreline
distribution, sampled jointly so structural identities hold in every
iteration (player goals <= shots on target <= shots; team A fouls committed
= team B fouls drawn; goalkeeper saves = opponent on-target shots minus
goals; assists <= goals; card caps).

Pure numpy/pandas — no Spark or MLflow imports at module scope. Notebook 05
wires the effects. All randomness flows through one seeded
``numpy.random.Generator``: identical inputs + config produce bit-identical
outputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SIMULATION_ENGINE_VERSION = "1.0.0"
SIMULATION_TABLE_NAME = "sim_football__fixture_simulation"

# All 11 targets the simulator emits, grouped by sampling treatment.
CHAIN_TARGETS = ("shots_total", "shots_on", "goals_total")
INDEPENDENT_NB_TARGETS = ("passes_total", "offsides", "cards_yellow")
MIRRORED_FOUL_TARGETS = ("fouls_committed", "fouls_drawn")
DERIVED_TARGETS = ("goals_saves", "goals_assists")
BERNOULLI_TARGETS = ("cards_red",)
ALL_SIMULATION_TARGETS = (
    CHAIN_TARGETS
    + INDEPENDENT_NB_TARGETS
    + MIRRORED_FOUL_TARGETS
    + DERIVED_TARGETS
    + BERNOULLI_TARGETS
)

_MIN_EXPOSURE = 1e-6


class SimulationInputError(ValueError):
    """Raised when the active prediction set cannot support a simulation."""


@dataclass(frozen=True)
class SimulationConfig:
    """Run parameters; the digest participates in the deterministic set id."""

    n_sims: int = 10_000
    seed: int = 7
    thresholds: Tuple[int, ...] = (1, 2, 3)
    # Empirical share of goals with a recorded assist; provenance query in
    # ADR 0005 (ticket N3 calibrates the default).
    assist_per_goal_rate: float = 0.72
    max_yellow_per_player: int = 2
    max_red_probability: float = 0.5

    def digest(self) -> str:
        payload = {
            "n_sims": int(self.n_sims),
            "seed": int(self.seed),
            "thresholds": [int(t) for t in self.thresholds],
            "assist_per_goal_rate": float(self.assist_per_goal_rate),
            "max_yellow_per_player": int(self.max_yellow_per_player),
            "max_red_probability": float(self.max_red_probability),
            "engine_version": SIMULATION_ENGINE_VERSION,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass
class SimulationInputs:
    """Pivoted per-player inputs for one fixture.

    ``players`` has one row per player with identity columns
    (team_id, player_id, player_name, position_group, is_starting),
    the minutes decomposition (p_plays, expected_minutes_if_plays), and one
    ``rate90_<target>`` column per target (per-90 intensity recovered from
    the active prediction set).
    """

    fixture_id: int
    prediction_set_id: str
    players: pd.DataFrame
    team_ids: Tuple[int, int]
    alpha_team: Dict[str, float] = field(default_factory=dict)

    def rates(self, target: str) -> np.ndarray:
        return self.players[f"rate90_{target}"].to_numpy(dtype=float)


def deterministic_sim_set_id(
    fixture_id: int,
    *,
    prediction_set_id: str,
    config: SimulationConfig,
) -> str:
    """Same fixture + active prediction set + config ⇒ same sim_set_id.

    Mirrors ``deterministic_prediction_set_id`` (§15.4 Option C): reruns
    replace themselves instead of stacking new active sets.
    """

    payload = {
        "fixture_id": int(fixture_id),
        "prediction_set_id": str(prediction_set_id),
        "config_digest": config.digest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_simulation_inputs(
    active_predictions: pd.DataFrame,
    *,
    alpha_team: Optional[Mapping[str, float]] = None,
) -> SimulationInputs:
    """Pivots one fixture's active prediction set into simulation inputs.

    Per-90 intensity is recovered as ``predicted_mean / expected_exposure``
    (backlog §2.5): the simulator consumes the governed prediction table, not
    the models. Rows missing the K5 decomposition degrade gracefully:
    ``p_plays`` defaults to 1 and ``expected_minutes_if_plays`` to
    ``expected_minutes`` (deterministic minutes, no participation sampling).
    """

    required = {
        "fixture_id", "team_id", "player_id", "target_event",
        "predicted_mean", "expected_minutes",
    }
    missing = required - set(active_predictions.columns)
    if missing:
        raise SimulationInputError(f"Prediction rows missing columns: {sorted(missing)}")
    if active_predictions.empty:
        raise SimulationInputError("No active prediction rows supplied")

    fixture_ids = active_predictions["fixture_id"].unique()
    if len(fixture_ids) != 1:
        raise SimulationInputError(
            f"Simulation inputs must cover exactly one fixture, got {sorted(fixture_ids)}"
        )
    fixture_id = int(fixture_ids[0])

    targets_present = set(active_predictions["target_event"].unique())
    missing_targets = [t for t in ALL_SIMULATION_TARGETS if t not in targets_present]
    if missing_targets:
        raise SimulationInputError(
            f"Active prediction set lacks targets required by the simulator: {missing_targets}"
        )

    team_ids = sorted(int(t) for t in active_predictions["team_id"].unique())
    if len(team_ids) != 2:
        raise SimulationInputError(f"Expected exactly 2 teams, got {team_ids}")

    prediction_set_ids = active_predictions.get("prediction_set_id")
    prediction_set_id = (
        str(prediction_set_ids.iloc[0]) if prediction_set_ids is not None else ""
    )

    identity_cols = ["team_id", "player_id"]
    carry_cols = [
        column
        for column in (
            "player_name", "position_group", "is_starting",
            "expected_minutes", "p_plays", "expected_minutes_if_plays",
        )
        if column in active_predictions.columns
    ]
    players = (
        active_predictions.sort_values(identity_cols)
        .groupby(identity_cols, as_index=False)[carry_cols]
        .first()
    )

    for team_id in team_ids:
        team_size = int((players["team_id"] == team_id).sum())
        if team_size < 11:
            raise SimulationInputError(
                f"Team {team_id} has only {team_size} predicted players; need >= 11"
            )

    minutes = pd.to_numeric(players["expected_minutes"], errors="coerce").fillna(0.0)
    if "p_plays" in players.columns:
        players["p_plays"] = (
            pd.to_numeric(players["p_plays"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        )
    else:
        players["p_plays"] = 1.0
    if "expected_minutes_if_plays" in players.columns:
        players["expected_minutes_if_plays"] = pd.to_numeric(
            players["expected_minutes_if_plays"], errors="coerce"
        ).fillna(minutes).clip(0.0, 120.0)
    else:
        players["expected_minutes_if_plays"] = minutes.clip(0.0, 120.0)
    if "is_starting" in players.columns:
        players["is_starting"] = players["is_starting"].fillna(False).astype(bool)
    else:
        players["is_starting"] = True

    exposure = np.clip(minutes.to_numpy(dtype=float) / 90.0, _MIN_EXPOSURE, None)
    pivot = active_predictions.pivot_table(
        index=identity_cols, columns="target_event", values="predicted_mean", aggfunc="first"
    )
    pivot = pivot.reindex(
        pd.MultiIndex.from_frame(players[identity_cols]), fill_value=0.0
    )
    for target in ALL_SIMULATION_TARGETS:
        means = pd.to_numeric(pivot[target], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        players[f"rate90_{target}"] = np.clip(means, 0.0, None) / exposure

    return SimulationInputs(
        fixture_id=fixture_id,
        prediction_set_id=prediction_set_id,
        players=players.reset_index(drop=True),
        team_ids=(team_ids[0], team_ids[1]),
        alpha_team={
            str(k): max(0.0, float(v)) for k, v in dict(alpha_team or {}).items()
        },
    )


def sample_minutes(
    inputs: SimulationInputs,
    config: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Minutes matrix (players x sims) from the ADR 0004 decomposition.

    Starters play their conditional expectation deterministically (v1
    simplification, ADR 0005); bench players either come on for their
    conditional expectation (probability ``p_plays``) or stay unused (0
    minutes) — the bimodal behavior the mean-based prediction cannot express.
    """

    players = inputs.players
    p_plays = players["p_plays"].to_numpy(dtype=float)[:, None]
    if_plays = players["expected_minutes_if_plays"].to_numpy(dtype=float)[:, None]
    is_starting = players["is_starting"].to_numpy(dtype=bool)[:, None]

    participation = np.where(
        is_starting,
        1.0,
        rng.random((len(players), config.n_sims)) < p_plays,
    )
    return participation * if_plays


def sample_nb_totals(
    rng: np.random.Generator,
    means: np.ndarray,
    alpha: float = 0.0,
) -> np.ndarray:
    """Team totals from NB(mean, Var = mean + alpha*mean^2) per simulation.

    Realized as the Poisson–Gamma mixture (rate ~ Gamma(1/alpha, alpha*mean),
    total ~ Poisson(rate)); alpha = 0 degenerates to a plain Poisson draw.
    """

    means = np.clip(np.asarray(means, dtype=float), 0.0, None)
    if alpha and alpha > 0.0:
        rates = rng.gamma(1.0 / alpha, alpha * means)
        return rng.poisson(rates).astype(np.int64)
    return rng.poisson(means).astype(np.int64)


def multinomial_allocate(
    rng: np.random.Generator,
    totals: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Exact multinomial split of ``totals`` (sims,) by ``weights`` (P, sims).

    Sequential conditional binomials with suffix-sum denominators — the
    textbook decomposition of a multinomial — vectorized across simulations
    and independent of numpy's batched-multinomial support. Columns whose
    weights are all zero keep their counts unallocated only if the caller
    passed a positive total with no weight anywhere; ``allocate_event_totals``
    pre-repairs that case, so here the last positive-weight player absorbs
    the exact remainder (its conditional ratio is exactly 1.0).
    """

    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    n_players, n_sims = weights.shape
    suffix = np.cumsum(weights[::-1], axis=0)[::-1]
    remaining = np.asarray(totals, dtype=np.int64).copy()
    out = np.zeros((n_players, n_sims), dtype=np.int64)
    for index in range(n_players - 1):
        denominator = suffix[index]
        ratio = np.divide(
            weights[index],
            denominator,
            out=np.zeros(n_sims, dtype=float),
            where=denominator > 0.0,
        )
        draw = rng.binomial(remaining, np.clip(ratio, 0.0, 1.0))
        out[index] = draw
        remaining -= draw
    out[n_players - 1] = remaining
    return out


def allocate_event_totals(
    rng: np.random.Generator,
    totals: np.ndarray,
    weights: np.ndarray,
    on_pitch: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multinomial allocation with the degenerate-weight edges repaired.

    Simulations where the total is positive but every weight is zero fall
    back to a uniform split across ``on_pitch`` players; if nobody is on the
    pitch the total itself is forced to zero (you cannot record events with
    no players). Returns ``(counts, repaired_totals)``.
    """

    weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    totals = np.asarray(totals, dtype=np.int64).copy()
    weight_sums = weights.sum(axis=0)
    degenerate = (weight_sums <= 0.0) & (totals > 0)
    if degenerate.any():
        pitch_counts = on_pitch.sum(axis=0)
        repairable = degenerate & (pitch_counts > 0)
        if repairable.any():
            weights = weights.copy()
            weights[:, repairable] = on_pitch[:, repairable].astype(float)
        totals[degenerate & (pitch_counts == 0)] = 0
    return multinomial_allocate(rng, totals, weights), totals
