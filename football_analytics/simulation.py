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
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
    # Empirical share of goals with a recorded assist, calibrated from
    # completed international fixtures on 2026-07-07 (0.688; provenance query
    # in ADR 0005).
    assist_per_goal_rate: float = 0.688
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


@dataclass
class SimulationResult:
    """Raw draws for one fixture: counts[target] is a (players x sims) matrix."""

    inputs: SimulationInputs
    config: SimulationConfig
    minutes: np.ndarray
    counts: Dict[str, np.ndarray]

    def team_mask(self, team_id: int) -> np.ndarray:
        return (self.inputs.players["team_id"] == team_id).to_numpy()

    def team_totals(self, target: str, team_id: int) -> np.ndarray:
        """Per-simulation team total for one event (sims,)."""

        return self.counts[target][self.team_mask(team_id)].sum(axis=0)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise ratio clipped to [0, 1]; 0 where the denominator is 0."""

    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(np.asarray(numerator, dtype=float)),
        where=np.asarray(denominator) > 0.0,
    )
    return np.clip(ratio, 0.0, 1.0)


def _starting_goalkeeper_index(players: pd.DataFrame, team_id: int) -> int:
    """Row index of the team's starting goalkeeper.

    Falls back to the team player with the highest saves intensity when the
    lineup carries no explicit starting G (position data errors happen).
    """

    team = players[players["team_id"] == team_id]
    goalkeepers = team[
        team.get("position_group", pd.Series(dtype=object)).eq("G")
        & team["is_starting"]
    ]
    if not goalkeepers.empty:
        return int(goalkeepers.index[0])
    return int(team["rate90_goals_saves"].idxmax())


def simulate_fixture(
    inputs: SimulationInputs,
    config: Optional[SimulationConfig] = None,
) -> SimulationResult:
    """Runs the full coherent Monte Carlo game (ADR 0005 sampling design).

    Draw order is fixed, so identical inputs + config are bit-identical:
    minutes → volume events (shots_total, passes_total, offsides,
    cards_yellow) → shot chain thinning → fouls mirror → saves identity →
    assists → red cards.
    """

    cfg = config or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    players = inputs.players
    n_players = len(players)

    minutes = sample_minutes(inputs, cfg, rng)
    exposure = minutes / 90.0
    on_pitch = exposure > 0.0
    team_masks = {
        team_id: (players["team_id"] == team_id).to_numpy()
        for team_id in inputs.team_ids
    }

    def weights_for(target: str) -> np.ndarray:
        return inputs.rates(target)[:, None] * exposure

    counts: Dict[str, np.ndarray] = {}

    # 1. Volume events: NB team total → multinomial allocation, per team.
    for target in ("shots_total", "passes_total", "offsides", "cards_yellow"):
        alpha = inputs.alpha_team.get(target, 0.0)
        allocated_all = np.zeros((n_players, cfg.n_sims), dtype=np.int64)
        raw_weights = weights_for(target)
        for team_id, mask in team_masks.items():
            team_weights = np.where(mask[:, None], raw_weights, 0.0)
            totals = sample_nb_totals(rng, team_weights.sum(axis=0), alpha)
            allocated, _ = allocate_event_totals(
                rng, totals, team_weights, mask[:, None] & on_pitch
            )
            allocated_all += allocated
        counts[target] = allocated_all
    # Cap after allocation: a fourth yellow to one player is a data
    # impossibility; the cap rarely binds and the tiny total shift is
    # accepted (ADR 0005).
    counts["cards_yellow"] = np.minimum(counts["cards_yellow"], cfg.max_yellow_per_player)

    # 2. Shot chain: binomial thinning keeps goals <= shots_on <= shots_total
    # in every draw. Ratios come from per-90 intensities per player.
    q_on = _safe_ratio(inputs.rates("shots_on"), inputs.rates("shots_total"))
    counts["shots_on"] = rng.binomial(counts["shots_total"], q_on[:, None])
    q_goal = _safe_ratio(inputs.rates("goals_total"), inputs.rates("shots_on"))
    counts["goals_total"] = rng.binomial(counts["shots_on"], q_goal[:, None])

    # 3. Fouls mirror: one draw per directed pair, allocated to both sides,
    # so committed(A) == drawn(B) exactly. Totals are validated against both
    # teams' pitches first so the two allocations always share the total.
    committed = np.zeros((n_players, cfg.n_sims), dtype=np.int64)
    drawn = np.zeros((n_players, cfg.n_sims), dtype=np.int64)
    alpha_fouls = inputs.alpha_team.get("fouls_committed", 0.0)
    w_committed = weights_for("fouls_committed")
    w_drawn = weights_for("fouls_drawn")
    team_a, team_b = inputs.team_ids
    for committing_team, drawing_team in ((team_a, team_b), (team_b, team_a)):
        commit_mask = team_masks[committing_team][:, None]
        draw_mask = team_masks[drawing_team][:, None]
        commit_weights = np.where(commit_mask, w_committed, 0.0)
        draw_weights = np.where(draw_mask, w_drawn, 0.0)
        pair_means = 0.5 * (commit_weights.sum(axis=0) + draw_weights.sum(axis=0))
        totals = sample_nb_totals(rng, pair_means, alpha_fouls)
        both_on_pitch = (
            ((commit_mask & on_pitch).sum(axis=0) > 0)
            & ((draw_mask & on_pitch).sum(axis=0) > 0)
        )
        totals = np.where(both_on_pitch, totals, 0)
        allocated_commit, _ = allocate_event_totals(
            rng, totals, commit_weights, commit_mask & on_pitch
        )
        allocated_drawn, _ = allocate_event_totals(
            rng, totals, draw_weights, draw_mask & on_pitch
        )
        committed += allocated_commit
        drawn += allocated_drawn
    counts["fouls_committed"] = committed
    counts["fouls_drawn"] = drawn

    # 4. Saves identity: the starting goalkeeper faces the opponent's
    # on-target shots; whatever does not go in, he saved (ADR 0005 §5-6).
    saves = np.zeros((n_players, cfg.n_sims), dtype=np.int64)
    for team_id, opponent_id in ((team_a, team_b), (team_b, team_a)):
        opponent_mask = team_masks[opponent_id]
        opponent_on_target = counts["shots_on"][opponent_mask].sum(axis=0)
        opponent_goals = counts["goals_total"][opponent_mask].sum(axis=0)
        goalkeeper_index = _starting_goalkeeper_index(players, team_id)
        saves[goalkeeper_index] = np.maximum(opponent_on_target - opponent_goals, 0)
    counts["goals_saves"] = saves

    # 5. Assists: at most one per goal (Binomial(team goals, assist rate)),
    # allocated by assist intensity — assists <= goals per team by design.
    assists = np.zeros((n_players, cfg.n_sims), dtype=np.int64)
    w_assists = weights_for("goals_assists")
    for team_id, mask in team_masks.items():
        team_goals = counts["goals_total"][mask].sum(axis=0)
        assist_totals = rng.binomial(team_goals, cfg.assist_per_goal_rate)
        team_weights = np.where(mask[:, None], w_assists, 0.0)
        allocated, _ = allocate_event_totals(
            rng, assist_totals, team_weights, mask[:, None] & on_pitch
        )
        assists += allocated
    counts["goals_assists"] = assists

    # 6. Red cards: per-player Bernoulli, capped at 1 by construction.
    red_probability = np.clip(
        inputs.rates("cards_red")[:, None] * exposure, 0.0, cfg.max_red_probability
    )
    counts["cards_red"] = rng.binomial(1, red_probability)

    return SimulationResult(inputs=inputs, config=cfg, minutes=minutes, counts=counts)


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


def _distribution_stats(draws: np.ndarray, thresholds: Sequence[int]) -> Dict[str, float]:
    """Summary statistics of one count distribution (sims,)."""

    draws = np.asarray(draws, dtype=float)
    percentiles = np.percentile(draws, [5, 25, 50, 75, 95])
    stats = {
        "sim_mean": float(draws.mean()),
        "sim_std": float(draws.std()),
        "p05": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p50": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
    }
    for threshold in thresholds:
        stats[f"p_ge_{threshold}"] = float((draws >= threshold).mean())
    return stats


def _score_matrix_json(result: SimulationResult) -> str:
    """Scoreline probability matrix as compact JSON, keyed "goalsA-goalsB"."""

    team_a, team_b = result.inputs.team_ids
    goals_a = result.team_totals("goals_total", team_a)
    goals_b = result.team_totals("goals_total", team_b)
    pairs, counts = np.unique(np.stack([goals_a, goals_b]), axis=1, return_counts=True)
    n_sims = float(len(goals_a))
    matrix = {
        f"{int(a)}-{int(b)}": round(count / n_sims, 6)
        for a, b, count in zip(pairs[0], pairs[1], counts)
    }
    return json.dumps({
        "team_a_id": int(team_a),
        "team_b_id": int(team_b),
        "matrix": matrix,
    })


def summarize_simulation(
    result: SimulationResult,
    *,
    sim_set_id: str,
    created_at_utc: Optional[datetime] = None,
) -> pd.DataFrame:
    """Flattens raw draws into the sim table rows (backlog N4).

    Grain: fixture_id / sim_set_id / entity_type ('player'|'team'|'fixture')
    / entity_id / target_event. The fixture-level row carries the scoreline
    matrix; player and team rows carry distribution statistics.
    """

    inputs = result.inputs
    config = result.config
    created_at = created_at_utc or datetime.now(timezone.utc)
    base = {
        "fixture_id": inputs.fixture_id,
        "sim_set_id": sim_set_id,
        "prediction_set_id": inputs.prediction_set_id,
        "n_sims": int(config.n_sims),
        "seed": int(config.seed),
        "engine_version": SIMULATION_ENGINE_VERSION,
        "score_matrix_json": None,
        "created_at_utc": created_at,
        "is_active_simulation": True,
    }

    records = []
    players = inputs.players
    for target in ALL_SIMULATION_TARGETS:
        draws = result.counts[target]
        for position, row in enumerate(players.itertuples()):
            records.append({
                **base,
                "entity_type": "player",
                "entity_id": int(row.player_id),
                "entity_name": getattr(row, "player_name", None),
                "team_id": int(row.team_id),
                "position_group": getattr(row, "position_group", None),
                "is_starting": bool(row.is_starting),
                "target_event": target,
                **_distribution_stats(draws[position], config.thresholds),
            })
        for team_id in inputs.team_ids:
            records.append({
                **base,
                "entity_type": "team",
                "entity_id": int(team_id),
                "entity_name": None,
                "team_id": int(team_id),
                "position_group": None,
                "is_starting": None,
                "target_event": target,
                **_distribution_stats(
                    result.team_totals(target, team_id), config.thresholds
                ),
            })

    records.append({
        **base,
        "entity_type": "fixture",
        "entity_id": int(inputs.fixture_id),
        "entity_name": None,
        "team_id": None,
        "position_group": None,
        "is_starting": None,
        "target_event": "scoreline",
        "sim_mean": None,
        "sim_std": None,
        "p05": None,
        "p25": None,
        "p50": None,
        "p75": None,
        "p95": None,
        **{f"p_ge_{t}": None for t in config.thresholds},
        "score_matrix_json": _score_matrix_json(result),
    })
    return pd.DataFrame.from_records(records)


def ensure_fixture_simulation_table(spark: Any, simulation_table: str) -> None:
    """Creates the N4 simulation Delta table when missing."""

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {simulation_table} (
            fixture_id INT,
            sim_set_id STRING,
            prediction_set_id STRING,
            entity_type STRING,
            entity_id BIGINT,
            entity_name STRING,
            team_id INT,
            position_group STRING,
            is_starting BOOLEAN,
            target_event STRING,
            n_sims INT,
            seed INT,
            engine_version STRING,
            sim_mean DOUBLE,
            sim_std DOUBLE,
            p05 DOUBLE,
            p25 DOUBLE,
            p50 DOUBLE,
            p75 DOUBLE,
            p95 DOUBLE,
            p_ge_1 DOUBLE,
            p_ge_2 DOUBLE,
            p_ge_3 DOUBLE,
            score_matrix_json STRING,
            created_at_utc TIMESTAMP,
            is_active_simulation BOOLEAN
        )
        USING DELTA
    """)


# Must mirror ensure_fixture_simulation_table; strings left to inference.
_SIMULATION_COLUMN_TYPES = {
    "fixture_id": "int",
    "entity_id": "bigint",
    "team_id": "int",
    "is_starting": "boolean",
    "n_sims": "int",
    "seed": "int",
    "sim_mean": "double",
    "sim_std": "double",
    "p05": "double",
    "p25": "double",
    "p50": "double",
    "p75": "double",
    "p95": "double",
    "p_ge_1": "double",
    "p_ge_2": "double",
    "p_ge_3": "double",
    "created_at_utc": "timestamp",
    "is_active_simulation": "boolean",
}


def write_fixture_simulation(
    spark: Any,
    records: pd.DataFrame,
    *,
    simulation_table: str,
) -> Dict[str, int]:
    """Stores simulation summaries with the shared Option C semantics.

    Same sim_set_id reruns replace themselves; older sets for the same
    fixture + engine version flip to is_active_simulation=false.
    """

    from football_analytics.delta_write import write_active_flag_records

    return write_active_flag_records(
        spark,
        records,
        table=simulation_table,
        ensure_table=ensure_fixture_simulation_table,
        column_types=_SIMULATION_COLUMN_TYPES,
        set_id_column="sim_set_id",
        flip_key_columns=("fixture_id", "engine_version"),
        active_flag_column="is_active_simulation",
    )
