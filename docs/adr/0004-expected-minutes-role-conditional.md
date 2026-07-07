# ADR 0004: Role-Conditional Expected Minutes

- **Status:** Accepted
- **Date:** 2026-07-07
- **Supersedes:** ADR 0002
- **Relates to:** business_logic.md §12; ml_upgrade_backlog.md §2.1, tickets K1–K5

## Problem

ADR 0002's estimator (trimmed mean of the last 5 played appearances) is
role-blind: `is_starting` was consulted only for players with zero history.
Consequences observed in production predictions:

- a regular 90-minute starter named **on the bench** kept an expected-minutes
  estimate of ~90, inflating every count prediction for him ~5x;
- a habitual substitute promoted to the XI kept his ~20-minute cameo average;
- unused-sub namings (0 minutes) were invisible to the estimator, so bench
  estimates never priced in `P(does not come on)`.

## Decision

Decompose conditional on the player's role in **today's confirmed lineup**:

```text
expected_minutes = p_plays * expected_minutes_if_plays
```

- **Confirmed starter:** `p_plays = 1`. `if_plays` = trimmed mean of minutes
  over the last 5 prior **starts** (drop single min + max when >=3
  observations, plain mean with 1-2), shrunk toward the starter prior with
  `k = 2`: `(n*hist + 2*prior) / (n + 2)`.
- **Confirmed bench:** `p_plays` = share of the last 10 prior **bench
  namings** with >=1 minute (unused-sub namings count as 0-minute
  observations), shrunk with `k = 4`. `if_plays` = trimmed mean of the
  nonzero bench cameos, shrunk toward the cameo prior with `k = 2`.

Role history comes from `stg_football__player_match_stats.games_substitute`
(the provider's own flag). Stats rows exist only for completed fixtures, so
future fixtures can never contribute phantom bench zeros, and coverage does
not depend on historical lineup ingestion. Today's role comes from
`stg_football__lineups.is_starting`.

The mart exposes the decomposition (`p_plays`, `expected_minutes_if_plays`)
because the downstream Monte Carlo simulation needs the bimodal bench
behavior (either ~20+ minutes or nothing), while the Poisson/NB mean
predictions only need the product — the count mean is linear in exposure, so
using `E[minutes]` as the offset is exact even when minutes are bimodal.

## Prior calibration (2026-07-07)

Priors are dbt vars, measured over **all completed international player rows**
(n = 90,470) with:

```sql
SELECT
  count(*) AS completed_player_rows,
  avg(CASE WHEN NOT coalesce(games_substitute, false)
           THEN least(cast(coalesce(games_minutes, 0) as double), 130.0)
      END) AS starter_mean_minutes,          -- 80.41
  avg(CASE WHEN coalesce(games_substitute, false) AND coalesce(games_minutes, 0) >= 1 THEN 1.0
           WHEN coalesce(games_substitute, false) THEN 0.0
      END) AS bench_participation_rate,      -- 0.3952
  avg(CASE WHEN coalesce(games_substitute, false) AND coalesce(games_minutes, 0) >= 1
           THEN least(cast(games_minutes as double), 130.0)
      END) AS bench_cameo_mean_minutes       -- 22.97
FROM football_analytics.silver.stg_football__player_match_stats
```

| dbt var | value | measured |
| --- | --- | --- |
| `starter_minutes_prior` | 80.0 | 80.41 |
| `bench_participation_prior` | 0.4 | 0.3952 |
| `bench_cameo_minutes_prior` | 23.0 | 22.97 |

Notable: a no-history bench player is now priced at `0.4 * 23 ≈ 9.2` expected
minutes versus the old flat 15, and international starters average 80 — not
90 — minutes (heavy substitution, especially in friendlies).

Unit tests pin their own prior values through dbt unit-test `overrides.vars`
so recalibrating production vars never breaks float-exact expectations.

## Shrinkage constants

`k` is the prior's weight in pseudo-observations. `k = 2` for minutes (two
role-typical games' worth of prior belief — history dominates from n=3);
`k = 4` for bench participation because a Bernoulli observation carries less
information than a minutes observation and early rates of 0/2 or 2/2 should
not saturate the probability.

## Alternatives considered

- **ML minutes model** (features: rotation risk, competition stage, opponent
  strength): highest ceiling, still deferred. Adoption trigger: it must beat
  this estimator by >=10% relative MAE on a season holdout, measured through
  `mon_football__expected_minutes_accuracy` (ticket K4).
- **Role history from historical lineups** (`stg_football__lineups`): would
  need an explicit fixture-completion filter to avoid phantom bench zeros
  from future confirmed lineups, and coverage depends on lineup ingestion
  depth. The stats-based `games_substitute` flag has neither problem.
- **Position-specific priors:** with zero history the starter/bench split
  explains far more variance than position (ADR 0002's argument still
  holds); revisit only if K4 monitoring shows systematic per-position bias.
