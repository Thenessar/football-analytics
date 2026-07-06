# ADR 0002: Expected Minutes Estimator

- **Status:** Accepted
- **Date:** 2026-07-06
- **Relates to:** business_logic.md §12.2/§12.3; implementation_backlog.md tickets E1/E2

## Question

Which estimator should produce the pre-match `expected_minutes` value for
players in a confirmed lineup (§12.2 candidate list)?

## Decision

The recommended first estimator from §12.3: **trimmed mean over the last 5
played appearances** (`expected_minutes_l5_trimmed_or_median` family),
implemented in `fct_football__player_expected_minutes` as:

- **≥3 prior appearances:** mean of the last 5 played appearances after
  dropping the single minimum and single maximum. Robust to one red-card exit
  or one garbage-time cameo without needing a warehouse median-over-window
  function; with ≤5 observations, trimming min/max is equivalent in spirit to
  a median and fully deterministic.
- **1–2 prior appearances:** plain average (nothing to trim).
- **0 prior appearances:** role prior — 85 minutes for starters, 15 for
  substitutes. Chosen over a position-specific prior because with no history
  the starter/sub split explains far more variance than position does, and it
  keeps the fallback free of extra joins.
- Output clamped to [0, 120]; the branch taken is recorded in
  `expected_minutes_method` so downstream consumers and monitoring can
  segment by estimator quality.

A "played appearance" is a fixture with provider minutes in 1..130, the same
rule `fct_football__player_match_features` uses for modeling rows.

## Alternatives left as future work (from §12.2)

- **Weighted recency decay** (exponential weights over last N appearances) —
  strictly more expressive; try once Epic F evaluation can measure whether it
  beats the trimmed mean on held-out seasons.
- **Separate expected-minutes model** (features: rotation risk, competition
  stage, scoreline expectations) — highest ceiling, needs training
  infrastructure from Epic F first.
- **Starter/sub priors blended with history** (shrinkage toward role prior for
  low sample counts) — natural v2; the `expected_minutes_method` column
  already exposes the sample-size segments needed to evaluate it.
