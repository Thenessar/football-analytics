# ADR 0001: Expected Possession Approach

- **Status:** Accepted
- **Date:** 2026-07-06
- **Relates to:** business_logic.md §11.5 open question; implementation_backlog.md ticket C2

## Question

Should expected possession be (a) an explicit intermediate model, (b) a feature
engineered from Elo and historical team statistics, or (c) an implicit signal
learned by the final event models?

## Decision

**Option (b): engineered features**, delivered in
`fct_football__team_match_stats_context`:

- `expected_possession_share` — the team's rolling L5 possession average
  normalized against the opponent's rolling L5 possession average
  (`team_l5 / (team_l5 + opponent_l5)`), falling back to `0.5` when either
  side lacks history. Pre-match safe: both inputs use only fixtures strictly
  prior to the current one.
- `elo_expected_score` — classic Elo win expectancy
  `1 / (1 + 10^((opponent_elo_pre - team_elo_pre) / 400))` from
  `fct_football__team_elo_history` `_pre` columns (default 1500 when missing).
- `elo_possession_interaction` — `elo_expected_score * expected_possession_share`,
  the multiplicative interaction called out in §11.5 letting downstream models
  separate "strong team that hoards the ball" from "strong team that plays direct".
- `formation_possession_profile` — the team's average possession over its last
  10 fixtures *with the same formation* (nullable when the formation is new or
  unconfirmed; downstream models should treat null as "no profile", not zero).

## Why not the alternatives

- **(a) Explicit intermediate model:** highest ceiling but adds a trained
  artifact to version, monitor, and re-fit before any event model exists to
  consume it. Nothing downstream requires a point prediction of possession —
  only informative pre-match signals. Revisit once the Epic F training loop is
  live and feature importances show possession features carrying weight.
- **(c) Fully implicit:** free, but the final event models would have to learn
  the possession structure from raw rolling columns; the explicit share and
  interaction terms are cheap and encode the known relationship directly.

## Consequences

- No new model in the DAG; `fct_football__team_match_stats_context` gained an
  Elo dependency, which moves it into the `dbt_build_python_dependents` build
  phase (it now sits downstream of the python Elo models).
- If (a) is picked up later, it should slot in as a new mart feeding
  `expected_possession_share` and leave the column contract unchanged.
