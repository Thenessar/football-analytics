# ADR 0005: Monte Carlo Fixture Simulation Design

- **Status:** Accepted
- **Date:** 2026-07-07
- **Relates to:** ml_upgrade_backlog.md §2.4/§2.5, Epic N; ADR 0004 (minutes decomposition)

## Goal

One simulated full game per fixture from the confirmed lineups: per-player
counts for all 11 target events, team totals, and a scoreline distribution —
coherent by construction, with honest uncertainty. "If we predict 10 shots on
target, which players take them?" is answered by conditioning player counts
on sampled team totals.

## Inputs

The simulator consumes the **governed active prediction set**
(`pred_football__player_event_predictions`), not the models (backlog §2.5).
Per-90 intensity is recovered as `predicted_mean / (expected_minutes / 90)`;
the ADR 0004 decomposition (`p_plays`, `expected_minutes_if_plays`) rides on
the same rows since K5. Team-total NB dispersions (`alpha_team`) come from
the registered model version tags written by M1. Lineage is
`prediction_set_id`; reruns are idempotent through a deterministic
`sim_set_id` = sha256(fixture, prediction_set_id, engine version + config
digest).

## Sampling design (per iteration, vectorized across all iterations)

1. **Minutes:** starters play `expected_minutes_if_plays` deterministically;
   bench players come on with probability `p_plays` for
   `expected_minutes_if_plays` minutes, else 0. Exposure `m/90` scales every
   intensity below.
2. **Team totals:** for volume events (`shots_total`, `passes_total`,
   `offsides`, `cards_yellow`) the team total is drawn from a negative
   binomial with mean `sum_p(rate90_p * m_p / 90)` and the M1 `alpha_team`
   dispersion, realized as a Poisson–Gamma mixture (`alpha = 0` degenerates
   to Poisson).
3. **Allocation:** totals are split across players by a multinomial with
   weights `rate90_p * m_p / 90`, implemented as sequential conditional
   binomials with suffix-sum weights (exact, vectorized, no dependency on
   batched-multinomial numpy support). Conditioning on the total guarantees
   adding-up and induces the realistic mild negative correlation between
   teammates. All-zero-weight edge: totals re-allocate uniformly across
   on-pitch players; with nobody on pitch the total is forced to 0.
4. **Shot chain:** per player `shots_on ~ Binomial(shots_total, rate90_on /
   rate90_total)` and `goals ~ Binomial(shots_on, rate90_goals / rate90_on)`
   (ratios clipped to [0, 1]), so `goals <= shots_on <= shots_total` holds in
   every draw.
5. **Fouls mirror:** fouls are one event stream per directed team pair:
   `F_AB ~ NB(mean = avg(sum_A committed-rate, sum_B drawn-rate))`, allocated
   twice — to team A players by committed intensity and to team B players by
   drawn intensity. `sum(fouls_committed_A) = sum(fouls_drawn_B)` exactly.
6. **Saves identity:** the starting goalkeeper's saves are
   `max(0, opponent shots_on total - opponent goals total)`; other players
   get structural zeros. The trained `goals_saves` model still produces the
   marginal prediction-table row; the simulation table's saves come from the
   identity so the game stays coherent.
7. **Assists:** team assists `~ Binomial(team goals, assist_per_goal_rate)`,
   allocated by assist intensity. Guarantees `assists <= goals` per team.
   The rate is calibrated to **0.688** (2026-07-07) from all completed
   fixtures — 68.8% of goals carry a recorded assist:

   ```sql
   SELECT sum(coalesce(goals_assists, 0)) / nullif(sum(coalesce(goals_total, 0)), 0)
   FROM football_analytics.silver.stg_football__player_match_stats
   -- -> 0.6880212282031842
   ```
8. **Cards:** yellows go through the NB-total + allocation path, clipped at
   2 per player; reds are per-player Bernoulli
   (`min(rate90_red * m/90, 0.5)`), at most 1.

## Approximations and v1 simplifications (with revisit triggers)

1. **No red-card feedback on minutes/rates.** A simulated red card does not
   truncate the player's minutes or reduce his team's subsequent output.
   Revisit if N6 shows card-heavy fixtures with systematically over-predicted
   volume events.
2. **No cross-event tempo factor.** Each event family draws its own team
   total; within-family correlation comes from the shared minutes and the
   structural chains only. Revisit if N6 team-total interval coverage is too
   narrow on 3+ targets.

   Note (2026-07-07): the first production fit found `alpha_team < alpha_player`
   for every target — the opposite of the intuition below that team totals are
   "usually more overdispersed". Summing many heterogeneous players toward a
   team total averages out player-specific dispersion, so team totals sit
   closer to Poisson. The code uses the measured `alpha_team` regardless, so
   this is an observation, not a defect; it does mean team-total intervals are
   tighter than a naive player-dispersion carry-over would predict.
3. **Starters' minutes are not resampled.** Only bench participation is
   random. Revisit alongside (1) — both need a within-match timeline model.
4. **Assist allocation does not exclude the goal scorer.** Assists are
   allocated multinomially across the team, so a scorer can be credited an
   assist on his own goal in rare draws; team-level totals and the
   `assists <= goals` invariant are unaffected. Revisit only if per-player
   joint goal+assist props become a product target.
5. **Saves identity ignores own goals and blocked shots.** API-Football
   counts blocked shots separately from shots on target, so
   on-target ≈ goals + saves holds; own goals are rare and excluded.
6. **Sum of player means = team mean.** No separate team-total model;
   revisit if N6 shows systematic team-total bias on 3+ targets (then add a
   reconciliation layer, backlog §2.4 alternative).

## Amendment (2026-07-11, engine 1.1.0): Elo goal anchor — FA-105

Simplification 6 produced compressed, near-identical team goal totals in
production because the player models under-discriminate (prediction quality
backlog E-6); the Elo layer's `expected_goals_for_pre` already prices
opponent strength per fixture and was unused. The engine now supports
**top-down proportional reconciliation of goal intensities**
(`apply_team_goal_anchor`): each team's player `goals_total` per-90
intensities are scaled by one multiplicative factor so the expected team
total equals `w * expected_goals_for_pre + (1 - w) * sum of player means`.
Player shares are preserved and the structural chains are untouched — only
the shots_on → goals conversion ratio scales, so `goals <= shots_on <=
shots_total` still holds in every draw, and assists/saves inherit the
anchored goals through their existing identities.

- `SimulationConfig.team_goal_anchor_weight` (in the config digest, so
  anchored sim sets supersede unanchored ones) defaults to **0.0 — the
  anchor is OFF until the FA-106 holdout backtest fits `w`** against
  team-total bias and interval coverage; record before/after numbers in the
  backlog when it lands.
- Anchors flow in per fixture: the notebook batch-reads
  `fct_football__team_elo_history.expected_goals_for_pre` (one query per
  run); the simulation backtest reads it off the feature-mart rows, which is
  what makes the `w` sweep cheap.
- Known bound: the per-player conversion ratio is clipped at 1, so extreme
  upward factors can undershoot the anchor; the backtest records realized
  bias. Extend the anchor to `shots_total` only if FA-106 shows totals bias
  there too.

**Outcome (2026-07-11): w = 0 adopted — anchor off.** The FA-106 holdout
w-sweep (1,018+ team-fixtures) showed monotone degradation of the goals
family as w rises (goals_total PIT deviation 0.007 → 0.039, saves top-1
allocation 0.976 → 0.810) and no improving metric: post-FA-101 the player
models receive their opponent-Elo features at serving time, so the sum of
player means already prices opponent strength and the Elo-only anchor
double-counts it. Simplification 6 stands. The machinery remains available
via `team_goal_anchor_weight` (digest-tracked); reopen only with fresh
backtest evidence of team-total bias, and add a simulated-vs-actual team
mean bias column to `score_simulation_backtest` when doing so. Real trigger
hit by the same run: `passes_total` player intervals are far too narrow
(0.581 coverage at 0.80 nominal, PIT deviation 0.189) — revisit
simplification 3/6 for passes with per-player dispersion (landed as the
engine 1.2.0 amendment below, FA-108; off pending the N6 re-run).

## Amendment (2026-07-11, engine 1.2.0): per-player allocation dispersion — FA-108

The FA-106 holdout run hit one real revisit trigger: `passes_total` player
80% interval coverage 0.581 (nominal 0.80), PIT max deviation 0.189. The
cause is structural: the multinomial allocation conditions on the team
total, so a player's variance is capped near `μ·(1−share)` plus a small
share²-scaled team-dispersion term — for high-count events (passes means
25–70) that is far below the real per-player variance `μ + α_player·μ²`
measured by M1. This is the flip side of the simplification-2 note
(`alpha_team < alpha_player` everywhere): team totals average out player
heterogeneity, so honoring `alpha_team` alone under-disperses individuals,
and the shortfall scales with μ² — which is why only passes tripped the
trigger.

The fix (`sample_dispersed_allocation_weights`): for targets listed in
`SimulationConfig.player_dispersion_targets`, each player's allocation
weight is multiplied by an i.i.d. mean-1 `Gamma(1/α_player, α_player)`
multiplier between the team-total draw and the multinomial split — a
generalized Dirichlet-multinomial allocation. Player marginals gain the NB
quadratic term (`≈ α_player·μ²` extra variance; exactly NB if the total
were Poisson at the perturbed sum), while expected shares, the structural
chains, and the NB(`alpha_team`) team-total draw are untouched — team
calibration, which FA-106 showed is healthy, is preserved by construction.
The alternative of drawing the total from the perturbed Poisson sum was
rejected: it would *imply* team dispersion (≈ α_player/n_eff) instead of
honoring the measured `alpha_team` contract from M1.

- Applies to the step-3 allocation of the four volume events
  (`shots_total`, `passes_total`, `offsides`, `cards_yellow`); the shot
  chain propagates upstream dispersion automatically via thinning.
- `player_dispersion_targets` participates in the config digest
  (order-insensitively) and `SIMULATION_ENGINE_VERSION` is 1.2.0, so
  dispersed sim sets supersede v1 sets cleanly. **Default: empty — OFF
  until the N6 re-run validates the real-data coverage** (evidence-gated;
  backlog FA-108). With the flag off, or without a fitted `α_player`, the
  engine is draw-for-draw identical to 1.1.0 (the helper consumes no
  randomness).
- `alpha_player` flows exactly like `alpha_team`: model-version tags →
  `load_dispersion_tags` (notebook 05) / `LoadedEventModel.alpha_player`
  (backtest) → `SimulationInputs.alpha_player`.
- Synthetic validation (tests, 440 players / 40 team-fixtures): in an
  NB(α=0.3) passes world the v1 allocation reproduces the symptom (0.405
  coverage, PIT deviation 0.232); with the layer on, coverage is 0.798 at
  0.80 nominal with PIT deviation 0.023, shots and team-total calibration
  unchanged. Finite-team normalization keeps realized player variance at
  ~0.92× the naive `μ + α·μ²` target (the (1−share)² factor) — immaterial
  at N6 tolerances.

## Output

`gold.sim_football__fixture_simulation` — grain `fixture_id / sim_set_id /
entity_type ('player'|'team'|'fixture') / entity_id / target_event`, with
`sim_mean`, `sim_std`, count percentiles (p05..p95), `p_ge_1..3`, lineage
(`prediction_set_id`, `engine_version`, `seed`, `n_sims`), and Option C
active-flag semantics (append + deterministic set id + older sets flipped
inactive). The fixture-level row carries the scoreline probability matrix as
JSON (`target_event = 'scoreline'`).
