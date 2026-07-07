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

## Output

`gold.sim_football__fixture_simulation` — grain `fixture_id / sim_set_id /
entity_type ('player'|'team'|'fixture') / entity_id / target_event`, with
`sim_mean`, `sim_std`, count percentiles (p05..p95), `p_ge_1..3`, lineage
(`prediction_set_id`, `engine_version`, `seed`, `n_sims`), and Option C
active-flag semantics (append + deterministic set id + older sets flipped
inactive). The fixture-level row carries the scoreline probability matrix as
JSON (`target_event = 'scoreline'`).
