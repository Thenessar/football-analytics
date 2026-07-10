# Prediction Quality Backlog — Serving Skew, Player Discrimination, Fixture Differentiation

Successor to [`ml_upgrade_backlog.md`](ml_upgrade_backlog.md) (Epics K–O, closed 2026-07-07). That backlog built the evaluation and simulation machinery; this one fixes why production predictions are flat: every player collapses toward a team/position average and every simulated game looks the same. Source of truth for business rules remains [`business_logic.md`](business_logic.md).

**Audience: AI agents.** Working Agreement from `ml_upgrade_backlog.md` §0 applies verbatim (one commit per ticket pushed to `main`, pytest + dbt parse green, pure logic in `football_analytics/`, MLflow metric quota discipline, leakage rules non-negotiable). Jira epic: **FA-100**; each ticket below carries its Jira id, its **agent persona** (adopt it — it encodes the experience the ticket needs), and its dependencies. Commit message format: `<imperative summary> (FA-1xx)`.

---

## 1. Evidence (measured 2026-07-10, France vs Morocco fixture 1578539, model `player_event__shots_total` v7)

Diagnosis ran against production Delta tables (time-travel to the prediction timestamp 2026-07-09 19:28 UTC) and the registered model probed locally with counterfactual sweeps.

**E-1 — The feature data at serving time was correct.** Mbappé's inference row carried `shots_total_l5_p90 = 4.78` (matches his actual 4.8 avg), fresh through the prior matchday. Rolling-history staleness was NOT the cause of this fixture's predictions (the ff47eb6 checkpoint-lookback fix addresses a real but separate risk).

**E-2 — The player-Elo feature family is degenerate on every inference row.** `fct_football__player_elo_history` is built from `fct_football__player_match_features` (played appearances only), so future fixtures have no rows. The feature mart coalesces:
- `player_offensive_modifier_pre` → **0.0 for every player** (training median F ≈ 0.105, Mbappé trains at avg **1.363**, clip is 1.5)
- `player_offensive_elo_pre` / `player_offensive_rating_pre` → `team_elo_attack_pre` → **identical for all teammates** (observed: 1.0 for all 26 France players)
- all `player_attack_delta_vs_team` / `player_attack_vs_opp_defense` interactions therefore also collapse.

The team-side asymmetry: `build_team_elo_history` (`elo.py:204`) iterates ALL fixtures and emits pre snapshots for scheduled ones (NaN-goals branch skips the update); `build_player_elo_history` (`elo.py:429`) cannot, because its input contains only played rows.

**E-3 — The model's #1 feature is the one that's degenerate.** Gain share in v7 `shots_total`: `player_offensive_modifier_pre` **24.3%**, `formation_row` 14.2%, `shots_total_l5_p90` 10.9%, `goals_saves_l5_p90` 8.1% (a goalkeeper proxy — position is not a feature), `is_starting` 4.9%. Counterfactual sweeps on Mbappé's served row:
- modifier 0.0 → 1.0 lifts rate/90 from **1.134 → 1.724** (+52%)
- `shots_total_l5_p90` 0 → 6 moves rate/90 only **1.066 → 1.134** — with modifier stuck at 0, the trees route down "fringe player" paths where the l5 splits barely apply.

So at serving, the model literally cannot tell Mbappé from a median forward. Predicted 1.086 shots (86 min) vs his 4.8/game reality; France team total ≈ 7.3 shots vs ~15–16 actual.

**E-4 — Even in-distribution, dynamic range is compressed.** Best case (modifier 1.5 + l5 4.78): rate/90 = 1.74 — the model never predicts above ~1.8 shots/90 for anyone while true rates reach ~5. Validation metrics agree: `rank_spearman` 0.44, `top1_hit_rate` 0.42, `skill_vs_player_l5` **0.086** (barely beats the naive own-L5 baseline), yet `ece_ge_1` 0.0085 — calibrated but undiscriminating. Causes: only 5-game windows (noisy → correctly shrunk), player modifier clipped at ±1.5 with lr 0.05 (saturates for elite players), no position features, untuned `min_child_samples=80`.

**E-5 — Validation metrics overstate production skill.** The L2 backtest and N6 sim backtest score completed-fixture rows, which carry fixture-exact player-Elo values that serving never has. Production runs with the E-2 degenerate features, so every historical adoption decision was measured on rows the serving path cannot produce.

**E-6 — Simulation flatness is downstream.** Team totals = Σ player intensities (ADR 0005 §2.4), so compressed player rates ⇒ compressed team totals. Active sims: Colombia 1.10 / Switzerland 1.02, France 1.28 / Morocco 0.90 goals, all with identical [0, 3] 90% intervals. `expected_goals_for_pre` (Elo layer) is computed but unused as a totals anchor.

---

## 2. Ticket map, priority order, and expected lift

| Jira | Ticket | Persona | Depends on | Status |
| --- | --- | --- | --- | --- |
| FA-101 | P1 — Current player-Elo state for future fixtures | Ledger | — | ✅ DONE (2026-07-10) |
| FA-103 | P2b — Inference feature-health monitor | Sentinel | — (parallel to FA-101) | ✅ DONE (2026-07-10) |
| FA-102 | P2a — Serving-parity backtest mode | Mirror | FA-101 | ✅ DONE (2026-07-10) |
| FA-104 | P3 — Player-identity features + position + tuning | Shrink | FA-101, FA-102 | ⬜ TODO |
| FA-106 | P4-pre — Run N6 simulation holdout backtest (F-3) | Meter | FA-101 deployed | ⬜ TODO |
| FA-105 | P4 — Anchor sim team totals to Elo goal model | Croupier | FA-106 | ⬜ TODO |
| FA-107 | P5 — Follow-up reruns, retrain + re-register | Registrar | FA-101, FA-102, FA-104 | ⬜ TODO |

**What to do next:** pick the highest ticket in this table whose dependencies are all ✅ DONE. Start with **FA-101 and FA-103 in parallel** (no dependencies). FA-101 requires **no retraining** — the models were trained on good features; serving just has to supply them (the +52% sweep in E-3 is recovered immediately). FA-104 raises the discrimination ceiling; FA-105 differentiates fixtures. FA-102/FA-103 make sure this class of bug can never ship silently again. When a ticket lands, flip its Status here to ✅ DONE (date) in the same commit, per Working Agreement.

Post-FA-101 expectation to keep honest: Mbappé lands ~1.7–1.9 shots/90, not 4.8. Predicting the full spread is FA-104's job, and even then a correct conditional mean sits *between* the L5 average and the population rate — the L5 window is a noisy estimator and some shrinkage is statistically right.

---

## Epic P — Prediction Quality

### P1 / FA-101. Emit current player-Elo state for future fixtures (fixes E-2, the production bug)
- **Jira:** FA-101 · Story · Priority Highest · 5 pts · Labels `dbt`, `feature-pipeline`, `no-retrain` · Blocks FA-102, FA-104, FA-106.
- **Agent persona — "Ledger":** senior feature-pipeline engineer, 8 years building point-in-time-correct feature stores for betting/trading systems. Deep experience with chronological state machines in pandas, dbt Python models on Databricks, and Delta time travel. Paranoid about leakage by profession: treats every feature as guilty until proven strictly-prior; has been burned by training/serving skew before and writes the parity test before the fix.
- **Files:** `football_analytics/elo.py` (`build_player_elo_history`), `dbt/models/marts/fct_football__player_elo_history.py`, `dbt/models/marts/fct_football__player_event_features.sql`, `tests/test_elo.py` (or the existing elo test module), dbt unit tests.
- **Required behavior:**
  - `build_player_elo_history` additionally emits one **current-state row per player** after the chronological pass: final `offensive_modifier`/`defensive_modifier`/`missed_fixture_count` (post last played fixture). Mark rows with a discriminator column (e.g. `fixture_id` null + `is_current_state = true`). Lineups are unknown at medallion build time (T-60), so current-state rows must be per-player, not per-future-fixture.
  - Feature mart: inference rows join current-state by `player_id` and recompute the family exactly as training defines it: `player_offensive_elo_pre = team_elo.team_elo_attack_pre + current_modifier` (team Elo future-fixture rows already exist), rating likewise; `missed_fixture_count_pre` from current state. Completed-fixture rows keep the fixture-exact join unchanged.
  - Leakage-safe by construction: current state = state after all *played* fixtures, which is exactly the pre-match state of any future fixture.
- **Acceptance criteria:**
  - dbt unit test: inference rows for a mocked future fixture carry distinct per-player `player_offensive_modifier_pre` matching each player's post-history state; completed-row values unchanged (bit-identical for the training path).
  - New dbt data test (also the P2 monitor's assertion): for any non-completed fixture/team with ≥ 11 lineup rows, `count(distinct player_offensive_modifier_pre) > 1`.
  - After deploy, rerun inference on one upcoming fixture and record before/after predicted_mean spread for the top-l5 player here.
  - **No model retrain required or performed in this ticket** (the registered v7 boosters already consume these features).
- **Implementation notes (2026-07-10):**
  - `build_player_elo_history` appends one current-state row per `(team, player)` after the chronological pass: `is_current_state = true`, `fixture_id` null, final modifiers + `missed_fixture_count_pre`, and `fixture_date_utc` = the team's last processed fixture date (state-as-of timestamp — usable by the FA-103 staleness check). Appearance snapshots carry `is_current_state = false`.
  - `PLAYER_ELO_SCHEMA` gained nullable `fixture_id` and non-null `is_current_state`; the dbt Python model now converts NaN→None and nullable-int floats→int before `createDataFrame` (pandas promotes int columns with nulls to float64, which Spark's `IntegerType` rejects).
  - Feature mart: `player_elo` CTE is restricted to `fixture_id is not null` (completed rows bit-identical — fixture-exact values stay first in every coalesce); new `player_elo_current` CTE joins by `(team_id, player_id)` — not `player_id` alone, matching the builder's state keying and avoiding fan-out for players with appearances for more than one association — gated to `not base.is_completed_fixture` so a completed row can never leak its own fixture's update. Elo/rating recomputed as `team_elo_attack_pre + current modifier`, exactly the training-time snapshot formula.
  - `assemble_hierarchical_feature_frame` drops `is_current_state` rows before merging (training path stays fixture-exact).
  - Tests: `test_player_elo_history_emits_current_state_rows_after_history_pass` (decay + missed-count carried into current state), `test_assemble_hierarchical_feature_frame_ignores_current_state_rows`, dbt unit test `player_event_features_inference_rows_use_current_player_elo_state` (distinct per-player modifiers on a mocked future fixture, completed rows unchanged; binary-exact mock values), and data test `assert_fct_football__player_event_features_inference_elo_differentiated` (the standing ≥11-lineup distinct-modifier assertion FA-103 builds on).
  - **Pending deploy:** the before/after predicted-mean spread on a live fixture cannot be recorded until the marts are rebuilt on Databricks and inference reruns (prod jobs paused in 7fcb212; SQL warehouse quota limited). Record it alongside the FA-106 run, which requires this deploy anyway.
  - Follow-up (out of scope here): `fct_football__lineup_elo_strength` still joins fixture-exact rows only, so `team_lineup_attack_strength` for future fixtures falls back to team Elo instead of a lineup-weighted average; it could join `player_elo_current` the same way. Candidate for FA-104, which owns the feature-set revision.

### P2a / FA-102. Serving-parity backtest mode (fixes E-5)
- **Jira:** FA-102 · Story · Priority High · 5 pts · Labels `evaluation`, `backtest` · Depends on FA-101.
- **Agent persona — "Mirror":** ML evaluation engineer, ex-model-risk-management at a quant fund. Specialty: making offline metrics predict online behavior — has audited dozens of models whose validation looked great while serving inputs were silently broken. Fluent in scikit-learn-free numpy scoring code, chronological cross-validation, and MLflow artifact discipline. Motto: "score the rows the server sees, not the rows the warehouse has."
- **Files:** `football_analytics/evaluation.py` or `ml_training.py` (backtest row degradation helper), `scripts/train_poisson_lgbm.py` (`--serving-parity` flag).
- **Required behavior:** backtest mode that transforms validation rows into *serving-shaped* rows before scoring — apply the same joins/coalesces the mart applies to non-completed fixtures (post-FA-101 this means: current-state Elo instead of fixture-exact, everything else identical). Report the paired delta (`training-shaped` vs `serving-shaped`) per target for the headline metrics (RPS, rank_spearman, top1, skill scores) as one artifact. Pre-FA-101 this delta is the measured cost of the bug; post-FA-101 it must be ≈ 0 and stays as a regression gate. MLflow quota discipline per Working Agreement #4 (scalars = cross-fold means only; tables to artifacts).
- **Acceptance criteria:** unit test proves the degradation helper reproduces the mart's inference-row coalesce semantics on a fixture with known values; parity delta ≈ 0 post-FA-101 asserted on synthetic data and documented as the standing regression gate; backtest artifact includes the paired-delta table.
- **Implementation notes (2026-07-10):**
  - `degrade_rows_to_serving_shape` (ml_training.py): modifiers/`missed_fixture_count_pre` pass through with the mart's 0-coalesce; `player_*_elo_pre`/`player_*_rating_pre` rebuilt as team baseline + modifier (the serving formula); interaction features recomputed from the rebuilt bases via `add_model_interaction_features`; labels/exposure/rolling windows untouched. Post-FA-101 the transform is an identity on self-consistent mart rows.
  - `run_rolling_origin_backtest(serving_parity=True)`: each fold's booster additionally scores serving-shaped validation rows; result gains `parity_report` (per season/target/metric: training_shaped, serving_shaped, delta) and `parity_summary` (cross-fold means + fold counts). Full metric superset is paired, which covers the headline set (RPS, rank_spearman, top1, skill scores).
  - `scripts/train_poisson_lgbm.py --backtest --serving-parity`: parity tables land as `serving_parity_report.csv/json` + `serving_parity_summary.csv` in the same backtest artifact folder; no new MLflow metrics (quota discipline — only a `serving_parity` param is logged).
  - Tests: helper-semantics test on known values incl. the no-Elo-history coalesce; **standing regression gate** `test_serving_parity_backtest_reports_zero_delta_on_consistent_features` (delta < 1e-9 on self-consistent synthetic data); an E-5 replica proving a nonzero RPS delta when the model leans on a feature serving cannot rebuild.
  - Scope note: the transform covers the player-Elo family exactly as the ticket pre-committed ("everything else identical"). The known residual serving skew in `team_lineup_attack/defense_strength` (see FA-101 notes) is deliberately NOT degraded here so the delta gate stays ≈ 0; when the lineup-strength follow-up lands in FA-104, extend the helper and keep the gate.
  - The production parity run (real-data delta table on the Databricks feature table) rides with the FA-104 backtest / FA-107 verification, same quota batch.

### P2b / FA-103. `mon_football__inference_feature_health` monitor (guards the class)
- **Jira:** FA-103 · Story · Priority High · 3 pts · Labels `dbt`, `monitoring` · Depends on nothing (start in parallel with FA-101; assertions get stricter after it lands).
- **Agent persona — "Sentinel":** analytics engineer specialized in data observability — built dbt-test suites and freshness monitors for feature stores serving live models. Thinks in grains and null-rates; writes views that are empty-but-queryable before data lands and cheap enough for a free-tier warehouse.
- **Files:** new `dbt/models/marts/mon_football__inference_feature_health.sql` (+ schema.yml), `docs/business_logic.md` §16.3.
- **Required behavior:** view over non-completed feature-mart rows: per fixture/team — null rates and within-team distinct-count for the player-differentiating feature families (`*_modifier_pre`, `*_elo_pre`, `*_rating_pre`, `*_l5_p90`, `appearances_l5_count`), plus max `fct_football__player_match_features.fixture_date_utc` vs `current_timestamp()` (history staleness). Empty-but-queryable convention.
- **Acceptance criteria:** monitor flags a synthetic all-constant team in a dbt unit test; empty-but-queryable with zero upcoming fixtures; wired into the §16.3 monitoring list with schema.yml docs.
- **Implementation notes (2026-07-10):**
  - Long-format view: one row per upcoming fixture/team/monitored feature (18 features: 6 player-Elo family, `appearances_l5_count`, 11 `*_l5_p90`), built as a Jinja `union all` over a CTE pre-filtered to `not is_completed_fixture` — 18 passes over a near-empty slice, cheap on the free-tier warehouse.
  - Per row: `lineup_row_count`, `null_count`, `null_rate`, `distinct_value_count` (count-distinct ignores nulls, so all-null ⇒ 0 and is also degenerate), and `is_flagged_constant` = `lineup_row_count >= 11 and distinct_value_count <= 1`. Sparse events can flag legitimately on quiet teams — the view reports; the hard gate stays the FA-101 data test on modifiers.
  - Staleness: `team_history_max_fixture_date_utc` (per team) + global `history_max_fixture_date_utc` and `history_staleness_hours` vs `current_timestamp()`; aggregates-without-group-by keep the view empty-but-queryable (zero rows only when there are no upcoming fixtures, never an error).
  - Unit test `inference_feature_health_flags_all_constant_team`: 11-player future lineup with constant modifier (flagged), 11-distinct elo (not flagged), all-null features (flagged, null_rate 1.0), completed row excluded.
  - Wired into `business_logic.md` §16.3 and schema.yml with per-column docs.

### P3 / FA-104. Player-identity rate features with shrinkage + position features + tuning (fixes E-4)
- **Jira:** FA-104 · Story · Priority High · 8 pts · Labels `feature-engineering`, `lightgbm`, `tuning` · Depends on FA-101 (baseline) and FA-102 (honest measurement).
- **Agent persona — "Shrink":** sports-analytics ML scientist, PhD in applied statistics, shipped player-prop pricing models at a sportsbook. Expert in count regression (Poisson/NB), empirical-Bayes shrinkage, and LightGBM internals (monotone constraints, exposure offsets, Optuna). Knows a 5-game window is noise and that the fix is a longer, shrunken identity signal — not trusting the window harder.
- **Files:** `dbt/models/marts/fct_football__player_event_features.sql` (+ schema.yml), `football_analytics/ml_training.py` (`DEFAULT_LIGHTGBM_FEATURES`), `scripts/tune_lgbm.py` run (F-4), `docs/adr/` note if the modifier clip changes.
- **Required behavior:**
  - Per target: empirical-Bayes career per-90 rate per player — `(sum(event) + k · posgroup_rate · sum(exposure)) / (sum(exposure) + k)` over **all** strictly-prior appearances with exponential decay (halflife ≈ 10 appearances), exposed as `<event>_eb_p90`. This is the low-noise player-identity signal the model currently lacks; the L5 window stays as the form signal.
  - Add `position_group` one-hots (or ordinal) + `is_goalkeeper` to `DEFAULT_LIGHTGBM_FEATURES`; expect the `goals_saves_l5_p90`-as-GK-proxy gain (E-3) to migrate.
  - Evaluate raising `DEFAULT_PLAYER_MODIFIER_CLIP` (1.5 saturates for elite attackers — E-4) and/or the 0.05 learning rate via the L2 backtest before adopting.
  - Run M3 tuning (F-4) on the new feature set; commit adopted `config/lgbm_params/*.json` with the comparison table.
  - Optional, evidence-gated: LightGBM `monotone_constraints` (+1) on own-event `_eb_p90`/`_l5_p90` features.
- **Acceptance criteria:** rolling-origin backtest (L2) shows, vs the P1-fixed baseline: `rank_spearman` and `top1_hit_rate` up, `skill_vs_player_l5` materially above 0.086, RPS non-worse, ECE within 0.02. Record the table here. Leakage rule: decay windows strictly prior to the row's fixture (mirror the `prior_appearances` ranked-join pattern).

### P4-pre / FA-106. Run the N6 simulation holdout backtest (F-3, evidence for FA-105)
- **Jira:** FA-106 · Task · Priority High (unblocks FA-105) · 2 pts · Labels `databricks`, `backtest`, `data-gated` · Depends on FA-101 being deployed (otherwise it measures the broken serving path).
- **Agent persona — "Meter":** MLOps operator comfortable driving Databricks jobs on a free-tier budget — batches SQL, watches the daily warehouse quota, and records numbers verbatim into docs. No modeling opinions; produces evidence.
- **Entry point:** `scripts/backtest_simulation.py --season <latest completed>` (never run; quota exhausted 2026-07-07 — see ml_upgrade_backlog F-3). Measures interval coverage, PIT uniformity, team-total bias, and allocation top-1/top-3 — the pre-committed evidence for FA-105's adoption and ADR 0005's revisit triggers.
- **Acceptance criteria:** artifact (JSON + CSV) produced; per-target coverage/PIT/team-total-bias/allocation numbers recorded here and in the ml_upgrade_backlog N6 notes; targets with 80% coverage outside [70, 90] flagged against the ADR 0005 revisit triggers.

### P4 / FA-105. Anchor simulation team totals to the Elo goal model (fixes E-6)
- **Jira:** FA-105 · Story · Priority Medium-High · 5 pts · Labels `simulation`, `monte-carlo` · Depends on FA-106 (evidence).
- **Agent persona — "Croupier":** simulation engineer with a quant-sports background — builds vectorized Monte Carlo engines (numpy Generators, NB via Poisson–Gamma) and has implemented forecast reconciliation (top-down proportional scaling) in hierarchical demand systems. Obsessive about invariants tests and bit-identical seeds.
- **Files:** `football_analytics/simulation.py`, `notebooks/05_fixture_simulation.py`, `tests/test_simulation.py`, ADR 0005 update.
- **Required behavior:** reconcile per-fixture team goal intensities: scale each team's player `goals_total` intensities multiplicatively so their sum matches a blend `w · expected_goals_for_pre + (1 − w) · Σ player means` (w fitted on the FA-106 backtest; shares preserved, structural chains untouched). Extend to `shots_total` only if FA-106 shows totals bias there too. This was the anticipated "team-total model + reconciliation" alternative in ml_upgrade_backlog §2.4 — the Elo layer already prices opponent strength per fixture (E-6) and is free.
- **Acceptance criteria:** invariants tests still pass; determinism under seed preserved; simulated team-goal means across a set of mismatched fixtures track `expected_goals_for_pre` ordering; scoreline matrices differ visibly between a favorite-vs-minnow and an even fixture; FA-106 coverage/bias numbers recorded before/after.

### P5 / FA-107. Re-run the data-gated follow-ups on the fixed stack, retrain + re-register
- **Jira:** FA-107 · Story · Priority Medium · 5 pts · Labels `mlops`, `release` · Depends on FA-101, FA-102 (parity gate green), FA-104 (feature set final).
- **Agent persona — "Registrar":** ML release engineer — owns model registries, version tags, and rollout verification. Treats every registration as an audited artifact: lineage params, dispersion tags, alias hygiene. Never ships without a monitored first live run and knows the MLflow free-tier metric quota by heart.
- **Tasks:**
  1. F-1 (M4 sparse-target decision): `scripts/train_poisson_lgbm.py --backtest --targets cards_yellow,cards_red,goals_total,goals_assists`, apply the pre-committed ECE rule verbatim, update ADR 0003.
  2. F-5 (SHAP review) re-run on the FA-104 feature set.
  3. Retrain + re-register all 11 targets (`scripts/train_poisson_lgbm.py --registered-model-prefix football_analytics.gold.player_event`); confirm α tags + pyfunc artifacts on the new versions.
  4. Verify the first live fixture with `mon_football__prediction_vs_actual`, `mon_football__inference_feature_health` (FA-103), and the FA-102 parity delta.
- **Acceptance criteria:** per-item numbers recorded in this file; the ml_upgrade_backlog Follow-Ups table updated to point here; new versions serving with a demonstrably player-differentiated spread on a live fixture (top-l5 player ≫ team median, recorded).

---

## Dependency graph (Jira ids)

```text
FA-101 (P1, no deps) ──→ FA-102 (P2a, parity delta ≈ 0 gate) ──→ FA-104 (P3) ──→ FA-107 (P5)
FA-101 ────────────────→ FA-104
FA-101 deployed ───────→ FA-106 (P4-pre, F-3 run) ──→ FA-105 (P4)
FA-103 (P2b) — no deps, start in parallel with FA-101
FA-105 and FA-102/FA-104 are independent branches
```

**Sprint order:** FA-101 + FA-103 now → FA-102 → FA-104 ∥ (FA-106 → FA-105) → FA-107.
