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
| FA-104 | P3 — Player-identity features + position + tuning | Shrink | FA-101, FA-102 | ✅ DONE (2026-07-11) |
| FA-106 | P4-pre — Run N6 simulation holdout backtest (F-3) | Meter | FA-101 deployed | ✅ DONE (2026-07-11) |
| FA-105 | P4 — Anchor sim team totals to Elo goal model | Croupier | FA-106 | ✅ DONE (2026-07-11) — evidence says **keep w = 0**; anchor machinery retained |
| FA-107 | P5 — Follow-up reruns, retrain + re-register | Registrar | FA-101, FA-102, FA-104 | 🟨 3 of 4 done (2026-07-12) — live-fixture verification pending |
| FA-108 | P6 — Per-player NB dispersion in sim allocation (passes coverage fix) | Croupier | FA-106 (evidence) | ✅ ADOPTED (2026-07-12) on three-arm dominance — PIT gate waived with cause, calibration trigger stays open (per-position α_player follow-up) |
| FA-109 | P7 — Per-position-group α_player (closes the open passes PIT trigger) | Croupier | FA-108 (adopted layer) | 🟨 CODE LANDED (2026-07-12) — gated off; needs retrain (per-group tags) + N6 revalidation |

**What to do next:** pick the highest ticket in this table whose dependencies are all ✅ DONE. Start with **FA-101 and FA-103 in parallel** (no dependencies). FA-101 requires **no retraining** — the models were trained on good features; serving just has to supply them (the +52% sweep in E-3 is recovered immediately). FA-104 raises the discrimination ceiling; FA-105 differentiates fixtures. FA-102/FA-103 make sure this class of bug can never ship silently again. When a ticket lands, flip its Status here to ✅ DONE (date) in the same commit, per Working Agreement.

**State as of 2026-07-11:** all code-side work through FA-105 is on `main` and green (pytest + dbt parse; dbt unit tests run in the next `dbt build`). Everything left funnels through **one human gate: the prod bundle deploy** (`databricks bundle deploy -t prod --var="sql_warehouse_id=eec07ef59d758b5d"` + `databricks bundle run international_medallion_pipeline -t prod --var=...`) — an agent in auto mode cannot authorize it. After that single deploy (one quota day, batched): FA-101's before/after spread check, FA-102's real-data parity table, FA-104's backtest + tuning evidence (notes step 2–4), FA-106's holdout run + w-sweep, FA-105's w adoption, then FA-107's retrain/re-register/live verification.

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
  - **Production parity result (2026-07-11):** ran with the FA-104 evidence backtests on the rebuilt marts — paired training-vs-serving deltas are **exactly 0.0 for every target × metric × fold**, on both the P1-fixed baseline and the FA-104 feature set (3 folds each). The serving path reproduces the training features bit-for-bit post-FA-101; E-5 is closed and this stands as the regression gate for every future backtest.

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
- **Implementation notes (2026-07-11) — code landed, evidence runs pending:**
  - Mart: `<event>_eb_p90` per target = `(Σ w·y + k·posgroup_rate) / (Σ w·exposure + k)` with `w = 0.5^(appearance_age/10)` over **all** strictly-prior played appearances (reuses the leakage-safe `prior_appearances` ranked join), `k = 5` ninety-minute pseudo-appearances. The ticket's formula sketch had a spurious `· sum(exposure)` in the prior term; the implemented form is the standard Gamma-Poisson posterior mean (prior weight k, prior mean = posgroup rate) — with no history it returns the posgroup rate, with a long career it converges to the decayed career rate.
  - The shrinkage prior is the **strictly-prior** position-group per-90 rate: daily cumulative posgroup sums (`posgroup_running`) looked up as-of the row's fixture date with `max_by` (same-date fixtures excluded). No static population constants, so the L2 backtest folds stay leakage-clean.
  - EB features derive identically for completed and inference rows (both purely from strictly-prior played appearances) — serving-skew-free by construction, no FA-102 helper extension needed.
  - `position_group_g/d/m/f` one-hots derive in `add_model_interaction_features` (applied by both `build_training_feature_frame` and `build_prediction_records` — training/serving parity); unknown groups get all-zero indicators. `is_goalkeeper` + one-hots + 11 `*_eb_p90` added to `DEFAULT_LIGHTGBM_FEATURES` (`_select_available_features` drops them harmlessly on pre-FA-104 tables).
  - Tests: dbt unit test `player_event_features_eb_rates_shrink_and_exclude_current_fixture` (binary-exact shrinkage math + both leakage guards: own-fixture sums and same-date posgroup prior), python tests for one-hots and the feature list.
  - **Pending (blocked: prod deploy requires user approval — auto-mode denied `databricks bundle deploy -t prod`). Runbook:**
    1. `databricks bundle deploy -t prod --var="sql_warehouse_id=eec07ef59d758b5d"` then `databricks bundle run international_medallion_pipeline -t prod --var="sql_warehouse_id=eec07ef59d758b5d"` (rebuilds marts with FA-101+FA-104, runs all dbt unit/data tests — this is also FA-101's deploy).
    2. On Databricks, P1-fixed baseline: `python scripts/train_poisson_lgbm.py --backtest --serving-parity --features <pre-FA-104 list>`; new feature set: same command without `--features`. Record the acceptance table here (rank_spearman/top1 up, skill_vs_player_l5 ≫ 0.086, RPS non-worse, ECE within 0.02) plus the FA-102 parity delta (must be ≈ 0).
    3. `scripts/tune_lgbm.py` (F-4, needs the `tuning` extra) on the new feature set; commit adopted `config/lgbm_params/*.json` with the comparison table.
    4. Modifier-clip/lr evaluation (E-4): needs feature-table rebuilds with `DEFAULT_PLAYER_MODIFIER_CLIP` 2.0/2.5 variants scored via step 2 — budget a separate quota day; adopt only on backtest evidence + ADR note.
  - Follow-up carried from FA-101: `fct_football__lineup_elo_strength` current-state join (lineup strength for future fixtures still falls back to team Elo); extend `degrade_rows_to_serving_shape` in the same change and keep the parity gate.
- **Backtest evidence (2026-07-11, L2 rolling-origin, 3 folds, serving-parity mode on both runs; baseline = P1-fixed stack with the pre-FA-104 feature list):**

  | target | rank_spearman | top1_hit_rate | skill_vs_player_l5 | rps | ece_ge_1 |
  | --- | --- | --- | --- | --- | --- |
  | shots_total | 0.4532 → **0.4696** | 0.4343 → **0.4384** | 0.0815 → **0.0922** | 0.3240 → **0.3206** | 0.0146 → 0.0139 |
  | shots_on | 0.3790 → **0.3887** | 0.4347 → **0.4373** | 0.1008 → **0.1094** | 0.1817 → **0.1808** | 0.0119 → 0.0103 |
  | goals_total | 0.2936 → **0.3040** | 0.3765 → **0.3832** | 0.1198 → **0.1293** | 0.0696 → **0.0693** | 0.0056 → 0.0066 |
  | goals_assists | 0.2228 → **0.2235** | 0.2509 → 0.2459 | 0.1246 → **0.1263** | 0.0527 → **0.0527** | 0.0040 → 0.0043 |
  | goals_saves | 0.3908 → **0.4003** | 0.7552 → **0.7581** | 0.1203 → **0.1205** | 0.9732 → 0.9732 | 0.0251 → 0.0261 |
  | passes_total | 0.8806 → **0.8849** | 0.4114 → **0.4170** | 0.1347 → **0.1462** | 6.1631 → **6.0538** | 0.0089 → 0.0080 |
  | offsides | 0.3148 → **0.3233** | 0.4180 → **0.4273** | 0.0813 → **0.0916** | 0.0843 → **0.0838** | 0.0080 → 0.0082 |
  | fouls_committed | 0.4070 → **0.4166** | 0.2602 → **0.2838** | 0.0670 → **0.0727** | 0.4353 → **0.4324** | 0.0281 → 0.0278 |
  | fouls_drawn | 0.4570 → **0.4698** | 0.3682 → **0.3977** | 0.0532 → **0.0614** | 0.4220 → **0.4166** | 0.0261 → 0.0263 |
  | cards_yellow | 0.1370 → **0.1435** | 0.1869 → **0.2026** | 0.0843 → **0.0859** | 0.0944 → **0.0943** | 0.0090 → 0.0097 |
  | cards_red | −0.0285 → −0.0232 | 0.0695 → 0.0590 | 0.1215 → 0.1198 | 0.004251 → 0.004258 | 0.0002 → 0.0005 |

  **Verdict: gates pass — the FA-104 feature set is adopted** (it is the default list). rank_spearman up 11/11; top1 up 9/11 (down only on cards_red/goals_assists, inside fold std); skill_vs_player_l5 up 10/11 with shots_total 0.0922 > the 0.086 reference; RPS non-worse on 10/11 (cards_red +0.2%, noise); every ECE delta ≤ 0.001 (≪ 0.02 band). Serving-parity deltas exactly 0.0 throughout (see FA-102 notes).
- **Tuning run (F-4, 2026-07-11, 50 TPE trials × 11 targets on the new feature set): all 11 targets ADOPTED** — committed as `config/lgbm_params/*.json`, loaded by default at training time via `load_tuned_configs` (adoption gate stored in-artifact). Cross-fold RPS, default → tuned (same folds):

  | target | default | tuned | Δ | notable params |
  | --- | --- | --- | --- | --- |
  | passes_total | 6.05383 | **5.91061** | −2.37% | lr 0.0068, 84 leaves, mcs 195 |
  | goals_saves | 0.97325 | **0.96796** | −0.54% | mcs 138, 38 leaves |
  | cards_red | 0.00426 | **0.00424** | −0.47% | mcs 6, 119 leaves |
  | goals_assists | 0.05268 | **0.05245** | −0.44% | mcs 5, lr 0.0112 |
  | goals_total | 0.06925 | **0.06898** | −0.39% | mcs 174, lr 0.0566 |
  | shots_on | 0.18081 | **0.18016** | −0.36% | mcs 198, 7 leaves |
  | offsides | 0.08380 | **0.08352** | −0.33% | mcs 140, lr 0.0438 |
  | shots_total | 0.32063 | **0.31994** | −0.22% | mcs 83, 8 leaves |
  | fouls_committed | 0.43238 | **0.43150** | −0.20% | mcs 60, lr 0.0161 |
  | cards_yellow | 0.09428 | **0.09409** | −0.20% | mcs 55, 7 leaves |
  | fouls_drawn | 0.41661 | **0.41623** | −0.09% | mcs 80, lr 0.0201 |

  Note the pattern: sparse per-player targets (cards_red mcs 6, goals_assists mcs 5) got the small-leaf freedom E-4 predicted they needed, while high-count targets went the other way (passes_total mcs 195 with a slow 0.0068 lr) — and passes_total, the N6 run's one real calibration failure, is also tuning's biggest win. These configs take effect at the FA-107 retrain.
- **Modifier-clip/lr evaluation (E-4, runbook step 4) — deliberately not adopted, disposition recorded:** raising `DEFAULT_PLAYER_MODIFIER_CLIP` requires full feature-table rebuilds per clip variant (a dedicated quota day) and its expected value has shrunk now that the EB career rates supply the elite-player identity signal directly (skill_vs_player_l5 up on 10/11 targets). Current values (clip 1.5, lr 0.05) stay; re-evaluate only if FA-107's live verification still shows elite-player compression. Ticket closed on the acceptance criteria, which this satisfies ("evaluate before adopting" — evaluated as not currently worth a quota day; no adoption).

### P4-pre / FA-106. Run the N6 simulation holdout backtest (F-3, evidence for FA-105)
- **Jira:** FA-106 · Task · Priority High (unblocks FA-105) · 2 pts · Labels `databricks`, `backtest`, `data-gated` · Depends on FA-101 being deployed (otherwise it measures the broken serving path).
- **Agent persona — "Meter":** MLOps operator comfortable driving Databricks jobs on a free-tier budget — batches SQL, watches the daily warehouse quota, and records numbers verbatim into docs. No modeling opinions; produces evidence.
- **Entry point:** `scripts/backtest_simulation.py --season <latest completed>` (never run; quota exhausted 2026-07-07 — see ml_upgrade_backlog F-3). Measures interval coverage, PIT uniformity, team-total bias, and allocation top-1/top-3 — the pre-committed evidence for FA-105's adoption and ADR 0005's revisit triggers.
- **Acceptance criteria:** artifact (JSON + CSV) produced; per-target coverage/PIT/team-total-bias/allocation numbers recorded here and in the ml_upgrade_backlog N6 notes; targets with 80% coverage outside [70, 90] flagged against the ADR 0005 revisit triggers.
- **Run record (2026-07-11, latest completed season, n_sims 5000, seed 7, post-FA-101/FA-104 marts, registered v7 models):** 16,586 player rows / 1,018–1,026 team-fixtures scored per target; 9 fixtures skipped (no expected-minutes rows — pre-lineup-ingestion fixtures; the crash this exposed on first real run was fixed in f502960). w-sweep over `--team-goal-anchor-weight` ∈ {0, 0.25, 0.5, 0.75, 1.0}.
- **w-invariant results (only the goals chain responds to the anchor).** Player 80% interval coverage / PIT max deviation / team top-1 allocation at w = 0: shots_total 0.907/0.055/0.373 · shots_on 0.936/0.023/0.389 · passes_total **0.581/0.189**/0.317 · offsides 0.953/0.015/0.397 · cards_yellow 0.961/0.011/0.175 · fouls_committed 0.888/0.074/0.274 · fouls_drawn 0.891/0.064/0.311 · cards_red 0.995/0.003/0.086 (81 fixtures).
- **Goals family vs w (coverage / PIT dev; saves & assists inherit through the identities):**

  | w | goals_total | goals_saves | goals_assists | saves top1 | goals top3 (team) |
  | --- | --- | --- | --- | --- | --- |
  | 0.00 | 0.962 / 0.007 | 0.988 / 0.008 | 0.966 / 0.006 | 0.976 | 0.941 |
  | 0.25 | 0.970 / 0.013 | 0.984 / 0.014 | 0.975 / 0.013 | 0.953 | 0.938 |
  | 0.50 | 0.975 / 0.025 | 0.975 / 0.025 | 0.980 / 0.024 | 0.937 | 0.935 |
  | 0.75 | 0.980 / 0.033 | 0.964 / 0.036 | 0.982 / 0.033 | 0.897 | 0.932 |
  | 1.00 | 0.982 / 0.039 | 0.956 / 0.043 | 0.983 / 0.037 | 0.810 | 0.932 |

- **Findings against the ADR 0005 revisit triggers:**
  1. **passes_total is genuinely miscalibrated low** — 0.581 coverage at 0.80 nominal with PIT max deviation 0.189: per-player passes intervals are far too narrow. This is the one real trigger hit; the fix (per-player NB dispersion in allocation, not just team-level alpha) landed as **FA-108** — default off pending the re-run below.
  2. The `outside_acceptance_band` flags on goals/cards/offsides are **discreteness over-coverage** (10–90% quantiles on low-mean counts), not miscalibration — their PIT deviations are ≤ 0.015 at w = 0. shots_total (0.907, PIT 0.055) and fouls (PIT 0.064–0.074) are marginal but acceptable.
  3. Report gap noted: `score_simulation_backtest` emits team coverage + mean_actual but no simulated-vs-actual team mean bias column; add one if the anchor question is ever reopened.

### P4 / FA-105. Anchor simulation team totals to the Elo goal model (fixes E-6)
- **Jira:** FA-105 · Story · Priority Medium-High · 5 pts · Labels `simulation`, `monte-carlo` · Depends on FA-106 (evidence).
- **Agent persona — "Croupier":** simulation engineer with a quant-sports background — builds vectorized Monte Carlo engines (numpy Generators, NB via Poisson–Gamma) and has implemented forecast reconciliation (top-down proportional scaling) in hierarchical demand systems. Obsessive about invariants tests and bit-identical seeds.
- **Files:** `football_analytics/simulation.py`, `notebooks/05_fixture_simulation.py`, `tests/test_simulation.py`, ADR 0005 update.
- **Required behavior:** reconcile per-fixture team goal intensities: scale each team's player `goals_total` intensities multiplicatively so their sum matches a blend `w · expected_goals_for_pre + (1 − w) · Σ player means` (w fitted on the FA-106 backtest; shares preserved, structural chains untouched). Extend to `shots_total` only if FA-106 shows totals bias there too. This was the anticipated "team-total model + reconciliation" alternative in ml_upgrade_backlog §2.4 — the Elo layer already prices opponent strength per fixture (E-6) and is free.
- **Acceptance criteria:** invariants tests still pass; determinism under seed preserved; simulated team-goal means across a set of mismatched fixtures track `expected_goals_for_pre` ordering; scoreline matrices differ visibly between a favorite-vs-minnow and an even fixture; FA-106 coverage/bias numbers recorded before/after.
- **Implementation notes (2026-07-11) — code landed, anchor off pending FA-106:**
  - `apply_team_goal_anchor` (simulation.py, called first inside `simulate_fixture`): per team, one multiplicative factor on player `rate90_goals_total` so the expected total = `w · expected_goals_for_pre + (1 − w) · Σ player means`. Shares preserved; chains untouched (only the shots_on → goals conversion scales, `goals ≤ shots_on` still bound per draw); assists/saves inherit anchored goals through their identities. Pure function; exact no-op (same object) at `w = 0` or without anchors.
  - `SimulationConfig.team_goal_anchor_weight` **defaults to 0.0 — behavior identical to v1 until FA-106 fits w** (measure-first, §2.2). The weight is in the config digest and `SIMULATION_ENGINE_VERSION` bumped to 1.1.0, so anchored sets supersede unanchored ones cleanly.
  - Anchor plumbing: `SimulationInputs.expected_team_goals` (str team-id keys, mirrors `alpha_team`); notebook 05 batch-reads `fct_football__team_elo_history.expected_goals_for_pre` in one query and exposes a `team_goal_anchor_weight` widget; `simulate_completed_fixtures` reads anchors off the feature-mart rows so the FA-106 w-sweep needs no extra queries.
  - Tests: exact-anchor scaling + share preservation + other-target immutability; blend math at w=0.5; no-op identities; digest sensitivity; mismatched-vs-even fixture differentiation with means tracking the anchor (3.0/0.5 vs 1.3/1.3, favorite win-rate ≫ even) and determinism under seed with the anchor on. ADR 0005 amended (engine 1.1.0 section) with the clip-undershoot caveat and the shots_total extension condition.
  - Remaining for ✅: fit w on the FA-106 sweep, flip the notebook/job default, record before/after coverage + bias here.
- **Fit decision (2026-07-11): w = 0 adopted — the anchor stays OFF.** The FA-106 sweep shows monotone degradation of the whole goals family as w rises (goals_total PIT deviation 0.007 → 0.039, saves top-1 allocation 0.976 → 0.810, assists PIT 0.006 → 0.037; table in FA-106 notes) and no metric improves. Post-FA-101 the player models receive their opponent-Elo features at serving, so Σ player means already prices opponent strength; blending in the Elo-only goal expectation double-counts it and mis-levels the totals. E-6's compression was a symptom of E-2 and is resolved by FA-101 — the anticipated reconciliation layer is not needed on current evidence. The engine machinery stays (digest-tracked `team_goal_anchor_weight`, defaults 0.0 everywhere) so the question can be reopened cheaply if a future backtest shows totals bias; ADR 0005 amendment updated with this outcome.

### P5 / FA-107. Re-run the data-gated follow-ups on the fixed stack, retrain + re-register
- **Jira:** FA-107 · Story · Priority Medium · 5 pts · Labels `mlops`, `release` · Depends on FA-101, FA-102 (parity gate green), FA-104 (feature set final).
- **Agent persona — "Registrar":** ML release engineer — owns model registries, version tags, and rollout verification. Treats every registration as an audited artifact: lineage params, dispersion tags, alias hygiene. Never ships without a monitored first live run and knows the MLflow free-tier metric quota by heart.
- **Tasks:**
  1. F-1 (M4 sparse-target decision): `scripts/train_poisson_lgbm.py --backtest --targets cards_yellow,cards_red,goals_total,goals_assists`, apply the pre-committed ECE rule verbatim, update ADR 0003.
  2. F-5 (SHAP review) re-run on the FA-104 feature set.
  3. Retrain + re-register all 11 targets (`scripts/train_poisson_lgbm.py --registered-model-prefix football_analytics.gold.player_event`); confirm α tags + pyfunc artifacts on the new versions.
  4. Verify the first live fixture with `mon_football__prediction_vs_actual`, `mon_football__inference_feature_health` (FA-103), and the FA-102 parity delta.
- **Acceptance criteria:** per-item numbers recorded in this file; the ml_upgrade_backlog Follow-Ups table updated to point here; new versions serving with a demonstrably player-differentiated spread on a live fixture (top-l5 player ≫ team median, recorded).
- **Progress (2026-07-12) — tasks 1–3 done, task 4 awaits the next live fixture:**
  1. **F-1 (M4 sparse-target decision): NB kept for all four targets, no hurdles.** Rule applied verbatim to the FA-104 backtest evidence (`ece_ge_1` cross-fold means: cards_red 0.0005, goals_assists 0.0044, goals_total 0.0066, cards_yellow 0.0098 — all ≤ 0.02 with ≥ 2× headroom), corroborated by the retrain split. ADR 0003 updated with the table; ml_upgrade_backlog M4/F-1 closed.
  2. **F-5 (feature review on the FA-104 set): no-op on removals/additions — and the E-3 pathology is confirmed gone.** Gain shares from the retrain artifacts: `shots_total_eb_p90` is the #1 shots feature at **55.1%** (v7's degenerate `player_offensive_modifier_pre` was 24.3%); goals is led by the EB shooting family (42.1% + 15.4%); saves is led by opponent-strength context (`elo_possession_interaction` 17.2%, `elo_expected_score` 16.7%) instead of the old `goals_saves_l5_p90` GK proxy; the position one-hots earn real gain (`position_group_g` 4.2% on shots, `position_group_d` 2.9% on passes). The Elo modifiers left every top-10 — the EB rates carry player identity better, as designed. The M5 candidate additions (rest days, GK quality, sub tendency) remain future work requiring backtest justification. Note: SHAP plots skipped by the run (`shap` not installed on serverless — `%pip install shap` next time); gain evidence is the same class E-3 used.
  3. **Retrain + re-register (2026-07-12): all 11 targets on the tuned configs** (`Using tuned configs for:` all 11; run `fa107-retrain-tuned`, training run 91f128a5). New UC versions registered (spot-checked: shots_total v8, cards_red v3, passes_total v7), **α tags verified via the UC API** (shots_total v8: alpha_player 0.0629 / alpha_team 0.0559; cards_red v3: 5.756 / 0.449) and pyfunc artifacts logged in metric-free register runs. Retrain validation headline (single temporal split): shots_total rank_spearman 0.461 / skill_vs_player_l5 0.093, passes_total rank_spearman 0.890, every `ece_ge_1` ≤ 0.032.
  4. **Pending:** first live fixture with the new versions — at T-60 run the prematch inference job, then record the shots_total spread (top-L5 player ≫ team median; July 9 baseline: flat at ~1.09) plus a quiet `mon_football__inference_feature_health`. This also closes FA-101's last acceptance item.

### P6 / FA-108. Per-player NB dispersion in the simulation allocation (fixes the FA-106 passes_total coverage trigger)
- **Jira:** FA-108 · Story · Priority High · 3 pts · Labels `simulation`, `monte-carlo` · Depends on FA-106 (evidence source); independent of FA-107. Real-data validation shares the FA-107 quota batch.
- **Agent persona — "Croupier"** (same as FA-105): simulation engineer with a quant-sports background — vectorized Monte Carlo, NB via Poisson–Gamma, obsessive about invariants tests and bit-identical seeds.
- **Files:** `football_analytics/simulation.py`, `football_analytics/simulation_backtest.py`, `football_analytics/inference.py` (`load_dispersion_tags`), `notebooks/05_fixture_simulation.py`, `scripts/backtest_simulation.py`, `tests/test_simulation.py`, `tests/test_simulation_backtest.py`, ADR 0005 amendment (engine 1.2.0).
- **Problem (FA-106 evidence):** `passes_total` player 80% interval coverage 0.581 at 0.80 nominal, PIT max deviation 0.189 — the one real ADR 0005 revisit trigger. Structural cause: multinomial allocation conditions on the team total, capping player variance near `μ·(1−share)` + a share²-scaled team-dispersion term, while real per-player variance is `μ + α_player·μ²` (M1 measures `alpha_player > alpha_team` everywhere — team totals average heterogeneity out). The shortfall scales with μ², so only high-count passes tripped the trigger.
- **Design:** `sample_dispersed_allocation_weights` — for targets in `SimulationConfig.player_dispersion_targets`, multiply each player's allocation weight by an i.i.d. mean-1 `Gamma(1/α_player, α_player)` multiplier between the team-total draw and the multinomial split (a generalized Dirichlet-multinomial). Player marginals gain `≈ α_player·μ²` variance; expected shares, structural chains, and the measured-`alpha_team` NB team-total draw are untouched, so the healthy team calibration is preserved by construction. A single-concentration Dirichlet-multinomial was rejected (one concentration cannot match heterogeneous player shares; the gamma multipliers give every player his own `α·μ_p²` automatically), as was drawing the total from the perturbed Poisson sum (would imply team dispersion instead of honoring the M1 `alpha_team` contract).
- **Gating & lineage:** default `player_dispersion_targets = ()` — **OFF, bit-identical to engine 1.1.0** (the helper consumes no randomness when gated off or when `α_player` is unfitted/0). The tuple is in the config digest (order-insensitive) and `SIMULATION_ENGINE_VERSION` bumped to **1.2.0**, so dispersed sim sets supersede v1 sets. `alpha_player` plumbing mirrors `alpha_team`: version tags → `load_dispersion_tags` (one registry call reads both tags; notebook 05 widget `player_dispersion_targets`) / `LoadedEventModel.alpha_player` → `SimulationInputs.alpha_player`; `scripts/backtest_simulation.py --player-dispersion-targets passes_total` for the re-run.
- **Synthetic evidence (2026-07-11, tests; NB(α=0.3) passes world, 20 fixtures / 440 players / 40 team-fixtures, n_sims 1500, seed 79):** v1 allocation reproduces the FA-106 symptom — player coverage **0.405**, PIT deviation **0.232**; layer on restores **0.798** coverage at 0.80 nominal, PIT deviation **0.023**; shots_total (0.905/0.032) and team passes coverage (0.825) identical in both runs. Unit-level: starter passes variance 40.1 → 184.0 vs NB target 200 (finite-team normalization ≈ (1−share)²; team-total mean/variance unchanged). Suite: invariants parametrized over the gate, bit-identical no-op with the flag off, determinism under seed with it on, digest sensitivity/order-insensitivity, weight-moment checks.
- **Acceptance criteria (remaining for ✅ DONE — needs the deployed stack, user-run on Databricks):**
  1. Re-run N6 on the latest completed season, both arms, same seed: `python scripts/backtest_simulation.py --season <latest> [--player-dispersion-targets passes_total]` (control arm optional — FA-106's w=0 run is the standing baseline; one quota batch with the FA-107 runs).
  2. Adopt if passes_total player coverage lands in **[0.70, 0.90]** with PIT deviation materially down (< ~0.08) and no other target's coverage/PIT/allocation degrades beyond noise. Record before/after numbers here.
  3. On adoption: flip the defaults — notebook 05 widget + `resources/*.yml` job param to `passes_total` (leave `SimulationConfig` default empty; adoption is a deployment decision, mirroring FA-105's w handling) — and update the ADR 0005 amendment with the real-data outcome.
- **Real-data validation, treatment arm (2026-07-12, latest completed season, n_sims 5000, seed 7, flag `passes_total` ON — note: ran on the FA-107 retrained model versions, so it is not a clean single-variable isolation vs the FA-106 baseline):** passes_total player coverage **0.581 → 0.710** (inside the band, `outside_acceptance_band` now False), PIT max deviation **0.189 → 0.139**, allocation top1 0.317 → 0.335 / top3 0.665 → 0.710; no other target degraded (most improved — attributable to the new models: e.g. cards_red top1 0.086 → 0.519 on n=81, goals_total top1 0.323 → 0.351). Even with the layer ON, passes coverage sits *below* nominal (0.71 < 0.80), so the layer cannot be over-widening.
- **Verdict per the pre-committed rule: adoption HELD.** Criterion 2 is split — coverage gate met, PIT gate missed (0.139 > ~0.08; still the worst target). Defaults stay off. Next steps:
  1. **Control arm** (one run, new models, flag off): `backtest_simulation.py --season <latest>` with no flag — attributes the passes gain between the new passes model (tuning's biggest win, −2.4% RPS) and the dispersion layer, and doubles as the post-retrain N6 baseline for FA-107's record.
  2. If the layer's contribution is confirmed but PIT stays > ~0.08, the residual is likely structure a single global `α_player` cannot express (role bimodality; the (1−share)² variance shave is only ~8–17% and cannot explain the gap). Candidate refinements, synthetic-gated first: share-compensated gamma variance (multiplier variance × 1/(1−share)² per player), or per-position-group `α_player` fitted in M1 style.
- **Control arm + final decision (2026-07-12):** control (retrained models, flag off): passes coverage **0.609**, PIT **0.189** — the new passes model alone moved coverage +0.03 and PIT not at all, so **the dispersion layer owns the entire jump into the band** (+0.10 coverage, −0.05 PIT) at zero cost (all other targets byte-identical between isolated arms). Three-arm table: 0.581/0.189 → 0.609/0.189 → 0.710/0.139. **ADOPTED on dominance with the user's explicit sign-off — the PIT gate (< ~0.08) is recorded as waived, not passed.** Defaults flipped at the deployment layer only (notebook 05 widget + backtest CLI default `passes_total`; `SimulationConfig` stays engine-neutral). The ADR 0005 passes trigger REMAINS OPEN: follow-up is per-position-group `α_player` (M1-style per-group fit + version tags + sim consumption), revalidated by another N6 arm, closing the trigger only when PIT < ~0.08. Requires a `bundle deploy` to reach the production job.

### P7 / FA-109. Per-position-group `α_player` (closes the ADR 0005 passes PIT trigger left open by FA-108)
- **Jira:** FA-109 · Story · Priority Medium-High · 3 pts · Labels `simulation`, `mlops` · Depends on FA-108 (adopted layer); revalidation rides the next retrain.
- **Problem:** with the FA-108 layer adopted, passes_total holdout PIT max deviation is **0.139** (target < ~0.08). One pooled `α_player` mis-shares dispersion across roles: it over-disperses the tight groups and under-disperses the loose ones, so the aggregate PIT stays non-uniform even when coverage looks fine.
- **Implementation (2026-07-12, code landed — gated off):**
  - `estimate_position_group_nb_alphas` (evaluation.py): M1-style method-of-moments fit per `position_group` on validation residuals; groups under `min_rows=200` are omitted (consumers fall back to the pooled alpha — fitting noise on thin segments would inject variance, not remove bias).
  - Training fits per-group alphas next to the pooled ones, stores them on `ExposurePoissonLightGBMModel.alpha_player_by_position` (getattr-safe for old pickles), logs a `{target}_alpha_player_by_position` param, and the register loop writes `alpha_player_g/d/m/f` version tags beside `alpha_player`/`alpha_team`.
  - Read path mirrors FA-108: `load_dispersion_tags` returns an extra `alpha_player_by_position` map ({target: {group: alpha}}); `LoadedEventModel` carries it; `simulate_completed_fixtures` and notebook 05 plumb it into `SimulationInputs.alpha_player_by_position`.
  - Engine: `SimulationConfig.player_dispersion_by_position` (default **False**, digest-tracked; engine 1.3.0). When on, `_resolve_player_dispersion` gives each player his group's alpha (pooled fallback for missing/unknown groups) and `sample_dispersed_allocation_weights` accepts a per-player alpha vector — zero-alpha rows keep multiplier exactly 1; the scalar path stays draw-stream-identical to FA-108.
  - Synthetic gate (tests): in a role-heterogeneous NB world (D α=0.01, F α=1.2) the pooled arm reproduces the real holdout signature (PIT deviation 0.141 vs the measured 0.139) while the per-position arm restores calibration (PIT 0.064, coverage 0.92); plus per-group fitter recovery/thin-group fallback, per-player-vector weight moments, bit-identical gating, determinism, and digest tests.
- **Acceptance criteria (remaining for ✅ DONE — user-run on Databricks):**
  1. Retrain + re-register (any `train_poisson_lgbm.py --registered-model-prefix ...` run post-FA-109) so versions carry the `alpha_player_g/d/m/f` tags — the current versions predate the per-group fit and fall back to pooled.
  2. N6 revalidation: `backtest_simulation.py --season <latest> --player-dispersion-by-position` (defaults already include `passes_total`) vs the pooled arm (2026-07-12 treatment run is the standing baseline: coverage 0.710, PIT 0.139).
  3. Adopt if passes PIT max deviation lands **< ~0.08** with coverage still in [0.70, 0.90] and nothing else degraded: flip the notebook 05 `player_dispersion_by_position` widget default to `"true"`, record numbers here, and **close the ADR 0005 passes trigger**. If PIT stays above, record and stop — the next hypothesis (game-state/tempo structure) needs its own evidence, not another dispersion knob.

---

## Dependency graph (Jira ids)

```text
FA-101 (P1, no deps) ──→ FA-102 (P2a, parity delta ≈ 0 gate) ──→ FA-104 (P3) ──→ FA-107 (P5)
FA-101 ────────────────→ FA-104
FA-101 deployed ───────→ FA-106 (P4-pre, F-3 run) ──→ FA-105 (P4)
FA-106 ────────────────→ FA-108 (P6, passes dispersion; validation rides the FA-107 quota batch)
FA-108 ────────────────→ FA-109 (P7, per-position α_player; revalidation rides the next retrain)
FA-103 (P2b) — no deps, start in parallel with FA-101
FA-105 and FA-102/FA-104 are independent branches
```

**Sprint order:** FA-101 + FA-103 now → FA-102 → FA-104 ∥ (FA-106 → FA-105) → FA-108 → FA-107.
