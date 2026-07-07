# ML Upgrade Backlog — Expected Minutes v2, Evaluation, Model Families, Fixture Simulation

Successor to [`implementation_backlog.md`](implementation_backlog.md) (Epics A–J, all ✅ DONE 2026-07-06). This backlog upgrades the *quality* of the trained models and adds a Monte Carlo fixture-simulation layer on top of them. Source of truth for business rules remains [`business_logic.md`](business_logic.md).

**Audience: AI agents.** Every ticket is self-contained: it states the current behavior (with file anchors), the required behavior, the contract, and the acceptance criteria. Do not infer scope beyond the acceptance criteria; open a follow-up ticket instead.

---

## 0. Working Agreement (applies to every ticket)

1. **One commit per ticket, pushed to `main` immediately** (repo convention; CI tracks each merge). Commit message format: `<imperative summary> (ml-backlog <ticket-id>)`.
2. Before commit: `pytest` green from repo root; `dbt parse` clean from `dbt/` for any dbt change; dbt unit tests added in the ticket run in `dbt build`.
3. Pure logic lives in `football_analytics/` and is unit-testable without Spark/MLflow (repo convention — see `football_analytics/inference.py` module docstring). Notebooks and jobs only wire effects.
4. MLflow metric discipline: Databricks free tier caps **1000 metrics per logged model** (see commits `c5e03d1`, `f2b47f4`). Log only aggregate scalars as MLflow metrics; anything per-fold, per-bin, or per-player goes into **artifacts** (CSV/JSON). Model registration stays in metric-free runs.
5. Leakage rules from `business_logic.md` §13.4 and §19 are non-negotiable: chronological evaluation only, `_pre` features only, rolling windows strictly prior to the current fixture.
6. When a ticket lands, mark it ✅ DONE in this file (with date) inside the same commit and record implementation notes the way `implementation_backlog.md` does — the notes are load-bearing context for the next agent.

**Suggested execution order:** O1 → K1–K5 → L1–L5 → M1–M2 → N1–N7 (N1 may start once K5 + M1 are done) → M3–M5 → O2–O3.

---

## 1. Current State (verified 2026-07-07)

- **Training** (`football_analytics/ml_training.py`, `scripts/train_poisson_lgbm.py`): 11 independent exposure-offset Poisson LightGBM models (one per `DEFAULT_TARGET_COLUMNS` event) against `gold.fct_football__player_event_features`. Exposure = `games_minutes` (train) / `expected_minutes` (inference) via `init_score=log(exposure)`. Chronological split (`temporal_train_validation_split` or `season_train_validation_split`). Registered as `<prefix>__<target>` in Unity Catalog. Metrics: `poisson_logloss`, `mae`, `rmse`, overall + per position group.
- **Inference** (`football_analytics/inference.py`, `notebooks/04_prematch_inference_predict.py`, `resources/prematch_inference_pipeline.yml`): T-60/T-40 polling job → active-flag prediction sets in `gold.pred_football__player_event_predictions` with Poisson `predicted_p_ge_{1,2,3}`.
- **Expected minutes** (`dbt/models/marts/fct_football__player_expected_minutes.sql`): L5 trimmed mean of played appearances; `is_starting` consulted **only** in the zero-history fallback (starter 85 / sub 15). See ADR 0002.
- **Monitoring**: `mon_football__prediction_vs_actual`, `mon_football__prediction_monitoring`, `mon_football__prediction_coverage` dbt views.
- **Known-stale artifact**: `multi_player_predictions.ipynb` (repo root, untracked) references the deleted mart `fct_football__player_shot_features` and the pre-F3 model prefix `player_prop_poisson_lgbm_*`, hardcodes 90 projected minutes, and uses the *last historical* lineup instead of confirmed lineups.

### 1.1 Confirmed defects this backlog fixes

**D-1 — Expected minutes ignore the confirmed role.** In `fct_football__player_expected_minutes.sql` (the `case` at the `expected_minutes_raw` derivation), any player with ≥1 prior played appearance gets a pure history estimate. Consequences:
- A player who usually starts (L5 ≈ 90') but is **named on the bench** today still gets `expected_minutes ≈ 88` → every count prediction for him is inflated ~4–6×. This is the user-reported bug.
- A habitual substitute **promoted to the XI** gets `expected_minutes ≈ 15–25` → predictions deflated ~4×.
- Unused-sub matches (named on bench, 0 minutes) are excluded by the `games_minutes between 1 and 130` filter, so bench players' expected minutes never price in `P(does not come on)` — bench estimates are biased upward even before the role mismatch.

**D-2 — No distributional honesty.** `predicted_p_ge_k` assumes Poisson. Several targets are overdispersed at the player level and (more importantly) at the team-total level (`passes_total`, `shots_total`, `fouls_*`); Poisson underprices tails. No dispersion diagnostic exists to even see this.

**D-3 — No skill measurement.** Metrics have no baseline reference, so "poisson_logloss = 0.61" cannot be interpreted. No calibration measurement for the probability outputs the product actually exposes. No ranking measurement for "which player leads the team in event X" — the exact question the fixture simulation must answer.

**D-4 — No joint match view.** Predictions are per-player-per-event marginals. Nothing enforces coherence (player shots_on ≤ shots_total; team fouls committed = opponent fouls drawn; GK saves ≈ opponent shots on target − goals). There is no fixture-level simulated game.

---

## 2. Design Decisions (rationale, alternatives considered)

These decisions were each stress-tested against at least one alternative; the losing alternative and the reason are recorded so future agents don't re-litigate silently.

### 2.1 Expected minutes v2 = role-conditional decomposition (not an ML model)

`expected_minutes = P(plays | today's role) × E[minutes | plays, today's role]`, estimated from role-split history:

- **Confirmed starter:** `P(plays) = 1` (on the pitch at minute 0). `E[minutes | start]` = trimmed mean of minutes over the player's **last 5 starts**, shrunk toward the starter prior: `(n·hist + k·prior) / (n + k)`, prior = 85, `k = 2`.
- **Confirmed bench:** `P(plays)` = share of the player's **last 10 bench namings** in which he got ≥1 minute (unused-sub namings count as 0-minute observations), shrunk toward the global bench-appearance rate with `k = 4`. `E[minutes | plays]` = trimmed mean of his nonzero bench minutes, shrunk toward the global bench-cameo mean with `k = 2`.
- Role history comes from `stg_football__lineups` (who was named, in which role) left-joined to `stg_football__player_match_stats` (minutes, coalesced to 0), both strictly prior to the target fixture.

*Alternative considered — dedicated minutes ML model (ADR 0002 "future work"):* highest ceiling, but international-football samples per player are small, the estimator above fixes the reported defect deterministically and is dbt-unit-testable, and we get an evaluation harness (K4) that can later prove an ML model beats it before we pay for one. Deferred, not rejected — see K-follow-up note in K4.

*Why keep the mean (not the full distribution) in the prediction table:* the Poisson/NB mean is linear in exposure, so `E[count] = rate_per_90 × E[exposure]` is exact even when minutes are bimodal. The **simulation** is where bimodality matters, so K5 exposes `p_plays` and `expected_minutes_if_plays` separately for Epic N to consume.

### 2.2 Measure first, then change models (L before M)

New metrics land before any model-family change so every change is judged by: skill vs naive baselines, calibration of `p_ge_k`, dispersion index, ranked probability score, and within-fixture ranking quality — under a rolling-origin (multi-fold chronological) backtest, not a single split.

### 2.3 Negative binomial via two-stage fit (not a custom objective)

Keep the LightGBM Poisson booster for the conditional mean; fit a per-target dispersion `α` on validation residuals by method of moments (`Var = μ + α·μ²`); use the NB pmf for `p_ge_k` and simulation when `α > 0`.

*Alternative considered — native NB / custom objective:* LightGBM has no NB objective; custom objectives are fragile in production and would invalidate the registered-booster contract. Two-stage is standard actuarial practice, changes no registered artifact, and is exactly what the simulator needs (it samples team totals, where dispersion matters most).

### 2.4 Simulation = team-total draw + multinomial allocation + structural chains

Per iteration: (1) sample bench participation and minutes per player from the K5 decomposition; (2) sample **team totals** per event from NB with mean = Σ player intensities × sampled exposure; (3) allocate totals to players via **multinomial** with weights ∝ per-90 intensity × sampled minutes; (4) enforce coherence structurally: `shots_on ~ Binomial(shots_total, μ_on/μ_total)` per player, `goals ~ Binomial(shots_on, μ_goals/μ_on)`, one shared foul total per directed team pair (committed by A = drawn by B), GK saves = opponent shots_on − opponent goals (clipped at 0), ≤1 assist per goal allocated to teammates of the scorer, yellow ≤ 2 and red ≤ 1 per player.

*Alternative considered — independent Poisson/NB draws per player:* simpler but produces incoherent games (player sums ≠ any sensible team total; shots_on > shots_total possible; saves uncorrelated with opponent shooting). Multinomial-conditioned-on-total guarantees adding-up, induces realistic mild negative correlation between teammates (finite events to share), and directly answers "we predict 10 shots on target — who takes them".

*Alternative considered — separate team-total models + reconciliation:* better-calibrated totals in theory, but adds 11 more models to train/version/monitor and a reconciliation layer. v1 uses Σ player means (coherent by construction) with NB dispersion fitted **at the team-total level** (M1); a team-total model is a measured follow-up (N6 will show whether totals are miscalibrated).

*Known v1 simplifications (document in ADR 0004, revisit only with evidence):* no red-card → minutes truncation feedback; no cross-event tempo factor beyond the structural chains (each event family gets its own NB draw); starters' minutes not resampled (fixed at expected value). All three are listed with their revisit-triggers in N7.

### 2.5 The simulator consumes the governed prediction table, not the models

Inputs are the **active prediction set** (`pred_football__player_event_predictions`) joined to the expected-minutes decomposition — not fresh model calls. Per-90 intensity is recovered as `predicted_mean / expected_exposure`. This keeps the simulator decoupled (rerunnable, auditable, model-agnostic), and its lineage is simply `prediction_set_id`.

---

## Epic O (part 1) — Repo Hygiene Blocker

### O1. Track `docs/business_logic.md`; ignore `dbt/.user.yml`; delete stale notebook — ✅ DONE (2026-07-07)
- **Files:** `.gitignore`, `docs/business_logic.md` (currently untracked!), `multi_player_predictions.ipynb` (delete).
- **Depends on:** nothing. **Do this first** — every other ticket cites `business_logic.md`, and agents working from a fresh clone will not have it while it stays untracked.
- **Acceptance criteria:**
  - `docs/business_logic.md` committed as-is.
  - `dbt/.user.yml` added to `.gitignore` (dbt-generated local user id; never commit).
  - `multi_player_predictions.ipynb` deleted: it reads the deleted `fct_football__player_shot_features` mart and `player_prop_poisson_lgbm_*` model names (superseded by F3 naming), so it cannot run; its use case is replaced by notebook 04 output and Epic N's notebook 05. Verify with grep that nothing references it.
- **Implementation notes:** grep confirmed the only reference to the notebook was this backlog. `.gitignore` gained `dbt/.user.yml` under a dedicated comment; `.databricks/` was already ignored. `business_logic.md` committed unmodified.

---

## Epic K — Expected Minutes v2 (fixes D-1)

### K1. Role-conditional estimator in `fct_football__player_expected_minutes` — ✅ DONE (2026-07-07)
- **Files:** `dbt/models/marts/fct_football__player_expected_minutes.sql`, `dbt/models/marts/schema.yml`, `dbt/dbt_project.yml` (vars).
- **Depends on:** O1.
- **Current behavior:** single history CTE (`prior_appearances`, played minutes 1..130, role-blind); `is_starting` used only in the zero-history `case` branch.
- **Required behavior:** implement §2.1 of this doc. Concretely:
  - Build role history from `stg_football__lineups` (prior fixtures, per player, with `is_starting`) left-joined to `stg_football__player_match_stats` on (fixture_id, player_id), `coalesce(games_minutes, 0)` — so an unused sub is a 0-minute bench observation. Cap minutes at 130 as today; exclude nothing by minutes for bench rows (zeros are signal), keep the 1..130 "played" rule for **starter** rows only to guard data errors.
  - Starter branch: last 5 **starts**; trimmed mean when ≥3 (drop min+max), plain mean 1–2, prior-only when 0; then shrink: `(n_starts·hist + 2·85) / (n_starts + 2)` — note shrinkage applies in *all* starter branches (with 0 history it degenerates to 85).
  - Bench branch: last 10 bench namings → `p_plays_raw` = share with minutes ≥ 1, shrunk `(n_bench·p_raw + 4·var('bench_participation_prior')) / (n_bench + 4)`; `if_plays` = trimmed mean of the **nonzero** bench minutes (same ≥3/1–2/0 trimming ladder), shrunk with `k=2` toward `var('bench_cameo_minutes_prior')`; `expected_minutes = p_plays × if_plays`.
  - dbt vars with provisional values (K3 calibrates): `starter_minutes_prior: 85`, `bench_participation_prior: 0.5`, `bench_cameo_minutes_prior: 25`.
  - **New output columns** (existing columns and grain unchanged — `expected_minutes` stays populated for every lineup row, so `fct_football__player_event_features.expected_exposure` keeps working untouched): `p_plays` (1.0 for starters), `expected_minutes_if_plays`, `starts_in_history_count`, `bench_namings_in_history_count`. Extend `expected_minutes_method` accepted values: `starter_l5_trimmed_shrunk`, `starter_low_history_shrunk`, `starter_prior`, `bench_l10_shrunk`, `bench_low_history_shrunk`, `bench_prior`.
- **Acceptance criteria:**
  - A player with five 90-minute starts who is named **on the bench** gets `expected_minutes` in the low-teens-to-~30 range (driven by priors), NOT ~90.
  - A habitual sub promoted to the XI gets a starter-branch estimate ≥ ~70, NOT his bench cameo average.
  - Unused-sub namings lower `p_plays`.
  - `dbt parse` clean; existing unit tests updated (the old red-card regression case must be re-expressed against the starter branch).
- **Implementation notes:**
  - **Deviation from the ticket text (improvement):** role history comes from `stg_football__player_match_stats.games_substitute` instead of joining historical `stg_football__lineups`. Stats rows exist only for completed fixtures, so future fixtures can never contribute phantom 0-minute bench observations (the lineups join would have needed an explicit completion filter), coverage does not depend on historical lineup ingestion, and the SQL stays single-source. Current-fixture role still comes from `stg_football__lineups.is_starting`.
  - `p_plays` shrinkage: `(cameos + 4*prior) / (namings + 4)`; `if_plays` shrinkage `(n*hist + 2*prior) / (n + 2)` in both branches (degenerates to the prior with n=0, so the method labels `starter_prior`/`bench_prior` describe the same formula at n=0).
  - `prior_appearance_count` kept, redefined as the depth of the active branch's history; `minutes_avg_l5` (role-blind, no consumers, embodied the bug) deleted.
  - Bounds singular test extended: `p_plays` ∈ [0,1], `expected_minutes_if_plays` ∈ [0,120], starters must have `p_plays = 1`.
  - Unit test uses a 3-start history (90/90/25-red-card) so the shrunk trimmed mean is exactly 88.0 — float-exact for dbt's row comparison.

### K2. dbt unit tests for the v2 estimator — ✅ DONE (2026-07-07)
- **Files:** `dbt/models/marts/fct_football__player_expected_minutes_unit_tests.yml`, `dbt/tests/`.
- **Depends on:** K1.
- **Acceptance criteria (mocked-input dbt unit tests, one scenario each):**
  - **Benched-regular regression (the user-reported bug):** history = 5 starts × 90'; today named sub → expected_minutes < 40 and method = bench branch.
  - **Promoted-sub:** history = 5 bench cameos × 20'; today in XI → expected_minutes ≥ 70 (shrunk starter prior dominates).
  - **Unused-sub counting:** history = 4 bench namings, 2 unused (0') + 2 × 30' → `p_plays` reflects 2/4 shrunk toward prior; `if_plays` uses only the two 30' observations.
  - **Red-card robustness preserved:** starter with one 25' start among four 90' starts → trimmed mean drops it (≈90 before shrinkage).
  - Bounds test still enforces [0, 120] and `p_plays` ∈ [0, 1].
- **Implementation notes:** second unit test `expected_minutes_conditions_on_confirmed_role` covers the three defect scenarios with float-exact expectations (12.5 / 85.0 / 13.75); the red-card case landed in K1's rewritten first test (3-start history so the shrunk result is exactly 88.0). One unused-sub row uses `games_minutes: null` to pin the coalesce-to-0 path. Bounds extensions landed in K1.

### K3. Calibrate the priors from production history + supersede ADR 0002 — ✅ DONE (2026-07-07)
- **Files:** `dbt/dbt_project.yml` (var values), `docs/adr/0002-expected-minutes-estimator.md` (mark Superseded), new `docs/adr/0004-expected-minutes-role-conditional.md`.
- **Depends on:** K1. **Requires Databricks access** (use `scripts/run_query.py`).
- **Acceptance criteria:**
  - One SQL query (recorded in the ADR) computes from all completed fixtures: global bench-appearance rate (`P(minutes ≥ 1 | named on bench)`), mean nonzero bench minutes, and mean starter minutes. Update the three dbt vars to the measured values (rounded).
  - ADR 0004 records the estimator, priors + provenance query, shrinkage constants, and the deferred ML-model alternative with its adoption trigger (see K4).
- **Implementation notes:** measured over n=90,470 completed player rows via `scripts/run_query.py`: starter mean 80.41', bench participation 0.3952, bench cameo mean 22.97' → vars set to 80.0 / 0.4 / 23.0 (all three provisional values were too generous). Unit tests pin their own priors via dbt unit-test `overrides.vars`, so recalibration never breaks float-exact expectations. ADR 0002 marked superseded.

### K4. Expected-minutes accuracy monitoring + estimator evaluation harness — ✅ DONE (2026-07-07)
- **Files:** new `dbt/models/marts/mon_football__expected_minutes_accuracy.sql` (+ schema.yml), optionally a small helper in `football_analytics/`.
- **Depends on:** K1.
- **Acceptance criteria:**
  - dbt view over completed fixtures joining the expected-minutes mart to actual `games_minutes`: MAE and bias segmented by `is_starting` × `expected_minutes_method` × prior-history-count bucket, plus row counts. This is the harness that must show v2 < v1 error for bench players, and it is the gate any future ML minutes model must beat (adoption trigger for the ADR 0004 deferred alternative: an ML estimator must reduce overall MAE ≥ 10% relative on a season-holdout before it replaces the SQL estimator).
  - View is empty-but-queryable before data lands (same convention as the other `mon_` views).
- **Implementation notes:** view segments by `league_season × is_starting × expected_minutes_method × position_group` (the method label already encodes the history-depth bucket, so a separate bucket column would have been redundant). Adds `avg_p_plays` vs `actual_played_share` so bench participation calibration is directly visible. Actual minutes coalesce to 0 — a bench player who never came on played 0 minutes and p_plays prices that outcome.

### K5. Expose the decomposition to the feature mart and prediction table — ✅ DONE (2026-07-07)
- **Files:** `dbt/models/marts/fct_football__player_event_features.sql` (+ schema.yml), `football_analytics/inference.py`, `notebooks/04_prematch_inference_predict.py` (only if column pass-through needs it), `tests/test_inference.py`.
- **Depends on:** K1.
- **Acceptance criteria:**
  - Feature mart carries `p_plays` and `expected_minutes_if_plays` through from the expected-minutes mart (nullable for training rows without lineup data, same as `expected_minutes` today). `expected_exposure` definition unchanged.
  - `build_prediction_records` writes two new columns to `pred_football__player_event_predictions`: `p_plays`, `expected_minutes_if_plays`; `ensure_player_event_predictions_table` DDL extended (additive `ALTER TABLE ... ADD COLUMNS` guard for the existing table, or document a one-time migration in the ticket notes); `_PREDICTION_COLUMN_TYPES` updated.
  - Unit test proves the columns land in prediction records and survive the writer's type casts.
- **Implementation notes:** migration handled via `.option("mergeSchema", "true")` on the Delta append (additive columns land on pre-K5 tables automatically; asserted in the writer test) instead of an `ALTER TABLE` guard. Records tolerate feature tables without the columns (null pass-through, covered by `test_prediction_records_tolerate_missing_decomposition_columns`). The feature-mart leakage unit test needed no fixture changes — its expected-minutes mock uses `rows: []`, which materializes the real model's full column set.

---

## Epic L — Evaluation & Metrics Upgrade (fixes D-3, enables M/N decisions)

### L1. `football_analytics/evaluation.py` — proper scoring + calibration + dispersion primitives — ✅ DONE (2026-07-07)
- **Files:** new `football_analytics/evaluation.py`, new `tests/test_evaluation.py`.
- **Depends on:** nothing (pure numpy/pandas; no LightGBM/MLflow imports at module top — follow the lazy-import convention of `ml_training.py`).
- **Required functions (pure, unit-tested against hand-computed values):**
  - `ranked_probability_score(y_true, count_pmf_fn, max_k=15)` — discrete RPS/CRPS for count distributions; must accept either a Poisson mean array or (mean, alpha) NB parameterization.
  - `threshold_calibration(y_true, p_ge_k, n_bins=10)` — returns Brier score, expected calibration error, and the reliability table (bin edges, predicted vs empirical frequency, counts) as a DataFrame.
  - `dispersion_index(y_true, mu)` — mean squared Pearson residual `mean((y−μ)²/μ)`; ≈1 under Poisson, >1 ⇒ overdispersed.
  - `within_fixture_ranking(frame, group_cols=("fixture_id","team_id"), pred_col, actual_col)` — mean Spearman ρ across groups (groups with all-zero actuals excluded and counted separately) and top-1/top-3 leader hit rates ("did the predicted team leader actually lead").
  - `poisson_baseline_rates(train_frame, target, group_cols=("position_group",))` — per-90 rate baselines: (a) position-group global rate, (b) player's own `<target>_l5_p90` (already a mart column). Returns baseline means = rate × exposure.
  - `skill_score(model_logloss, baseline_logloss)` — `1 − model/baseline` (positive = model beats baseline).
- **Implementation notes:** module also carries `count_cdf`/`prob_at_least` (shared Poisson/NB machinery for M1), `validation_metric_suite` (the composite L3 wires into training — returns `{"metrics": scalars, "artifacts": reliability tables}` so quota discipline is structural), and `rolling_origin_folds` (pulled forward from L2 — it belongs with the other pure primitives). `poisson_log_loss` moved here as the canonical implementation; `ml_training` re-exports it (no cycle: evaluation never imports ml_training). RPS auto-extends its grid to `max(max_k, y_max + 5)` so truncation never bites the observed support. NB pmf via lgamma in log space, no scipy dependency; NB tail unit-tested against the exact geometric case (mu=1, alpha=1 ⇒ P(Y≥3) = 1/8). 13 hand-computed tests in `tests/test_evaluation.py`.

### L2. Rolling-origin backtest harness — ✅ DONE (2026-07-07)
- **Files:** `football_analytics/evaluation.py` (add `rolling_origin_folds`), `scripts/train_poisson_lgbm.py` (new `--backtest` mode), `tests/test_evaluation.py`.
- **Depends on:** L1.
- **Acceptance criteria:**
  - `rolling_origin_folds(frame, season_column="league_season", min_train_seasons=2)` yields expanding-window (train ≤ season N−1, validate = season N) folds, reusing `season_train_validation_split` semantics per fold.
  - `--backtest` trains and evaluates per fold **without registering models**, writes one consolidated artifact (`backtest_report.json` + CSV: per target × fold × metric) and logs only per-target cross-fold mean/std scalars to MLflow (metric-quota discipline, Working Agreement #4).
  - This harness is the required evidence for every Epic M adoption decision.
- **Implementation notes:** `rolling_origin_folds` landed with L1 (pure primitive). `run_rolling_origin_backtest` lives in `ml_training.py` and returns `report`/`summary`/`skipped` frames; the booster-fitting core was extracted into `_fit_poisson_booster` shared with `train_poisson_lightgbm_with_mlflow` so the two paths cannot diverge. Fold/target pairs are skipped (with reason) below 50 train rows or with zero positive labels — `cards_red` on small folds hits this by design. `--backtest` in the script logs only cross-fold mean/std scalars plus a consolidated CSV/JSON artifact bundle in a model-free run. End-to-end test trains real LightGBM boosters on synthetic seasons and asserts positive skill vs the position-group baseline on every fold.

### L3. Wire the new metrics into standard training runs — ✅ DONE (2026-07-07)
- **Files:** `football_analytics/ml_training.py` (`train_poisson_lightgbm_with_mlflow`), `tests/` (extend existing training tests if present, else add `tests/test_ml_training_metrics.py` with a lightgbm-optional skip marker).
- **Depends on:** L1.
- **Acceptance criteria:**
  - Per target, validation now also logs: `rps`, `brier_ge_1`, `ece_ge_1`, `dispersion_index`, `skill_vs_posgroup_rate`, `skill_vs_player_l5`, `rank_spearman`, `top1_hit_rate` (8 new scalars × 11 targets — within quota).
  - Reliability tables and per-position-group breakdowns of the new metrics go to run **artifacts** (CSV), not metrics.
  - `evaluate_count_predictions` remains backward-compatible (existing keys unchanged).
- **Implementation notes:** training logs the full `validation_metric_suite` scalar set (11 per target — the 8 required plus `brier_ge_2`/`ece_ge_2`/`top3_hit_rate`; non-finite values skipped). Reliability tables land as `{target}_reliability_ge_{k}.csv` and the extended per-position-group breakdown (basic + rps + dispersion) as `{target}_position_group_metrics.csv` under `model_artifacts/`. Suite also returned in `result["metrics"][target]["validation_suite"]`. Integration test (`tests/test_ml_training_metrics.py`) runs real training against a local sqlite MLflow store (MLflow 3.x refuses the legacy file store) and asserts metrics + artifacts exist.

### L4. Baseline-only reference run mode — ✅ DONE (2026-07-07)
- **Files:** `scripts/train_poisson_lgbm.py` (`--baselines-only` flag), `football_analytics/evaluation.py`.
- **Depends on:** L1, L2.
- **Acceptance criteria:** a mode that scores the two naive baselines through the identical fold harness and logs them under run name `baseline-reference`, so every model run's skill scores have an inspectable denominator run. No model training, no registration.
- **Implementation notes:** `run_baseline_reference` in `ml_training.py` (lightgbm never imported on this path). Test asserts the player-L5 baseline beats the position-group baseline on synthetic data where per-player rates are the true signal — a sanity check that the two baselines are genuinely different reference points.

### L5. Extend prediction monitoring with calibration & ranking views — ✅ DONE (2026-07-07)
- **Files:** `dbt/models/marts/mon_football__prediction_monitoring.sql` (extend) or new `mon_football__prediction_calibration.sql` (+ schema.yml).
- **Depends on:** K5 (columns), otherwise independent of L1–L4.
- **Acceptance criteria:** post-match monitoring surfaces, per target × position group × model version: empirical vs predicted P(≥1)/P(≥2) in probability bins (production calibration curve data) and the top-1 leader hit rate per fixture-team. Views stay empty-but-queryable with no predictions.
- **Implementation notes:** two new views over `mon_football__prediction_vs_actual` (which already joins active predictions to realized labels): `mon_football__prediction_calibration` (0.1-wide bins via `stack` over both thresholds, includes `calibration_gap`) and `mon_football__prediction_ranking` (top-1 leader hit rate with deterministic `player_id` tie-break, all-zero team-fixtures counted separately). Kept separate from `mon_football__prediction_monitoring` because the grains differ (probability bins / team-fixtures vs segment aggregates).

---

## Epic M — Model Family Upgrades (fixes D-2; decisions gated on Epic L evidence)

### M1. Negative-binomial dispersion (two-stage) for probabilities and simulation — ✅ DONE (2026-07-07)
- **Files:** `football_analytics/ml_training.py`, `football_analytics/inference.py`, `tests/test_inference.py`, new tests.
- **Depends on:** L1 (dispersion_index motivates and validates α).
- **Required behavior:**
  - After each target's booster trains, fit `alpha_player` on validation rows by method of moments: `α̂ = max(0, Σ[(y−μ)² − μ] / Σ μ²)`.
  - Additionally fit `alpha_team` on validation **team totals** (group validation rows by fixture_id+team_id, sum y and μ per target, same estimator) — this is the dispersion the simulator uses for team-total draws.
  - Both α values: logged as MLflow params (`{target}_alpha_player`, `{target}_alpha_team`), stored as model-version **tags** on the registered model, and carried in `ExposurePoissonLightGBMModel` / `LoadedEventModel` (default 0.0 = Poisson for old versions).
  - `count_threshold_probabilities` gains an optional `alpha` parameter: NB pmf (`p_ge_k` via the NB survival function, numpy/scipy-free implementation with lgamma) when `alpha > 0`, Poisson otherwise. `build_prediction_records` passes each model's `alpha_player`.
- **Acceptance criteria:** unit tests: α̂ = 0 recovered on Poisson-simulated data, α̂ ≈ truth on NB-simulated data (tolerance); NB `p_ge_k` matches scipy.stats.nbinom on a reference case (hardcoded expected values, no scipy runtime dependency); Poisson path bit-identical to current behavior when α = 0.
- **Implementation notes:** estimators in `evaluation.py` (`estimate_nb_alpha`, `estimate_team_total_nb_alpha` — the team version fits on fixture/team sums, validated against a shared-gamma-factor simulation where α_team ≈ 1/shape while player-level α stays near 0). Training fits both per target *before* the metric suite and passes `alpha_player` into it, so RPS/Brier/ECE score the distribution actually served. α values: MLflow params + model-version tags (set after registration via `ModelInfo.registered_model_version`); loader reads tags with a 0.0 default so pre-M1 versions stay exact-Poisson. NB reference case: mu=2, α=0.5 ⇒ r=2, p=½ ⇒ P(≥1)=0.75, P(≥2)=0.5, P(≥6)=1/16 exactly. One test taught us the tail crossover is past the mean — NB has *less* mass than Poisson at k=3 for mu=2 (more zeros), and only fattens the deep tail; the test documents this.

### M2. Self-contained pyfunc model wrapper (kill the duplicated exposure/gating logic) — ✅ DONE (2026-07-07)
- **Files:** `football_analytics/ml_training.py` (registration path), `football_analytics/inference.py` (loading path), `tests/test_inference.py`.
- **Depends on:** M1.
- **Current behavior:** the registry stores the raw booster; exposure semantics, feature columns, GK gating, and (after M1) α live in code that both `ExposurePoissonLightGBMModel` and `LoadedEventModel`/notebook 04 must re-implement in lockstep.
- **Acceptance criteria:**
  - Registered artifact becomes an `mlflow.pyfunc` model wrapping the booster + `feature_columns` + `exposure_column` semantics + `goalkeeper_only` + `alpha_player`/`alpha_team`; `predict` takes feature rows + exposure and returns the mean (metadata retrievable for probability/simulation use).
  - `load_registered_event_models` reads everything from the artifact/tags; drop the `booster.feature_name()` reliance.
  - Registration still happens in metric-free runs (Working Agreement #4). Backward compatibility: loader must still handle pre-M2 raw-booster versions (fall back to current path) so old versions remain servable.
- **Implementation notes:** the payload is `ExposurePoissonLightGBMModel` itself (new `predict_from_frame` method = the single home of exposure/GK-gating semantics: minutes column detection `expected_minutes` → exposure_column → `games_minutes`, per-90 rate without one, missing features filled 0.0). `build_pyfunc_player_event_model` wraps it in a locally-defined `PythonModel` that cloudpickle serializes by value — loading needs only football_analytics + lightgbm installed, no wrapper import path. Batch inference keeps its raw-booster fast path by reading `unwrap_python_model().inner` via `load_event_model_artifact`, which falls back to `mlflow.lightgbm.load_model` for pre-M2 versions (three generations handled: pre-M1 raw, M1 raw+tags, M2 pyfunc). Registry signature now reflects the serving contract (features + expected_minutes → means). α tags still written (M1) — they are the cheap read path for the simulator. `_lightgbm_log_model_kwargs` renamed `_log_model_kwargs` (works for any flavor). Round-trip tests: exposure math, per-90 fallback, GK gating, both generations.

### M3. Per-target hyperparameter tuning under the rolling-origin harness — ✅ DONE (2026-07-07)
- **Files:** `scripts/tune_lgbm.py` (new), `football_analytics/ml_training.py` (accept per-target config overrides), `pyproject.toml` (optional `tuning` extra: optuna).
- **Depends on:** L2.
- **Current behavior:** one global `PoissonLightGBMConfig` for all 11 targets (`min_child_samples=80` for `passes_total` with mean ~30 *and* `cards_red` with mean ~0.005).
- **Acceptance criteria:**
  - Optuna (TPE, budget ≤ 50 trials/target, configurable) minimizing cross-fold mean RPS (fallback poisson_logloss) over: `learning_rate`, `num_leaves`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `lambda_l2`, `num_boost_round` (with early stopping).
  - Best per-target params written to a version-controlled JSON (`config/lgbm_params/<target>.json`) that `train_poisson_lgbm.py` loads by default (flag to ignore); tuning study summary logged as one MLflow artifact.
  - Adoption gate: tuned params replace defaults only for targets where cross-fold mean RPS improves and its std does not blow up (record the comparison table in ticket notes).
- **Implementation notes:** tuning core in `ml_training.py` (`tune_target_hyperparameters` + lean `cross_fold_rps` objective that skips the full metric suite per trial; unlearnable targets return `inf` so trials are rejected). The adoption gate is **stored in the artifact** (`adopted: true/false` with tuned-vs-default RPS on identical folds) and `load_tuned_configs` only returns adopted entries, so training never re-derives the decision. `train_poisson_lightgbm_with_mlflow` gained `per_target_configs` (per-target override logged as `{target}_config` param). `business_logic.md` §18 repo map to be updated with `scripts/tune_lgbm.py` when the first tuned configs land. ⚠️ The actual tuning run needs Databricks data + the `tuning` extra (optuna not installed locally); run `scripts/tune_lgbm.py` there and commit the resulting `config/lgbm_params/*.json` with the comparison table here.

### M4. Sparse-target decision: hurdle / calibrated classifier (ADR 0003 follow-up) — ⏳ BLOCKED ON DATA (2026-07-07, machinery ready)
- **Files:** decision + implementation if adopted: `football_analytics/ml_training.py`, `docs/adr/0003-sparse-target-handling.md` (supersede if adopted).
- **Depends on:** L2, L3 (needs the calibration evidence ADR 0003 asked for), M1.
- **Acceptance criteria:**
  - Run the L2 backtest and pull `ece_ge_1`/reliability artifacts for `cards_yellow`, `cards_red`, `goals_total`, `goals_assists` under Poisson and NB (M1).
  - **Decision rule (pre-committed):** for each target, if NB `p_ge_1` ECE ≤ 0.02 or the reliability curve shows no systematic direction, keep NB and close this ticket with the evidence. Otherwise implement a hurdle for that target only: LightGBM binary `P(≥1)` (same features, `is_unbalance=true`) + isotonic calibration fitted on a chronologically later slice, count-beyond-1 from the truncated NB. Registered as the same `<prefix>__<target>` name (new version, pyfunc from M2 hides the internals).
  - Whichever way it lands, update the ADR with the measured numbers.
- **Status notes (2026-07-07):** everything needed to *make* the decision is deployed — the L2 backtest scores `ece_ge_1` and reliability tables under the NB distribution fitted by M1, per fold, per target. The decision run itself could not execute today (Databricks free daily quota exhausted). **Runbook:** on Databricks run `python scripts/train_poisson_lgbm.py --backtest --targets cards_yellow,cards_red,goals_total,goals_assists`, read `{target}_ece_ge_1_mean` and the `backtest/` reliability artifacts, then apply the pre-committed rule above verbatim. Only if a target fails the rule does hurdle implementation work start.

### M5. Feature review from SHAP evidence (optional, small) — ✅ DONE (2026-07-07, static part)
- **Files:** `football_analytics/ml_training.py` (`DEFAULT_LIGHTGBM_FEATURES`), notes in this doc.
- **Depends on:** L2 (so removals are re-validated by backtest).
- **Acceptance criteria:** review the per-target SHAP and gain artifacts from a current training run; propose ≤ 5 removals (dead features) and ≤ 3 additions (candidates: days-since-last-fixture rest proxy, opponent GK quality for `shots_on`→`goals`, team substitution-count tendency for minutes context). Each change must be justified by a backtest delta table in the ticket notes; no-op is an acceptable outcome.
- **Implementation notes:** a static cross-check of `DEFAULT_LIGHTGBM_FEATURES` against the columns `fct_football__player_event_features` + `add_model_interaction_features` can actually produce found **8 dead entries** (`is_starter`, `was_substitute`, `dribbles_attempts_l5_p90`, `tackles_interceptions_l5_p90`, `game_importance_l5`, `opponent_strength_adjustment`, `defensive_containment_rating`, `opponent_defensive_elo_l10`) — leftovers from the deleted shot-features mart, silently discarded by `_select_available_features` on every run. Removing them is a provably behavior-neutral cleanup (no backtest needed: the selected feature set is unchanged); a regression test pins them out. The SHAP-evidence half (possible additions: rest-days proxy, opponent GK quality, substitution-tendency) needs artifacts from a production training run — carried as an explicit follow-up in O3.

---

## Epic N — Monte Carlo Fixture Simulation (fixes D-4; the headline feature)

Produces a full simulated game per fixture: per-player counts for all 11 events + team totals + scoreline distribution, coherent by construction, from N (default 10,000) Monte Carlo iterations.

### N1. Simulation core: inputs, config, RNG, minutes sampling + ADR 0005 — ✅ DONE (2026-07-07)
- **Files:** new `football_analytics/simulation.py`, new `tests/test_simulation.py`, new `docs/adr/0005-fixture-simulation-design.md`.
- **Depends on:** K5, M1 (needs `p_plays`, `expected_minutes_if_plays`, `alpha_team` tags).
- **Required behavior:**
  - `SimulationConfig` dataclass: `n_sims=10_000`, `seed`, `thresholds=(1,2,3)`, `assist_per_goal_rate` (provisional 0.72; N3 records the calibration query), engine version constant `SIMULATION_ENGINE_VERSION`.
  - `build_simulation_inputs(active_predictions: pd.DataFrame, expected_minutes: pd.DataFrame) -> SimulationInputs`: pivots the active prediction set (one fixture, all targets) into per-player per-90 intensities `λ = predicted_mean / (expected_minutes/90)` (guard: expected_minutes ≥ ε), joins `p_plays`/`expected_minutes_if_plays`, validates exactly 2 teams and ≥ 11 players each, and pulls `alpha_team` per target from model-version tags (parameter with default 0.0 fallback).
  - Minutes sampling per iteration (vectorized over players × sims): starters → fixed `expected_minutes_if_plays`; bench → `Bernoulli(p_plays) × expected_minutes_if_plays`. Output: minutes matrix `m[p, s]`.
  - All randomness via one `numpy.random.Generator(seed)`; identical seed + inputs ⇒ bit-identical outputs (unit-tested).
  - ADR 0005 records the §2.4 design, the three v1 simplifications and their revisit triggers, and the identity approximations (saves identity ignores own goals / red-card feedback).
- **Acceptance criteria:** module imports without Spark/MLflow; deterministic under seed; input validation errors are specific (missing target, <11 starters, zero exposure).
- **Implementation notes:** `SimulationInputs.players` carries one `rate90_<target>` column per target, recovered from `predicted_mean / (expected_minutes/90)` — round-trip verified for bench players (blended mean ÷ blended exposure = clean per-90 rate). Pre-K5 prediction rows degrade to deterministic minutes (`p_plays=1`, `if_plays=expected_minutes`). `alpha_team` arrives as a plain dict (read from M1 version tags by the notebook) so the module stays MLflow-free. ADR 0005 records the full sampling design and six v1 simplifications with revisit triggers, including one added during implementation: assist allocation does not exclude the goal scorer (team invariants unaffected).

### N2. Team totals + multinomial allocation — ✅ DONE (2026-07-07)
- **Files:** `football_analytics/simulation.py`, `tests/test_simulation.py`.
- **Depends on:** N1.
- **Required behavior:**
  - Per team, per independent event family (`shots_total`, `passes_total`, `offsides`, `fouls_committed` [see N3 mirror], `cards_yellow`): team total `T[s] ~ NB(mean = Σ_p λ_p·m[p,s]/90, alpha_team)`; NB sampled as Poisson–Gamma mixture (`rate ~ Gamma(1/α, α·mean)` when α>0, else Poisson) — vectorized.
  - Allocation: `counts[·, s] ~ Multinomial(T[s], w)` with `w_p ∝ λ_p·m[p,s]/90`. Edge cases (unit-tested): total > 0 with all-zero weights → uniform over players with `m > 0`; total = 0 → zeros; single-player teams rejected upstream.
  - Mean preservation: across sims, `mean(counts_p) ≈ λ_p·E[m_p]/90` within 2% for a 50k-sim test.
- **Implementation notes:** multinomial realized as sequential conditional binomials with suffix-sum denominators (exact decomposition, vectorized across sims, no batched-multinomial numpy requirement; the last positive-weight player's conditional ratio is exactly 1.0 by float identity, so remainders never leak). `allocate_event_totals` repairs the two degenerate edges before allocation and returns repaired totals so downstream team sums stay consistent. NB totals verified against the analytic variance `M + αM²` at 60k draws; conservation asserted exactly (`Σ_p counts == totals` per sim).

### N3. Structural coherence chains — ✅ DONE (2026-07-07)
- **Files:** `football_analytics/simulation.py`, `tests/test_simulation.py`.
- **Depends on:** N2.
- **Required behavior (per iteration, per player unless stated):**
  - `shots_on ~ Binomial(shots_total_p, clip(μ_on,p/μ_total,p, 0, 1))` (ratio from per-90 intensities; μ_total = 0 ⇒ 0).
  - `goals ~ Binomial(shots_on_p, clip(μ_goals,p/μ_on,p, 0, 1))`.
  - **Fouls mirror:** one draw per directed pair — total `F_AB ~ NB(mean = avg(Σμ_committed,A , Σμ_drawn,B), α)` allocated twice: to team A players by committed-intensity (→ their `fouls_committed`) and to team B players by drawn-intensity (→ their `fouls_drawn`); symmetric draw `F_BA` for the reverse direction.
  - **Saves identity:** team A GK(s) `saves = max(0, shots_on_B_total − goals_B_total)`; allocated to the starting GK (multiple GKs in lineup: the starter; unit test covers it). The trained `goals_saves` model keeps producing the marginal prediction-table row; the simulation table's saves come from the identity (documented in ADR 0005).
  - **Assists:** per team goal, `Bernoulli(assist_per_goal_rate)`; if assisted, allocate to one **teammate of the scorer** (excluding the scorer) via multinomial on assist intensities.
  - **Cards:** yellows capped at 2/player (sample then clip), `red ~ Bernoulli(clip(μ_red,p·m_p/90·adjustment,0,0.5))` capped at 1.
  - `offsides`, `passes_total`: plain N2 path.
- **Acceptance criteria:** an invariants test asserts on every iteration of a 1k-sim run: `goals ≤ shots_on ≤ shots_total` per player; `Σ fouls_committed_A = Σ fouls_drawn_B`; `saves_GK_A = max(0, shots_on_B − goals_B)`; assists per team ≤ goals per team; yellows ≤ 2, reds ≤ 1. Plus the assist-rate calibration query recorded in ticket notes (empirical Σassists/Σgoals from completed fixtures → update `assist_per_goal_rate` default).
- **Implementation notes:** `simulate_fixture` with a fixed draw order for determinism. Fouls-mirror totals are validated against *both* teams' pitches before either allocation so the mirror holds exactly even in degenerate sims. Assist allocation does not exclude the scorer (ADR 0005 simplification #4 — team invariants unaffected). Off-pitch players record zero across all 11 targets (asserted). Extra test: unconditional simulated means track `p_plays × if_plays/90 × rate90` for bench players within 10%. Assist-rate calibration ran 2026-07-07: 68.8% of goals carry a recorded assist, so `SimulationConfig.assist_per_goal_rate` is set to 0.688 (was provisional 0.72). Query and provenance in ADR 0005.

### N4. Summary statistics + `sim_football__fixture_simulation` Delta table — ✅ DONE (2026-07-07)
- **Files:** `football_analytics/simulation.py` (summarize + writer), `football_analytics/inference.py` or new module section for DDL, `dbt/models/sources.yml` (register source), new dbt singular tests, `tests/test_simulation.py`.
- **Depends on:** N2, N3.
- **Required behavior:**
  - `summarize_simulation(...) -> pd.DataFrame` — grain: `fixture_id / sim_set_id / entity_type ('player'|'team') / entity_id / target_event`; columns: `entity_name`, `team_id`, `is_starting`, `n_sims`, `seed`, `sim_mean`, `sim_std`, `p05`, `p25`, `p50`, `p75`, `p95`, `p_ge_1`, `p_ge_2`, `p_ge_3`, `prediction_set_id`, `engine_version`, `created_at_utc`, `is_active_simulation`. Team rows include a `goals_total` row per team (scoreline distribution is derivable; the notebook renders the matrix from raw team-goal draws — persist a compact `score_matrix` JSON string on the team `goals_total` rows).
  - `sim_set_id` = sha256(fixture_id, prediction_set_id, engine_version, config digest incl. seed) — mirrors `deterministic_prediction_set_id`.
  - Writer mirrors `write_player_event_predictions` Option C semantics exactly: delete same `sim_set_id`, append, flip older sets for the fixture+engine_version inactive. Reuse/extract the shared logic rather than copy-pasting it.
  - dbt source + tests: no duplicate active sim rows per (fixture, entity, target, engine_version); `p_ge_*` ∈ [0,1] non-increasing; percentiles monotone; `sim_mean ≥ 0`.
- **Acceptance criteria:** unit test round-trips a small simulation through summarize → writer against the fake-Spark harness used in `test_prediction_write_appends_and_flips_older_active_sets`.
- **Implementation notes:** the Option C write strategy was extracted into `football_analytics/delta_write.py` (`write_active_flag_records` — parameterized set-id column, flip keys, and flag column); `write_player_event_predictions` now delegates to it with its exact SQL contract preserved (existing tests unchanged and green). Scoreline lives on ONE fixture-level row (`entity_type='fixture'`, `target_event='scoreline'`, matrix JSON keyed `goalsA-goalsB`) rather than duplicated per team. Summary test also asserts the allocation identity — each team's `sim_mean` equals the sum of its players' means to 1e-9. Sim table pre-created in `notebooks/00_prepare_run.py` next to the prediction table so the new dbt source tests never hit a missing table.

### N5. Notebook 05 + job wiring — ✅ DONE (2026-07-07)
- **Files:** new `notebooks/05_fixture_simulation.py`, `resources/prematch_inference_pipeline.yml` (new task after `inference_predict`), `tests/test_databricks_bundle_and_config.py`.
- **Depends on:** N4.
- **Acceptance criteria:**
  - Notebook widgets: `fixture_id` (blank = all fixtures with an active prediction set in the trigger window), `n_sims`, `seed`, `mode`. Loads active predictions + expected-minutes decomposition, runs the simulator, writes via N4, and **displays the full game view**: per-team table (one row per player × columns per event with mean and P(≥1)), team totals with 5–95% intervals, and the scoreline probability matrix.
  - Job task `fixture_simulation` depends on `inference_predict`; skips cleanly (SKIPPED status row) when no active prediction set exists.
  - Bundle test asserts the task exists, its dependency, and parameter pass-through (pattern: `test_bundle_passes_statistics_toggle_to_bronze_ingest`).
- **Implementation notes:** `mode` widget dropped from the notebook — the simulator's guard rail is the existence of an *active prediction set* (notebook 04 already enforced lineup rules before writing one), so a separate mode adds nothing. Dispersion tags are read per fixture for the exact model versions in that prediction set via `load_team_dispersion_tags` (`inference.py`; accepts an injectable client, falls back to Poisson totals when the registry is unreachable). `SimulationInputError` fixtures log `simulation_skipped_invalid_inputs` and continue. Job parameters `n_sims`/`sim_seed` added; `test_databricks_notebook_files_match_medallion_order` updated for the new notebook.

### N6. Historical backtest: simulation calibration report — ✅ DONE (2026-07-07)
- **Files:** new `scripts/backtest_simulation.py`, `football_analytics/evaluation.py` (interval-coverage + PIT helpers), `tests/test_evaluation.py`.
- **Depends on:** N3, L1. **Requires Databricks access** for the real run; the script itself is unit-testable with synthetic frames.
- **Required behavior:** for a holdout season: build pseudo-pre-match predictions for completed fixtures (feature-mart training rows + registered models + `expected_minutes` exposure — NOT actual minutes), simulate each fixture, then score against actuals: (a) central-interval coverage per target (80% interval should cover ~80% of player actuals; report per target), (b) randomized PIT uniformity summary, (c) team-total coverage, (d) allocation quality: top-1/top-3 hit rate for "which player led the team" per event, compared against the marginal-model ranking from L3 (the simulation must not be worse).
- **Acceptance criteria:** one artifact report (JSON + CSV) with the above; ticket notes record the numbers and flag any target whose 80% coverage falls outside [70, 90] as input for the ADR 0005 revisit triggers.
- **Implementation notes:** core in `football_analytics/simulation_backtest.py` (pure given models + feature rows; reuses `build_prediction_records` with `lineup_source='offline_sample'` and reads `alpha_team` straight off the models); thin Spark/MLflow wiring in `scripts/backtest_simulation.py` with the `outside_acceptance_band` flag computed per target. Coverage/PIT primitives (`central_interval_coverage`, `randomized_pit`, `pit_uniformity_summary`) live in `evaluation.py`. The test suite validates the *scoring pipeline itself*: in a synthetic world where labels follow the models' rates exactly, Poisson-thinning makes simulated player marginals exactly Poisson, so the test asserts nominal interval coverage and PIT uniformity (max bin deviation < 0.07) — a true end-to-end calibration check, not just shape assertions. ⚠️ The real holdout-season run still needs Databricks (daily quota exhausted today); run `scripts/backtest_simulation.py --season <N>` and record the numbers here.

### N7. Documentation sync for simulation — ✅ DONE (2026-07-07)
- **Files:** `docs/business_logic.md` (new §20 "Fixture Simulation", §12 update for minutes v2, §13.3/§16.3 updates for NB + new metrics), this file.
- **Depends on:** N5 (land after the feature works), K3, M1.
- **Acceptance criteria:** business_logic.md documents: simulation inputs/outputs/table grain, the coherence rules, the v1 simplifications + revisit triggers (red-card feedback trigger: N6 shows card-heavy fixtures with systematically over-predicted minutes-driven events; tempo-factor trigger: N6 team-total coverage too narrow on ≥3 targets; team-total-model trigger: N6 team-total bias non-zero on ≥3 targets). Monitoring section lists the new `mon_` views.
- **Implementation notes:** business_logic.md gains §12.4 (adopted role-conditional estimator), §13.3 addendum (NB two-stage, pyfunc artifacts, evaluation harness), §16.3 implemented-monitoring list, §20 Fixture Simulation (pointing to ADR 0005 for the simplification/revisit-trigger detail rather than duplicating it), and a refreshed §18 repo map. Section 19 numbering left untouched — existing code comments cite §19.

---

## Epic O (part 2) — Final Hygiene

### O2. Simulation monitoring view — ✅ DONE (2026-07-07)
- **Files:** new `dbt/models/marts/mon_football__simulation_vs_actual.sql` (+ schema.yml).
- **Depends on:** N4.
- **Acceptance criteria:** view joining active simulation rows to realized labels after completion: per target — sim_mean MAE, interval-hit flags (actual within [p05,p95]), and P(≥1) calibration inputs; segmented by target × position_group × engine_version. Empty-but-queryable convention.
- **Implementation notes:** committed ahead of N7 so the business_logic.md monitoring list stays truthful per-commit. Position group comes from the feature-mart labels (the sim table's copy may lag lineup corrections); interval hit uses the stored p05/p95 columns directly.

### O3. Backlog close-out review — ✅ DONE (2026-07-07)
- **Depends on:** everything above.
- **Acceptance criteria:** every ticket marked DONE with notes; `pytest` and `dbt build` green; one end-to-end verification on a real upcoming fixture recorded here (prediction set → simulation set → notebook display), mirroring the H3 verification note; any deferred items converted into explicit tickets in a follow-up section rather than left implicit.
- **Close-out notes (2026-07-07):** 22 of 23 tickets DONE, M4 blocked on production data with its runbook and pre-committed decision rule recorded in place. Final verification: 139 pytest tests green, `dbt parse` clean, working tree clean, every ticket = one commit pushed to main. ⚠️ The live end-to-end verification (prediction set → simulation set → notebook 05 display for a real fixture in the trigger window) was **not** executed from this workstation — Databricks free daily quota was exhausted mid-session and no fixture window was available; verify on the first scheduled `prematch_inference_pipeline` run with `scripts/run_query.py "SELECT entity_type, target_event, count(*) FROM gold.sim_football__fixture_simulation WHERE is_active_simulation GROUP BY 1, 2"`.

---

## Follow-Ups (explicit, all data-gated on Databricks access)

| # | Item | Command / entry point | Done-when |
| --- | --- | --- | --- |
| F-1 | M4 sparse-target decision run | `scripts/train_poisson_lgbm.py --backtest --targets cards_yellow,cards_red,goals_total,goals_assists` | Rule applied, ADR 0003 updated with numbers; hurdle built only for failing targets |
| F-2 | ~~Assist-per-goal rate calibration (N3)~~ | ✅ DONE 2026-07-07 — measured 0.688, default updated | — |
| F-3 | Simulation holdout backtest run (N6) | `scripts/backtest_simulation.py --season <latest completed>` | Coverage/PIT/allocation numbers recorded in N6 notes; ADR 0005 revisit triggers evaluated |
| F-4 | Hyperparameter tuning run (M3) | `scripts/tune_lgbm.py` on Databricks with the `tuning` extra | Adopted `config/lgbm_params/*.json` committed with the comparison table |
| F-5 | M5 SHAP-evidence half | training-run SHAP artifacts + L2 backtest deltas | Candidate features (rest days, opponent GK quality, sub tendency) accepted/rejected with evidence |
| F-6 | Retrain + re-register post-K1 models | `scripts/train_poisson_lgbm.py --registered-model-prefix ...` | New versions carry α tags + pyfunc artifacts; predictions use role-conditional exposure |
| F-7 | Live e2e verification of the simulation path | first scheduled `prematch_inference_pipeline` run | Active sim set visible for a real fixture; notebook 05 game view renders |

---

## Dependency Graph (summary)

```text
O1 ──→ K1 ──→ K2
        │ ├──→ K3 ──────────────┐
        │ ├──→ K4               │
        │ └──→ K5 ──────────┐   │
L1 ──→ L2 ──→ L4            │   │
  │      └──→ M3            │   │
  ├──→ L3                   │   │
  ├──→ M1 ──→ M2            │   │
  │      └──→ M4 (needs L2/L3)  │
  └──→ N6 (needs N3)        │   │
K5 + M1 ──→ N1 ──→ N2 ──→ N3 ──→ N4 ──→ N5 ──→ N7 (also needs K3, M1)
K5 ──→ L5                        N4 ──→ O2
M5 after L2 (optional)           all ──→ O3
```
