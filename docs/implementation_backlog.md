# Implementation Backlog — Player Event Prediction Build-Out

Derived from [`docs/business_logic.md`](business_logic.md), specifically the roadmap in Section 17. Organized as epics → tickets, with dependencies and acceptance criteria, in suggested execution order. No implementation has been done yet — this is planning only.

Legend: **Depends on** lists tickets/epics that must land first. **Files** are the primary touch points based on current repo layout.

---

## Epic A — Bronze: `/fixtures/statistics` Ingestion

Closes the gap called out in business_logic.md §6/§9.3: team-level match stats are not yet part of the permanent medallion flow.

**Depends on:** nothing (foundation for everything else).

### A1. Add `bronze.football_fixture_statistics_raw` table + schema — ✅ DONE (2026-07-06)
- **Files:** wherever bronze DDL/table defs live alongside `football_fixtures_raw`, `football_match_raw`, `football_lineups_raw` (check `football_analytics/databricks_ingestion.py` and any DDL notebooks/resources).
- **Acceptance criteria:**
  - Table stores raw JSON envelope, `fixture_id`, ingestion timestamp, `response_hash`, and run metadata, matching the shape of existing raw bronze tables.
  - No transformation of the payload beyond envelope metadata.
- **Implementation notes:**
  - `football_analytics/databricks_ingestion.py`: added `BRONZE_FIXTURE_STATISTICS_RAW_PATH`, `write_fixture_statistics_bronze()` (delegates to `write_bronze_raw_envelopes`, so the table gets the standard envelope columns: `run_id`, `fixture_id`, `source_endpoint`, `request_params`, `target_date`, `raw_payload`, `response_hash`, `api_status`, `ingested_at_utc`), and `_football_api_fixture_statistics_schema()` (payload `value` kept as string because the provider mixes ints/percent-strings/nulls — Silver casts).
  - `STATISTICS_ENDPOINT = "fixtures/statistics"` also landed here (needed by the write helper) — **A2 should NOT re-add it**; A2 only needs the ingestion-plan/checkpoint wiring and orchestration function.
  - `dbt/models/sources.yml`: registered `football_fixture_statistics_raw` as a bronze source.
  - Tests: `tests/test_databricks_ingestion.py` covers the path contract and envelope/run-metadata shape (`test_fixture_statistics_bronze_writes_raw_envelope_with_run_context`).

### A2. Add `STATISTICS_ENDPOINT = "fixtures/statistics"` and ingestion plan wiring — ✅ DONE (2026-07-06)
- **Files:** `football_analytics/databricks_ingestion.py` (add alongside `FIXTURES_ENDPOINT`, `PLAYER_STATS_ENDPOINT`, `LINEUPS_ENDPOINT`; extend `endpoint_ingestion_plan`, checkpoint marking/upsert helpers).
- **Depends on:** A1.
- **Acceptance criteria:**
  - `GET /fixtures/statistics?fixture=<fixture_id>` is called only for fixtures in a completed state (`FT`, `AET`, `PEN`), per business_logic.md §9.6.
  - Endpoint participates in the existing checkpoint skip/force-refresh/`response_hash` logic (§9.7) — no duplicate calls for already-`COMPLETED` checkpoint rows unless `force_refresh=true`.
  - Parallel fetch path (`_iter_fixture_endpoint_fetch_results` equivalent) reused rather than duplicated.
- **Implementation notes:**
  - `STATISTICS_ENDPOINT` constant already landed with A1.
  - Added `ingest_fixture_statistics_for_fixtures_to_bronze()` mirroring the player-stats function: reads completed checkpoint IDs for the statistics endpoint, builds `endpoint_ingestion_plan` (so `force_refresh` and completed-skip work identically), fetches via the shared `_iter_fixture_endpoint_fetch_results`, writes via `write_fixture_statistics_bronze`, and records `response_hash` on every checkpoint upsert. Empty payloads land in Bronze but checkpoint as `SKIPPED` so a later run retries them.
  - Added `fixture_statistics_payload_has_data()` and `fixture_ids_for_fixture_statistics()` (delegates to the completed-only player-stats filter — same `FT`/`AET`/`PEN` rule).
  - `ingest_senior_mens_international_bronze()` now takes `include_statistics=True` and `bronze_statistics_path` and calls statistics ingestion per date for completed fixtures only; `BronzeIngestionSummary` gained `statistics_ingested`/`statistics_skipped`.
  - Tests: completed-checkpoint skip, `force_refresh` refetch, completed-only selection, empty-payload-→-SKIPPED-with-hash, and orchestrator wiring (`tests/test_databricks_ingestion.py`). Note this already covers most of A4's matrix except an explicit hash-change-triggers-refresh test.
  - **For A3:** the notebook/job still needs to pass `bronze_statistics_path=table_name(config, "bronze", "football_fixture_statistics_raw")` (and optionally an `include_statistics` widget); until then the orchestrator defaults to the legacy `/mnt/...` path.

### A3. Wire statistics ingestion into the orchestration notebook/job — ✅ DONE (2026-07-06)
- **Files:** `notebooks/01_bronze_ingest.py`, `resources/international_medallion_pipeline.yml`.
- **Depends on:** A2.
- **Acceptance criteria:**
  - `bronze_ingest` job step calls the new endpoint plan alongside fixtures/players/lineups.
  - Job graph order (`prepare_run → bronze_ingest → dbt_deps → dbt_seed → dbt_build → dbt_python_models → dbt_build_python_dependents`) is unchanged.
- **Implementation notes:**
  - Notebook: added `include_statistics` widget (default `true`); both modes wired — fixture_id mode calls `ingest_fixture_statistics_for_fixtures_to_bronze` directly, date mode passes `include_statistics`/`bronze_statistics_path` to the orchestrator. Bronze target resolves via `table_name(config, "bronze", "football_fixture_statistics_raw")`.
  - Job yml: added `include_statistics` job parameter and base_parameter pass-through on `bronze_ingest`; task graph untouched.
  - Test: `test_bundle_passes_statistics_toggle_to_bronze_ingest` in `tests/test_databricks_bundle_and_config.py`.

### A4. Checkpoint/idempotency tests for the new endpoint — ✅ DONE (2026-07-06)
- **Depends on:** A2.
- **Acceptance criteria:**
  - Test proves a completed checkpoint row is skipped without `force_refresh`.
  - Test proves changed `response_hash` triggers a refresh.
  - Test proves fixtures not yet completed are excluded from the plan.
- **Implementation notes (all in `tests/test_databricks_ingestion.py`):**
  - `test_fixture_statistics_completed_checkpoint_rows_skip_api_calls` — completed checkpoint rows (read from the checkpoint table via `completed_checkpoint_fixture_ids` with `endpoint=fixtures/statistics`) suppress refetching without `force_refresh`.
  - `test_fixture_statistics_refresh_lands_changed_payload_and_new_hash` — interpretation note: the plan cannot detect a hash change without fetching, so per §9.7 the refresh trigger is `force_refresh`; the test proves a changed payload is re-landed in Bronze and the checkpoint `response_hash` is updated to the new hash. If true hash-diff-driven skip/write logic is wanted later, that's a new feature ticket, not test-only work.
  - `test_fixture_statistics_plan_excludes_uncompleted_fixtures` — only `FT`/`AET`/`PEN` fixtures enter the statistics plan (NS/HT/PST excluded).
  - Plus A2-era tests: explicit completed-ids skip, force-refresh refetch, empty-payload→SKIPPED-for-retry.

---

## Epic B — Silver: Team Match Statistics Model

**Depends on:** Epic A (needs raw statistics rows to type/flatten).

### B1. `stg_football__team_match_stats` model — ✅ DONE (2026-07-06)
- **Files:** new file in `dbt/models/staging/`, register in `dbt/models/staging/schema.yml`.
- **Implementation notes:**
  - Wide grain (one row per `fixture_id`/`team_id`); provider EAV `type`/`value` pairs pivoted via `max(case when ...)`. Real API-Football stat names used: `Ball Possession`, `Total Shots`, `Shots on Goal`, `Shots off Goal`, `Blocked Shots`, `Shots insidebox`, `Shots outsidebox`, `Total passes`, `Passes accurate`, `Passes %`, `Fouls`, `Corner Kicks`, `Offsides`, `Yellow Cards`, `Red Cards`, `Goalkeeper Saves`, `expected_goals`.
  - Percent strings (`"61%"`) cast to double 0–100; provider nulls preserved (provider conflates zero and no-coverage — gold decides null handling).
  - Dedup keeps latest ingested envelope: `qualify row_number() over (partition by fixture_id, team_id order by ingested_at_utc desc, updated_at_utc desc) = 1` — uses bronze `ingested_at_utc` as primary sort since `updated_at_utc` is a run timestamp that ties within a run.
  - `team_name_normalized` via shared `normalize_name` macro; rows filtered to fixtures present in `stg_football__fixtures`, matching siblings.
  - Tests: unique `(fixture_id, team_id)` + not-null + fixture relationship in `staging/schema.yml`; percent bounds via singular test `dbt/tests/assert_stg_football__team_match_stats_percentages_in_range.sql`. `dbt parse` validated locally.
- **Acceptance criteria:**
  - Grain: one wide row per `fixture_id` + `team_id` (per business_logic.md §10.3 preferred shape, not long/EAV).
  - Columns typed per §10.3 example list (`possession_pct`, `shots_total_team`, `shots_on_team`, `passes_total_team`, `passes_accurate_team`, `passes_pct_team`, `fouls_team`, `corners_team`, `offsides_team`, `yellow_cards_team`, `red_cards_team`), with exact names reconciled against the real API-Football payload field names.
  - Deduplicated by business grain + latest `updated_at_utc`, matching the pattern in `stg_football__lineups` / `stg_football__player_match_stats`.
  - Team names normalized consistent with existing staging models' join-resilience approach.
- **Tests:** unique `(fixture_id, team_id)`, not-null on core identifiers, accepted range for `possession_pct` (0–100).

---

## Epic C — Gold: Team Possession & Match-Flow Context

**Depends on:** Epic B.

### C1. `fct_football__team_match_stats_context` — ✅ DONE (2026-07-06)
- **Files:** new file in `dbt/models/marts/`, register in `dbt/models/marts/schema.yml`.
- **Implementation notes:**
  - Base is `fct_football__team_match_context` (so future/scheduled fixtures get rows with null current-match stats — needed for inference); silver stats left-joined twice (own + opponent-as-allowed).
  - All §11.5 rolling features implemented with the repo-standard leakage-safe window `rows between 5 preceding and 1 preceding` partitioned by team, ordered by `fixture_date_utc, fixture_id`; opponent rolling features come from a self-join on `(fixture_id, opponent_team_id)`.
  - `expected_possession_share` = team possession L5 / (team L5 + opponent L5), fallback 0.5 with insufficient history (this anticipates C2 option (b); C2 still owes the written decision + `elo_possession_interaction`).
  - `opponent_fouls_drawn_allowed_l5` is defined as the opponent's own committed-fouls L5 average (fouls drawn by a team = fouls committed by its opponent; team-level fouls-drawn isn't in the provider payload). Documented in schema.yml.
  - `team_passes_l5_p90_or_per_match` resolved as per-match: `team_passes_l5_per_match` (team-level per-90 is meaningless since teams always play ~90+).
- **Acceptance criteria:**
  - Grain: one row per fixture/team.
  - Joins `stg_football__team_match_stats` with `fct_football__team_match_context` (opponent, home/away, competition) to produce a fixture/team-perspective mart.
  - Adds rolling/historical features called out in §11.5: `team_possession_l5_avg`, `opponent_possession_l5_avg`, `expected_possession_share`, `team_passes_l5_p90_or_per_match`, `opponent_passes_allowed_l5`, `team_shots_l5`, `opponent_shots_allowed_l5`, `team_fouls_l5`, `opponent_fouls_drawn_allowed_l5`.
  - All rolling windows use only fixtures strictly prior to the current fixture (leakage check — see D-series and Epic F leakage rules in §13.4).

### C2. Decide + document expected-possession approach — ✅ DONE (2026-07-06)
- **Depends on:** C1 (needs raw possession history to experiment with).
- **Decision:** Option (b) — engineered features from Elo + team stats. Recorded in `docs/adr/0001-expected-possession-approach.md`. Implemented in `fct_football__team_match_stats_context`: `elo_expected_score`, `elo_possession_interaction` (= elo_expected_score × expected_possession_share), and `formation_possession_profile` (team's L10 possession with the same formation, nullable). Note: the Elo join moves this mart downstream of the python Elo models (built in `dbt_build_python_dependents` phase). Option (a) explicit model deferred until Epic F feature importances justify it.
- **Acceptance criteria:**
  - Written decision (in this backlog's follow-up or a short ADR) on whether expected possession is: (a) an explicit intermediate model, (b) an engineered feature from Elo + team stats, or (c) left implicit for downstream models — per the open question in §11.5.
  - Chosen approach implemented as `elo_possession_interaction` and/or `formation_possession_profile` feature(s) in C1 or a follow-up column set.
  - This is a decision ticket — do not block C1/C3 on it; C1 can ship with possession features and this ticket resolves the "explicit model" question afterward.

### C3. Tests for team match-flow mart — ✅ DONE (2026-07-06)
- **Depends on:** C1.
- **Acceptance criteria:** unique `(fixture_id, team_id)`, sane bounds for possession/percent columns, not-null on rolling features once sufficient history exists (or explicit null-handling documented for early-season/insufficient-history rows).
- **Implementation notes:**
  - schema.yml: unique `(fixture_id, team_id)`, not-null on `fixture_id`/`team_id`/`opponent_team_id`/`home_away`, fixture relationship, accepted home/away values. Null-handling rule documented in the model description (L5 nulls only with zero prior covered fixtures).
  - Singular tests: `assert_fct_football__team_match_stats_context_bounds_sane.sql` (percent columns 0–100; share/Elo-score/interaction 0–1) and `assert_fct_football__team_match_stats_context_rolling_populated.sql` (any row with ≥1 prior covered fixture must have a non-null rolling average).

---

## Epic D — Gold: General Player Event Feature Mart

**Depends on:** Epic C (needs team flow context), plus existing Elo/lineup/formation marts (already implemented).

### D1. Design `fct_football__player_event_features` schema — ✅ DONE (2026-07-06)
- **Files:** design doc/PR description referencing `dbt/models/marts/fct_football__player_shot_features.sql`, `fct_football__player_match_features.sql`, `fct_football__lineup_elo_strength.sql`, `fct_football__formation_matchup_history.sql`, `fct_football__player_elo_history.py`.
- **Implementation notes:** design recorded in `docs/design/player_event_features_schema.md` — full column inventory for all §11.4 families + 11 labels, training/inference row population rules (`is_completed_fixture` flag), ranked-join rolling-history approach, and the explicit decision to **fold in** `fct_football__player_shot_features` (deprecated now, deleted in J1 after Epic F repoints training).
- **Acceptance criteria:**
  - Full column inventory covering all four feature families in §11.4: fixture context, team/opponent context, lineup context, player history, team flow context, and target labels for all 11 events in §2.
  - Explicit decision recorded on retaining `fct_football__player_shot_features` standalone vs. folding it in (§11.3 gives both options — pick one and document why).

### D2. Implement `fct_football__player_event_features` — ✅ DONE (2026-07-06)
- **Files:** new file in `dbt/models/marts/`, updates to `dbt/models/marts/schema.yml`.
- **Depends on:** D1, C1, existing Elo/lineup/formation marts.
- **Implementation notes:**
  - Training rows from `fct_football__player_match_features` (played, labels populated, `is_completed_fixture=true`); inference rows from lineups × team context for non-completed fixtures (null labels). Unused subs in completed fixtures excluded by construction (existing training convention).
  - Player history via ranked join to last 5 played appearances strictly prior (date, fixture_id tiebreak): `appearances_l5_count`, `minutes_l5`, per-event `*_l5_count` and `*_l5_p90` for all 11 targets.
  - Joins: team Elo (`_pre` only), player Elo (`_pre`), lineup XI strength, formation history (`_pre`), C1 team flow columns, E1 expected minutes. Coalesce defaults mirror `fct_football__player_shot_features`.
  - `exposure` (actual, training) vs `expected_exposure` (inference) both present per §13.3.
  - `fct_football__player_shot_features` marked DEPRECATED in schema.yml; deletion in J1 after Epic F repoints training (per D1 decision).
- **Acceptance criteria:**
  - Grain: one row per fixture/team/player.
  - Includes all target event labels from §2 for completed fixtures (for training) and null labels for future/scheduled fixtures (for inference).
  - Only pre-match / lagged features used as predictors; all Elo values are `_pre` variants (§13.4 leakage rule).
  - Rolling player history features (last-N rates and counts, minutes history, missed-fixture count) explicitly exclude the current fixture from their own window.
  - If D1 chose to fold in `fct_football__player_shot_features`, that model is deprecated/removed in this ticket or a fast-follow; if retained standalone, this mart supersedes it as the general-purpose source for training.

### D3. Leakage regression tests — ✅ DONE (2026-07-06)
- **Depends on:** D2.
- **Acceptance criteria:** automated check (dbt test or Python test) that rolling window features for a given fixture never reference stats from that same fixture; test fails loudly if a future column accidentally includes current-match data.
- **Implementation notes:** dbt unit test `player_event_features_rolling_windows_exclude_current_fixture` (`fct_football__player_event_features_unit_tests.yml`) mocks all nine refs; player 7 has 2 shots in fixtures 1–4 and an extreme 50 in fixture 5 — fixture 5's `shots_total_l5_count` must be 8, and every fixture's window is asserted against the strictly-prior progression (0/2/4/6/8). Runs automatically in `dbt build` before the model materializes.

---

## Epic E — Gold: Expected Minutes

**Depends on:** existing player match-stats history (already implemented); can proceed in parallel with Epic D, but D2 should consume its output, so sequence E before or alongside D2.

### E1. Implement `fct_football__player_expected_minutes` — ✅ DONE (2026-07-06)
- **Files:** new file in `dbt/models/marts/`.
- **Implementation notes:**
  - Grain: one row per fixture/team/player for everyone named in the lineup (starters **and** substitutes — subs get estimates too since they're valid prediction targets; filter on `is_starting` if only XI is wanted).
  - Estimator: trimmed mean over last 5 played appearances (drop single min + max when ≥3 observations, else plain average, else role prior: starter 85 / substitute 15). Recorded per row in `expected_minutes_method`. Clamped to [0, 120].
  - "Played appearance" = `games_minutes` 1..130, matching `fct_football__player_match_features`; history joins strictly prior fixtures (date, then fixture_id tiebreak) so the current fixture never feeds its own estimate.
  - Pure SQL model (no python deps) — builds in the `dbt_build` phase.
- **Acceptance criteria:**
  - Grain: one row per fixture/team/player for confirmed-lineup players.
  - Implements the recommended first estimator from §12.3: `expected_minutes_l5_trimmed_or_median` (trimmed mean or median over last 5 appearances).
  - Value is computable pre-match (uses only historical `games_minutes`, no current-fixture data).
  - Handles small-sample edge cases explicitly (fewer than 5 prior appearances): document fallback (e.g., position-specific prior or plain average) rather than silently failing.

### E2. Document the chosen estimator and open alternatives — ✅ DONE (2026-07-06)
- **Depends on:** E1.
- **Acceptance criteria:** short note (in schema.yml description or an ADR) stating which estimator was chosen from the §12.2 candidate list and why, with the remaining candidates (weighted recency decay, separate expected-minutes model, starter/sub priors) marked as future work.
- **Implementation notes:** `docs/adr/0002-expected-minutes-estimator.md` records the L5 trimmed-mean choice, the plain-average and role-prior fallbacks, and the three deferred candidates; schema.yml description links to it.

### E3. Tests for expected minutes — ✅ DONE (2026-07-06)
- **Depends on:** E1.
- **Acceptance criteria:** not-null for all confirmed starters, sane bounds (0–120), regression test that a player with a red-card-shortened appearance in history doesn't produce a wildly skewed estimate (validates trimming/median behavior).
- **Implementation notes:**
  - schema.yml: unique `(fixture_id, team_id, player_id)`, not-null on keys/`is_starting`/`expected_minutes` (covers starters — the estimator always produces a value), accepted values for `position_group` and `expected_minutes_method`.
  - `dbt/tests/assert_fct_football__player_expected_minutes_bounds_sane.sql` — null or out-of-[0,120] rows fail.
  - Red-card regression: **dbt unit test** (`fct_football__player_expected_minutes_unit_tests.yml`) with mocked staging inputs — a 25-minute red-card appearance among four 90s must yield 90 (trimmed), and no-history players get role priors 85/15. Runs automatically in `dbt build` (dbt ≥1.8 unit tests).

---

## Epic F — ML Training: Expand Targets & Model Registration

**Depends on:** Epic D (feature mart with all target labels), Epic E (expected minutes as exposure offset).

### F1. Expand target column list in training code
- **Files:** `football_analytics/ml_training.py`, `scripts/train_poisson_lgbm.py`.
- **Acceptance criteria:**
  - Default target list expanded from `shots_total, fouls_committed, dribbles_attempts` to all 11 events in §2: `offsides, shots_total, shots_on, goals_total, goals_assists, goals_saves, passes_total, fouls_drawn, fouls_committed, cards_yellow, cards_red`.
  - Each target trains independently (one Poisson LightGBM model per target, per §13.3 baseline) against `fct_football__player_event_features`.
  - Exposure offset uses `expected_minutes` (inference-time) or actual `games_minutes` (training-time), per §13.3.
  - Train/validation split is chronological, not random (§13.4, §19).

### F2. Per-target model family adjustments for sparse/special targets
- **Depends on:** F1.
- **Acceptance criteria:**
  - `goals_saves`: restricted to goalkeeper position group; non-goalkeepers produce structural zero rather than being modeled (§2, §13.3).
  - `cards_yellow`, `cards_red`, `goals_total`, `goals_assists`: documented calibration/handling approach for sparsity (e.g., classification + calibration, or accept Poisson baseline for v1 with a follow-up ticket) — per §2 notes and §13.3 "potential improvements."
  - Decision recorded on whether position-group-specific models are used for v1 or deferred.

### F3. MLflow tracking and registration conventions
- **Files:** `football_analytics/ml_training.py`, `scripts/train_poisson_lgbm.py`.
- **Depends on:** F1.
- **Acceptance criteria:**
  - Every trained model is logged to MLflow with a naming convention that maps 1:1 to `target_event` (e.g., `player_event__<target>`).
  - Model version and registry URI use the existing Unity Catalog registry configuration (per recent commit "Configure MLflow to explicitly use Unity Catalog registry URI").
  - Run metadata captures feature table name/version used for training, to support the `feature_table_name`/`feature_table_version` columns needed later in Epic H.

### F4. Chronological evaluation harness
- **Depends on:** F1.
- **Acceptance criteria:** evaluation always splits/reports by time order (e.g., train on seasons N-k..N-1, validate on season N); metrics logged per target and, if F2 adopts them, per position group.

---

## Epic G — Pre-Match Inference Job

**Depends on:** Epic F (trained + registered models), Epic E (expected minutes), Epic D (feature mart), existing lineup ingestion.

### G1. Inference orchestration script/notebook
- **Files:** new script under `football_analytics/` (e.g., inference module) or new notebook alongside `notebooks/00_prepare_run.py`/`01_bronze_ingest.py`/`02_dbt_python_models.py`.
- **Acceptance criteria:** implements the 10-step flow in §14.3 — fetch/refresh fixture metadata, fetch confirmed lineups, write to Bronze, run required dbt Silver/Gold transformations, build inference rows for confirmed players, estimate expected minutes, load MLflow model(s), generate predictions per player/target, store predictions, optionally display in notebook.

### G2. Missing-lineup guard rails
- **Depends on:** G1.
- **Acceptance criteria:**
  - Live/production mode: run is skipped or fails with a clear, loggable reason when either team's lineup is not confirmed (§9.6, §14.4).
  - Exploratory/offline mode: supports mock/projected lineups but tags output as non-official (`lineup_source != confirmed_api`).

### G3. Trigger window scheduling
- **Depends on:** G1.
- **Acceptance criteria:** job/schedule polls or runs frequently enough to catch the T-60 to T-40 minute window (§14.1); document the chosen polling cadence and its trade-offs (missed window vs. redundant runs).

---

## Epic H — Prediction Output Table

**Depends on:** Epic G (needs an inference job to write into it); schema can be drafted in parallel with Epic F/G.

### H1. Create `gold.pred_football__player_event_predictions`
- **Files:** new dbt model or Python-managed Delta table under `dbt/models/marts/`.
- **Acceptance criteria:**
  - Grain matches §15.2: `fixture_id, team_id, player_id, target_event, model_name, model_version, prediction_set_id`.
  - Full column set from §15.3 present (`prediction_set_id`, `prediction_run_id`, fixture/team/opponent/player identity, `position_group`, `is_starting`, `formation`, `target_event`, `expected_minutes`, `predicted_mean`, `predicted_p_ge_1/2/3`, model metadata, `feature_table_name`, `feature_table_version`, `lineup_source`, `is_active_prediction`).

### H2. Dedup/write strategy: append + active flag
- **Depends on:** H1.
- **Acceptance criteria:**
  - Implements the recommended default (§15.4 Option C): append every run for audit history, deterministic `prediction_set_id` per fixture inference run, older rows for the same fixture/model/version flipped to `is_active_prediction=false` when a new active set lands.
  - No duplicate active rows for the same `(fixture_id, team_id, player_id, target_event, model_name, model_version)`.

### H3. Wire G1 inference job to write into H1
- **Depends on:** G1, H2.
- **Acceptance criteria:** end-to-end run for a real upcoming fixture produces exactly one active prediction row per confirmed player/target, visible via `scripts/run_query.py` or notebook display.

---

## Epic I — Data Quality & Governance for Predictions

**Depends on:** Epic H.

### I1. Prediction table tests
- **Files:** `dbt/models/marts/schema.yml` (or wherever prediction table tests are declared).
- **Acceptance criteria:** implements the tests listed in §16.2 specific to predictions:
  - no duplicate active predictions by fixture/player/target/model/version,
  - complete prediction coverage for confirmed starting XI players,
  - accepted values for `position_group` and boolean flags,
  - sane `predicted_mean` / probability bounds (`predicted_p_ge_*` ∈ [0,1] and non-increasing across ge_1/ge_2/ge_3).

### I2. Coverage + monitoring groundwork
- **Depends on:** I1.
- **Acceptance criteria:** at minimum, a queryable check (view or scheduled query) surfacing the §16.3 monitoring list — model metrics by target/position group, calibration for sparse events, prediction drift by competition, actual-vs-predicted after match completion, lineup coverage, missing expected-minutes count, active model version. Full dashboarding can be a follow-up; this ticket just needs the underlying queries/views to exist.

---

## Epic J — Cleanup

**Depends on:** everything above landing (do last, per roadmap item 11 — "remove all unnecessary code... that are not used").

### J1. Audit and remove dead code/models
- **Files:** candidates to check once Epic D lands: `dbt/models/marts/fct_football__player_shot_features.sql` (if folded into D2), any now-superseded logic in `football_analytics/modeling.py`, `football_analytics/orchestrator.py`, `football_analytics/reporting.py`, `run_pipeline.py`.
- **Acceptance criteria:** each removal is justified by a grep/reference check showing no remaining callers; dbt `--exclude`/graph confirms no orphaned dependents.

---

## Suggested Execution Order

1. **A1–A4** — Bronze statistics ingestion (unblocks everything downstream).
2. **B1** — Silver team match stats.
3. **C1–C3** — Gold team match-flow context (C2 decision can trail).
4. **E1–E3** — Expected minutes (independent of D, needed by F).
5. **D1–D3** — General player event feature mart (the big one; needs C1 + existing Elo/lineup/formation marts).
6. **F1–F4** — ML training expansion + MLflow registration (needs D + E).
7. **G1–G3** — Pre-match inference job (needs F).
8. **H1–H3** — Prediction output table + write path (schema H1 can be drafted alongside F/G, but H3 needs G1).
9. **I1–I2** — DQ tests and monitoring groundwork.
10. **J1** — Cleanup pass.

This order matches the roadmap in business_logic.md §17 but resequences E (expected minutes) earlier and in parallel with C, since D2 (feature mart) needs both C1 and E1 as direct inputs and they don't depend on each other.
