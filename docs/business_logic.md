# Football Analytics - Business Logic and AI-Agent Build Guide

## 1. Purpose

This document defines the business goal, data logic, medallion architecture, and prediction target for the `football-analytics` repository.

It is intentionally written as an AI-agent-friendly project guide. Future agents should use it as the source of truth when extending ingestion, dbt models, ML training, inference, and prediction serving.

The central product goal is:

> Build the best possible player-level pre-match predictions for a defined set of football events, using confirmed lineups, historical player/team context, tactical context, and trained ML models.

Predictions should be generated roughly **60 to 40 minutes before kickoff**, once confirmed lineups are available.

## 2. Prediction Objective

For every eligible fixture and every confirmed lineup player, the system should predict the following player events:

| Target event | Source/stat column | Prediction type | Notes |
| --- | --- | --- | --- |
| Offsides | `offsides` | Count | Mostly relevant to attacking players, but keep player-level coverage broad. |
| Shots total | `shots_total` | Count | Core player prop target. |
| Shots on target | `shots_on` | Count | Related to shots total, finishing quality, and opponent defensive profile. |
| Goals total | `goals_total` | Count/probability | Sparse target; model may need special handling or calibration. |
| Assists | `goals_assists` | Count/probability | Sparse target; strongly context dependent. |
| Saves | `goals_saves` | Count | Goalkeeper-specific target; non-goalkeepers should usually be structural zero. |
| Passes total | `passes_total` | Count | Strongly affected by possession, role, formation, and team strength. |
| Fouls drawn | `fouls_drawn` | Count | Player style, opponent pressure, and role dependent. |
| Fouls committed | `fouls_committed` | Count | Defensive role and match state dependent. |
| Yellow cards | `cards_yellow` | Count/probability | Sparse disciplinary target; may need classification-style calibration. |
| Red cards | `cards_red` | Count/probability | Extremely sparse; likely requires structural priors and careful evaluation. |

The initial modeling approach may train independent count models per target, but future work may use multi-output models, hierarchical models, or target-specific model families where that improves performance.

## 3. Desired Prediction Experience

The intended production workflow is:

1. A fixture is known in advance - run ingests the fixtures for the upcoming week (e.g. 7 days in advance) using the `lookahead_days` parameter.
2. Roughly **60 to 40 minutes before kickoff**, confirmed lineups become available through API-Football.
3. The pipeline loads the confirmed lineups into Bronze and refreshes the medallion tables.
4. The inference job builds one feature row for every confirmed player in the fixture.
5. A trained model, or one model per event, predicts each target event for every player.
6. Predictions are either:
   - displayed directly in a notebook cell for exploratory use, or
   - written into a dedicated prediction table for downstream consumption.

The prediction output should be easy to compare, rerun, audit, and replace without creating duplicate active records for the same fixture/model version.

## 4. Success Criteria

The project succeeds when it can reliably:

- ingest historical and upcoming senior men's international fixtures,
- ingest completed match statistics and confirmed lineups,
- enrich player rows with team, opponent, formation, lineup, Elo, possession, and recent-form context,
- estimate each starting player's expected minutes,
- train and evaluate models for all target events,
- run pre-match inference from confirmed lineups,
- store one active prediction set per fixture/model/version strategy,
- make predictions reproducible through clear model metadata, run IDs, and feature lineage.

## 5. Scope

### In Scope

- Senior men's national-team fixtures.
- API-Football fixture, lineup, player statistics, and team match statistics endpoints.
- Databricks medallion architecture.
- dbt Silver and Gold transformations.
- MLflow-tracked model training and inference.
- Player-level predictions for the target events listed above.
- Fixture-level prediction sets written to a governed table.

### Out of Scope For Now

- Club football.
- Women's, youth, and Olympic competitions.
- Real-time in-play prediction updates.
- Betting execution or odds integration.
- Full online serving through Databricks Model Serving. This can be added later.

## 6. Current Repository Reality

The repo already implements a large part of the medallion foundation:

- Databricks Asset Bundle job orchestration.
- Bronze ingestion for:
  - `/fixtures`,
  - `/fixtures/players`,
  - `/fixtures/lineups`.
- dbt Silver models for normalized fixtures, player match stats, and lineups.
- dbt Gold models for player features, team context, Elo history, lineup strength, formation history, and player shot features.
- Local matchup simulation and PDF/Parquet output.
- Poisson LightGBM training helpers for count models.

Important gap:

> `/fixtures/statistics` is not yet part of the permanent medallion flow, but it should be added because team-level match statistics such as possession are critical for predicting passes, shots, fouls, saves, and overall match flow.

## 7. Source Systems

| Source | Endpoint/table | Business purpose | Current status |
| --- | --- | --- | --- |
| API-Football fixtures | `https://v3.football.api-sports.io/fixtures` | Fixture discovery, kickoff time, teams, status, score, competition | Implemented |
| API-Football player statistics | `https://v3.football.api-sports.io/fixtures/players` | Player event outcomes after completed fixtures | Implemented |
| API-Football lineups | `https://v3.football.api-sports.io/fixtures/lineups` | Confirmed starting XI, substitutes, formation, player positions | Implemented |
| API-Football fixture statistics | `https://v3.football.api-sports.io/fixtures/statistics` | Team-level match stats, including possession-related context | Required extension |
| FIFA ranking seed | `dbt/seeds/fifa_mens_world_ranking_december_2022.csv` | Initial national-team strength baseline | Implemented |
| League allowlist seed | `dbt/seeds/senior_mens_international_leagues.csv` | Scope control for senior men's international competitions | Implemented |

## 8. Medallion Architecture Overview

The pipeline follows this shape:

```text
API-Football
  -> Bronze raw payload tables
  -> Silver typed and deduplicated tables
  -> Gold feature and modeling marts
  -> ML training and MLflow registry
  -> Pre-match inference
  -> Prediction output table and/or notebook display
```

Databricks job order:

1. `prepare_run`
2. `bronze_ingest`
3. `dbt_deps`
4. `dbt_seed`
5. `dbt_build`
6. `dbt_python_models`
7. `dbt_build_python_dependents`

## 9. Bronze Ingestion

### 9.1 Goal

Bronze should persist raw provider responses with enough metadata to support replay, audit, deduplication, and late-arriving data.

Bronze should not over-transform provider payloads. It should preserve raw JSON and technical metadata.

### 9.2 Existing Bronze Tables

| Table | Contents |
| --- | --- |
| `bronze.football_fixtures_raw` | Raw `/fixtures` response envelopes. |
| `bronze.football_fixture_eligibility` | Business eligibility decisions for discovered fixtures. |
| `bronze.football_match_raw` | Raw `/fixtures/players` response envelopes. |
| `bronze.football_lineups_raw` | Raw `/fixtures/lineups` response envelopes. |
| `bronze.ingestion_state_checkpoint` | Endpoint status, attempts, errors, hashes, and timestamps. |

### 9.3 Required Bronze Extension

Add:

| Table | Contents |
| --- | --- |
| `bronze.football_fixture_statistics_raw` | Raw `/fixtures/statistics` response envelopes. |

Recommended endpoint identifier:

```text
fixtures/statistics
```

Recommended request:

```text
GET /fixtures/statistics?fixture=<fixture_id>
```

### 9.4 Fixture Discovery

For each requested date:

```text
GET /fixtures?date=<YYYY-MM-DD>&timezone=UTC
```

Fixture eligibility rules:

- fixture must belong to a reviewed senior men's national-team competition,
- league ID must exist in `senior_mens_international_leagues`,
- competition label must not indicate club, women's, youth, or Olympic football.

### 9.5 Match Lifecycle

| Business state | Meaning |
| --- | --- |
| `SCHEDULED` | Fixture exists, but lineups and completed stats are not available. |
| `LINEUPS_CONFIRMED` | Both teams have confirmed starting XI data. |
| `LIVE` | Match is in progress, suspended, interrupted, or completed but not fully populated with player stats. |
| `COMPLETED` | Match has terminal completed status and player statistics are available. |

Completed sport statuses:

```text
FT, AET, PEN
```

Terminal unplayed statuses:

```text
PST, CANC, ABD, AWD, WO
```

### 9.6 Endpoint Loading Rules

**Fixtures**

- Load for the latest run start date to current date + lookahead days range or every selected date/range
- Required for all downstream context.

**Player statistics**

- Load for completed fixtures only: `FT`, `AET`, `PEN`.
- Source endpoint:

```text
GET /fixtures/players?fixture=<fixture_id>
```

- These rows become historical labels for target events.

**Lineups**

- Load for fixtures that are not terminal unplayed.
- Source endpoint:

```text
GET /fixtures/lineups?fixture=<fixture_id>
```

- For pre-match inference, missing or incomplete confirmed lineups should block prediction generation unless the run is explicitly marked as exploratory/offline.

**Fixture statistics**

- Should be loaded for completed fixtures.
- Source endpoint:

```text
GET /fixtures/statistics?fixture=<fixture_id>
```

- These payloads should provide team-level match context such as possession, team shots, corners, fouls, passes, and other flow metrics when available.

### 9.7 Idempotency And Refresh

The ingestion checkpoint table should remain the central control point.

Recommended behavior:

- Skip endpoint calls already marked `COMPLETED`, unless `force_refresh=true`.
- Use `response_hash` to detect changed payloads.
- Allow late data refresh because fixture data evolves:
  - scheduled fixture,
  - fixture with lineups,
  - completed fixture without full stats,
  - completed fixture with full player and team statistics.

## 10. Silver Transformations

### 10.1 Goal

Silver converts raw JSON payloads into typed, deduplicated, relational tables.

Silver should be close to source semantics. It should not do advanced feature engineering.

### 10.2 Existing Silver Models

| Model | Grain | Purpose |
| --- | --- | --- |
| `stg_football__fixtures` | One row per fixture | Typed fixture metadata, teams, score, status, competition. |
| `stg_football__player_match_stats` | One row per fixture/team/player | Typed player event outcomes and match participation data. |
| `stg_football__lineups` | One row per fixture/team/player | Typed lineup data, starters, substitutes, formation, grid position. |

### 10.3 Required Silver Extension

Add:

| Model | Grain | Purpose |
| --- | --- | --- |
| `stg_football__team_match_stats` | One row per fixture/team/statistic or one wide row per fixture/team | Typed team-level statistics from `/fixtures/statistics`. |

Preferred output for ML is a wide table with one row per fixture/team. Example columns:

- `fixture_id`
- `team_id`
- `team_name`
- `possession_pct`
- `shots_total_team`
- `shots_on_team`
- `passes_total_team`
- `passes_accurate_team`
- `passes_pct_team`
- `fouls_team`
- `corners_team`
- `offsides_team`
- `yellow_cards_team`
- `red_cards_team`

Exact column names should follow the actual API payload.

### 10.4 Silver Rules

- Preserve provider IDs as canonical keys.
- Deduplicate by business grain and latest `updated_at_utc`.
- Normalize player and team names for resilient joins.
- Keep fixture/player/team relationships explicit.
- Keep target event labels in player stats.
- Keep team match statistics separate from player outcomes until Gold.

## 11. Gold Transformations

### 11.1 Goal

Gold should produce model-ready, leakage-safe feature tables and curated analytical facts.

Gold answers:

- Who is playing?
- How strong are the teams?
- How strong is the team's defense?
- How strong is the team's offense?
- How strong are the players?
- What tactical shape is expected?
- What has the player done recently?
- What has the team done recently?
- What type of match flow should the model expect (possession, shots, etc.)?
- What player event labels are available for training?
- How confirmed lineup affect the prediction?
- How does the formation affect the prediction?

### 11.2 Existing Gold Models

| Model | Grain | Purpose |
| --- | --- | --- |
| `fct_football__team_match_context` | One row per fixture/team | Team perspective of a fixture, opponent, score, home/away, competition, formations. |
| `fct_football__player_match_features` | One row per fixture/team/player | Canonical player-match feature mart. |
| `fct_football__team_elo_history` | One row per fixture/team | Point-in-time team Elo before and after each fixture. |
| `fct_football__player_elo_history` | One row per fixture/team/player | Point-in-time player Elo and player modifiers. |
| `fct_football__lineup_elo_strength` | One row per fixture/team | Starting XI strength for complete confirmed lineups. |
| `fct_football__formation_matchup_history` | One row per fixture/team | Historical formation and formation-vs-formation context. |
| `fct_football__player_shot_features` | One row per fixture/team/player | Specialized count-model mart currently focused on shot/player-prop features. |
| `fct_football__player_sapm` | One row per fixture/team/player | Shot Adjusted Plus-Minus-ready mart. |
| `dim_football__rating_baseline` | One row per team/ranking | FIFA ranking baseline for initial ratings. |

### 11.3 Required Gold Extensions

The Gold layer should evolve from shot-specific features into a general player-event prediction layer.

Recommended additions:

| Model | Grain | Purpose |
| --- | --- | --- |
| `fct_football__team_match_stats_context` | One row per fixture/team | Curated team stats and possession context from `/fixtures/statistics`. |
| `fct_football__player_event_features` | One row per fixture/team/player | General feature mart for all target events. |
| `fct_football__player_expected_minutes` | One row per fixture/team/player | Pre-match expected minutes estimate for confirmed lineup players. |
| `pred_football__player_event_predictions` | One row per prediction key | Stored pre-match predictions for each player and target event. |

`fct_football__player_shot_features` can either be retained as a shot-focused mart or folded into `fct_football__player_event_features`.

### 11.4 General Player Event Feature Mart

The model-ready feature mart should include:

- fixture context:
  - fixture ID,
  - kickoff time,
  - league,
  - season,
  - home/away,
  - venue if available;
- team and opponent context:
  - team ID/name,
  - opponent ID/name,
  - team Elo pre-match,
  - opponent Elo pre-match,
  - attack/defense Elo components,
  - expected goals for/against pre-match,
  - expected shots for/against pre-match,
  - expected possession for/against pre-match;
- lineup context:
  - starter/substitute flag,
  - position,
  - position group,
  - formation,
  - formation grid,
  - starting XI attack/defense strength;
- player history:
  - last-N event rates per 90,
  - last-N raw event counts,
  - minutes history,
  - missed fixture count,
  - player offensive/defensive Elo;
- team flow context:
  - historical possession,
  - expected possession,
  - team pass volume,
  - opponent pass volume allowed,
  - team shot volume,
  - opponent shot volume allowed,
  - pressure/foul/card tendencies if available;
- target labels for training:
  - all events listed in Section 2.

### 11.5 Possession And Match Flow

Possession should become a first-class modeling concept.

Business reasoning:

- Stronger teams, often represented by higher Elo or attack/overall strength, usually have more possession.
- Weaker teams may produce fewer sustained possessions but more counterattacking bursts.
- Passing volume is heavily possession-driven.
- Saves are affected by opponent shot volume and defensive pressure.
- Fouls, cards, and offsides depend on match control, game state, role, and tactical setup.

Recommended feature families:

- `team_possession_l5_avg`
- `opponent_possession_l5_avg`
- `expected_possession_share`
- `team_passes_l5_p90_or_per_match`
- `opponent_passes_allowed_l5`
- `team_shots_l5`
- `opponent_shots_allowed_l5`
- `team_fouls_l5`
- `opponent_fouls_drawn_allowed_l5`
- `elo_possession_interaction`
- `formation_possession_profile`

Open modeling question:

> Should expected possession be an explicit intermediate model, a feature engineered from Elo and historical team statistics, or an implicit signal learned by the final event models?

Future agents should evaluate this empirically.

## 12. Expected Minutes

Expected minutes are critical because most target events are exposure-driven.

A player who is likely to play 90 minutes has a different event expectation than a player likely to play 55 minutes, even if their per-minute rate is similar.

### 12.1 Required Feature

Add or derive:

```text
expected_minutes
```

for each player in the confirmed lineup.

This value should be available at inference time before the match starts.

### 12.2 Candidate Approaches

Open options:

- Mean minutes over the last 5 appearances.
- Median minutes over the last 5 appearances.
- Trimmed mean over the last 5 appearances, dropping the lowest and highest value.
- Robust weighted average with recency decay.
- Position-specific expected minutes priors.
- Starter/substitute-specific priors.
- Separate expected-minutes model.

### 12.3 Important Caveat

A simple mean can be distorted by unusual events:

- injury substitution after 10 minutes,
- red card,
- tactical early substitution,
- match abandonment,
- player returning from injury,
- rotation in friendlies.

Because the sample is small, a median or trimmed mean may be more robust than a plain average.

Recommended first implementation:

```text
expected_minutes_l5_trimmed_or_median
```

Keep the final method as an explicit open decision until validated.

## 13. ML Training

### 13.1 Goal

Train models that predict player-level event counts or probabilities for the target events in Section 2.

### 13.2 Current Training Foundation

The repo currently includes:

- `football_analytics/ml_training.py`
- `scripts/train_poisson_lgbm.py`

The existing default target columns are:

```text
shots_total
fouls_committed
dribbles_attempts
```

This should be expanded to:

```text
offsides
shots_total
shots_on
goals_total
goals_assists
goals_saves
passes_total
fouls_drawn
fouls_committed
cards_yellow
cards_red
```

### 13.3 Model Family Guidance

Initial baseline:

- one Poisson LightGBM model per target event,
- exposure offset based on `expected_minutes` or actual `games_minutes` during training,
- chronological train/validation split,
- MLflow tracking.

Potential improvements:

- Negative binomial or zero-inflated models for overdispersed targets.
- Binary classification or calibrated probability models for sparse cards/goals.
- Goalkeeper-only model for saves.
- Position-group-specific models.
- Multi-task models sharing representations between related targets.
- Calibration layer for probability thresholds.

### 13.4 Training Labels

Training uses completed fixtures only.

Labels come from `stg_football__player_match_stats` and downstream Gold marts.

Training must avoid leakage:

- use only pre-match or lagged features for prediction rows,
- use actual match outcomes only as labels,
- ensure rolling features exclude the current fixture,
- ensure Elo `_pre` values are used for training features.

## 14. Pre-Match Inference

### 14.1 Trigger Window

Inference should run when confirmed lineups are available, approximately:

```text
T-60 to T-40 minutes before kickoff
```

The job should poll or be scheduled frequently enough to catch this window.

### 14.2 Required Inputs

For a fixture-level inference run:

- `fixture_id`
- confirmed lineups for both teams,
- latest medallion data,
- trained model name(s),
- model version(s),
- feature table or feature assembly logic,
- prediction run ID.

### 14.3 Inference Steps

1. Fetch or refresh fixture metadata.
2. Fetch confirmed lineups.
3. Write lineups to Bronze.
4. Run dbt transformations required for Silver and Gold lineup features.
5. Build inference rows for confirmed players.
6. Estimate `expected_minutes`.
7. Load trained model(s) from MLflow.
8. Generate predictions for every player and every target event.
9. Store predictions in the prediction table.
10. Optionally display predictions in a notebook cell.

### 14.4 Missing Lineup Behavior

Live/production mode:

- do not generate official predictions without confirmed lineups,
- mark run as skipped or failed with a clear reason.

Exploratory/offline mode:

- may use mock or projected lineups,
- prediction output must be flagged as non-official.

## 15. Prediction Output Table

### 15.1 Recommended Table

```text
gold.pred_football__player_event_predictions
```

### 15.2 Recommended Grain

One row per:

```text
fixture_id
team_id
player_id
target_event
model_name
model_version
prediction_set_id
```

### 15.3 Recommended Columns

| Column | Purpose |
| --- | --- |
| `prediction_set_id` | Groups all predictions produced by one fixture inference run. |
| `prediction_run_id` | Technical run ID from Databricks/job context. |
| `fixture_id` | API-Football fixture ID. |
| `fixture_date_utc` | Kickoff timestamp. |
| `prediction_created_at_utc` | Timestamp when predictions were generated. |
| `team_id` | Player's team ID. |
| `team_name` | Player's team name. |
| `opponent_team_id` | Opponent team ID. |
| `opponent_team_name` | Opponent team name. |
| `home_away` | Home or away perspective. |
| `player_id` | API-Football player ID. |
| `player_name` | Player display name. |
| `position_group` | `G`, `D`, `M`, `F`. |
| `is_starting` | Whether player is in confirmed starting XI. |
| `formation` | Team formation. |
| `target_event` | One of the supported prediction targets. |
| `expected_minutes` | Pre-match expected minutes used for exposure. |
| `predicted_mean` | Expected event count. |
| `predicted_p_ge_1` | Probability event count is at least 1, if available. |
| `predicted_p_ge_2` | Probability event count is at least 2, if available. |
| `predicted_p_ge_3` | Probability event count is at least 3, if available. |
| `model_name` | MLflow registered model name or logical model family. |
| `model_version` | MLflow model version or artifact version. |
| `model_stage_or_alias` | MLflow alias/stage, if used. |
| `feature_table_name` | Source feature table used for inference. |
| `feature_table_version` | Optional Delta version or timestamp. |
| `lineup_source` | `confirmed_api`, `projected`, `offline_sample`, etc. |
| `is_active_prediction` | Boolean flag for current active prediction set. |

### 15.4 Deduplication Strategy

The table should avoid multiple active prediction rows for the same fixture/model strategy.

Acceptable strategies:

**Option A: merge/upsert**

- Merge by:

```text
fixture_id
team_id
player_id
target_event
model_name
model_version
```

- Update prediction values when the same key is rerun.

**Option B: delete + insert**

- Delete existing rows for:

```text
fixture_id
model_name
model_version
```

- Insert the new prediction set.

**Option C: append with active flag**

- Append all runs for audit.
- Set older rows for the same fixture/model/version to `is_active_prediction=false`.
- Set latest rows to `is_active_prediction=true`.

Recommended production default:

> Use append with `is_active_prediction`, plus a deterministic `prediction_set_id`, because it preserves audit history while making the active prediction set easy to query.

## 16. Data Quality And Governance

### 16.1 Scope Control

Scope is controlled by:

- Python validation in `football_analytics/quality/validators.py`,
- dbt join to `senior_mens_international_leagues`.

This prevents accidental club, women's, youth, or Olympic fixtures from entering the modeling layer.

### 16.2 Key Tests

Maintain and extend tests for:

- unique fixture IDs,
- unique player rows by `fixture_id`, `team_id`, `player_id`,
- not-null event labels,
- accepted values for boolean flags,
- accepted values for `position_group`,
- sane minutes,
- sane event counts,
- no duplicate active predictions by fixture/player/target/model/version,
- complete prediction coverage for confirmed starting XI players.

### 16.3 Prediction Quality Monitoring

Future monitoring should track:

- model metrics by target,
- model metrics by position group,
- calibration for sparse events,
- prediction drift by competition,
- actual-vs-predicted after match completion,
- coverage of confirmed lineups,
- missing expected-minutes estimates,
- model version currently active.

## 17. Roadmap For Future Agents

Future agents should prioritize work in this order:

1. Add `/fixtures/statistics` ingestion to Bronze with checkpoint support.
2. Add Silver team match statistics model.
3. Add Gold team possession and match-flow context.
4. Generalize `fct_football__player_shot_features` into `fct_football__player_event_features`.
5. Add expected-minutes features and document the chosen robust estimator.
6. Expand ML target list to all events in Section 2.
7. Add MLflow model training and registration conventions for each target.
8. Add pre-match inference job for confirmed lineups.
9. Add `gold.pred_football__player_event_predictions`.
10. Add tests for prediction table uniqueness, coverage, and active-row semantics.
11. Remove all unnecessary code, dbt models, and pipeline scripts that are not used.

## 18. Repository Map

| Area | Files |
| --- | --- |
| Databricks Bundle | `databricks.yml`, `resources/international_medallion_pipeline.yml` |
| Databricks notebooks | `notebooks/00_prepare_run.py`, `notebooks/01_bronze_ingest.py`, `notebooks/02_dbt_python_models.py` |
| Bronze ingestion | `football_analytics/databricks_ingestion.py`, `football_analytics/api/client.py` |
| Scope validation | `football_analytics/quality/validators.py`, `football_analytics/league_scope.py` |
| dbt Silver | `dbt/models/staging/*` |
| dbt Gold | `dbt/models/marts/*` |
| dbt seeds | `dbt/seeds/*` |
| ML training | `football_analytics/ml_training.py`, `scripts/train_poisson_lgbm.py` |
| Simulation and reports | `football_analytics/modeling.py`, `football_analytics/orchestrator.py`, `football_analytics/reporting.py`, `run_pipeline.py` |

## 19. Non-Negotiable Design Principles

Future changes should follow these principles:

- Predictions must be player-level and fixture-specific.
- Confirmed lineups are required for official pre-match predictions.
- Prediction targets are the events listed in Section 2.
- Expected minutes must be modeled or robustly estimated.
- Team possession and match-flow context must be added through `/fixtures/statistics`.
- Feature generation must avoid leakage.
- Model metadata must be stored with predictions.
- The prediction table must prevent duplicate active records for the same fixture/player/target/model/version.
- All model improvements should be evaluated chronologically, not randomly, because football data is time-dependent.
