# Databricks Bundle

This repo contains a Databricks Asset Bundle for the senior men's international medallion ingestion pipeline.

## One-time setup

1. Install the modern Databricks CLI.
2. Authenticate to your workspace:

```powershell
databricks auth login --host https://<your-workspace-url>
```

3. Use serverless notebooks for Python ingestion tasks.

4. Find a serverless Databricks SQL warehouse ID for dbt tasks and pass it as `sql_warehouse_id`.

5. Create the API secret in the workspace:

```text
scope: football-api
key: api-football-key
```

## Validate and deploy

Run these from the repo root:

```powershell
databricks bundle validate -t dev --var="sql_warehouse_id=<warehouse-id>"
databricks bundle deploy -t dev --var="sql_warehouse_id=<warehouse-id>"
databricks bundle run international_medallion_pipeline -t dev --var="sql_warehouse_id=<warehouse-id>"
```

The bundle builds the local `football_analytics` package as a wheel during deploy and installs that wheel on notebook and dbt tasks. This makes imports such as `from football_analytics.databricks.config import ...` available in the serverless job environment and lets dbt Python models reuse the shared chronological ELO implementation.

If you authenticated with a named profile, add `-p <profile-name>` to each command.

For a one-fixture manual run, pass the fixture as a job parameter in Databricks or set `fixture_id`.
For the default incremental load, leave `fixture_id`, `target_date`, `date_from`, and `date_to` blank; the job starts at the latest completed checkpoint date on or before today and runs through today plus `lookahead_days` (default `7`).
Set `target_date` to force one calendar date.
For a backfill, leave `fixture_id` blank and set `date_from` plus `date_to`.

## Execution flow

The Databricks job keeps operational ingestion in Python and runs deterministic transformations in dbt:

```text
00_prepare_run.py
01_bronze_ingest.py
dbt deps
dbt seed
dbt build --exclude tag:python+
02_dbt_python_models.py
dbt build --select tag:python+ --exclude tag:python
```

`00_prepare_run.py` creates target schemas only. `01_bronze_ingest.py` lands raw API-Football payloads and checkpoint state in Bronze. `dbt seed` materializes shared reference data such as the senior men's international league allowlist. The first warehouse `dbt build` excludes seeds plus `tag:python+`, so it builds upstream SQL models without touching Python models or Python-dependent marts. `02_dbt_python_models.py` runs as a serverless notebook task and builds only `tag:python` models: `fct_football__team_elo_history` and `fct_football__player_elo_history`. Those Python models use dbt-databricks serverless Python submission, so no all-purpose cluster ID is required. The final warehouse `dbt build` selects `tag:python+` while excluding the Python-tagged models themselves, refreshing SQL descendants such as lineup Elo strength and player shot features after fresh Elo tables exist.

Silver staging models and Gold mart models live under `dbt/models`, including Python models that materialize point-in-time team and player ELO history before downstream SQL marts consume those features.

The bundled workflow is configured for serverless workflow execution: notebook tasks omit cluster settings so they run on serverless workflow compute, and SQL dbt tasks use the supplied serverless SQL warehouse plus a lightweight dbt serverless environment. dbt task `catalog` and `schema` are deploy-time bundle variables, not `{{job.parameters.*}}` runtime references, because Databricks validates those fields during job deployment.
The dbt task environment defaults to serverless environment version `4`; override `serverless_environment_version` if your workspace requires a different supported version.

Historical backfills use these widgets:

```text
run_id
target_date
date_from
date_to
lookahead_days
force_refresh
include_lineups
endpoint_max_workers
api_rate_limit_per_minute
```

The senior men's international league allowlist is generated from `football_analytics/league_scope.py` into `dbt/seeds/senior_mens_international_leagues.csv` with `python scripts/generate_league_seed.py`. The FIFA men's ranking seed is versioned at `dbt/seeds/fifa_mens_world_ranking_december_2022.csv`; `dim_football__rating_baseline` normalizes the source `Raiting` typo to `rating`.

The schedule is paused by default. Set `schedule_pause_status=UNPAUSED` only when the job is ready to run automatically.

## Local dbt workflow

Install dependencies from `requirements.txt`, then create a local profile from `dbt/profiles.yml.example` without committing credentials.

```powershell
dbt deps --project-dir dbt
dbt parse --project-dir dbt
dbt compile --project-dir dbt
```

Use vars to point at non-default Unity Catalog schemas:

```powershell
dbt seed --project-dir dbt --vars "{catalog: football_analytics, bronze_schema: bronze_dev, silver_schema: silver_dev, gold_schema: gold_dev}"
dbt build --project-dir dbt --exclude resource_type:seed --vars "{catalog: football_analytics, bronze_schema: bronze_dev, silver_schema: silver_dev, gold_schema: gold_dev}"
```

The Python models are tagged with `python` and use dbt-databricks `serverless_cluster` submission. Locally, use `dbt build --select tag:python` only when your profile points at Databricks and `DBT_WORKSPACE_ROOT_PATH` points at a deployed bundle root containing the project wheel.

If Databricks credentials or a SQL warehouse are unavailable locally, validate with `pytest -q` and `dbt parse` where possible, then run bundle validation/deploy from an authenticated Databricks CLI session.
