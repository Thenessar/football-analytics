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

If you authenticated with a named profile, add `-p <profile-name>` to each command.

For a one-fixture manual run, pass the fixture as a job parameter in Databricks or set `fixture_id`.
For a daily load, leave `fixture_id` blank and set `target_date`.
For a backfill, leave `fixture_id` blank and set `date_from` plus `date_to`.

## Execution flow

The Databricks job keeps operational ingestion in Python and runs deterministic transformations in dbt:

```text
00_prepare_run.py
01_bronze_ingest.py
dbt deps
dbt seed
dbt build
04_quality_checks.py
```

`00_prepare_run.py` creates the target schemas and can still materialize the legacy Python seed table. `01_bronze_ingest.py` only lands raw API-Football payloads and checkpoint state in Bronze/Ops. Silver staging models and Gold mart models live under `dbt/models`.

The bundled workflow is configured for Free Edition/serverless-style execution: notebook tasks omit cluster settings so they run on serverless workflow compute, and dbt tasks use the supplied serverless SQL warehouse plus a lightweight dbt serverless environment.

Historical backfills use these widgets:

```text
run_id
target_date
date_from
date_to
force_refresh
include_lineups
load_rankings_seed
```

The FIFA men's ranking seed is versioned at `data/seeds/fifa_mens_world_ranking_december_2022.csv`.
The dbt copy is versioned at `dbt/seeds/fifa_mens_world_ranking_december_2022.csv`; `football_rating_baseline` normalizes the source `Raiting` typo to `rating`.

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
dbt build --project-dir dbt --vars "{catalog: football_analytics, bronze_schema: bronze_dev, silver_schema: silver_dev, gold_schema: gold_dev}"
```

If Databricks credentials or a SQL warehouse are unavailable locally, validate with `pytest -q` and `dbt parse` where possible, then run bundle validation/deploy from an authenticated Databricks CLI session.
