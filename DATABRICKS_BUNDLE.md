# Databricks Bundle

This repo contains a Databricks Asset Bundle for the senior men's international medallion ingestion pipeline.

## One-time setup

1. Install the modern Databricks CLI.
2. Authenticate to your workspace:

```powershell
databricks auth login --host https://<your-workspace-url>
```

3. Find an existing cluster ID in Databricks and pass it as `cluster_id`.

4. Create the API secret in the workspace:

```text
scope: football-api
key: api-football-key
```

## Validate and deploy

Run these from the repo root:

```powershell
databricks bundle validate -t dev --var="cluster_id=<cluster-id>"
databricks bundle deploy -t dev --var="cluster_id=<cluster-id>"
databricks bundle run international_medallion_pipeline -t dev --var="cluster_id=<cluster-id>"
```

If you authenticated with a named profile, add `-p <profile-name>` to each command.

For a one-fixture manual run, pass the fixture as a job parameter in Databricks or set `fixture_id`.
For a daily load, leave `fixture_id` blank and set `target_date`.
For a backfill, leave `fixture_id` blank and set `date_from` plus `date_to`.

## Manual notebook order

Run the notebooks in this order for manual Databricks operations:

```text
00_prepare_run.py
01_bronze_ingest.py
02_silver_normalize.py
03_gold_build.py
04_quality_checks.py
```

Historical backfills use these widgets:

```text
run_id
target_date
date_from
date_to
force_refresh
include_lineups
endpoint_max_workers
api_rate_limit_per_minute
load_rankings_seed
```

The FIFA men's ranking seed is versioned at `data/seeds/fifa_mens_world_ranking_december_2022.csv`.
The prepare notebook can materialize it as a Delta seed table before ingestion.

The schedule is paused by default. Set `schedule_pause_status=UNPAUSED` only when the job is ready to run automatically.
