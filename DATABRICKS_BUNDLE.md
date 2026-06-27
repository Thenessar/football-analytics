# Databricks Bundle

This repo contains a Databricks Asset Bundle for the World Cup ingestion pipeline.

## One-time setup

1. Install the modern Databricks CLI.
2. Authenticate to your workspace:

```powershell
databricks auth login --host https://<your-workspace-url>
```

3. Find an existing cluster ID in Databricks and pass it as `cluster_id`.

4. Create the API secret in the workspace:

```text
scope: worldcup
key: football_api_key
```

## Validate and deploy

Run these from the repo root:

```powershell
databricks bundle validate -t dev --var="cluster_id=<cluster-id>"
databricks bundle deploy -t dev --var="cluster_id=<cluster-id>"
databricks bundle run worldcup_pipeline -t dev --var="cluster_id=<cluster-id>"
```

If you authenticated with a named profile, add `-p <profile-name>` to each command.

The schedule is paused by default. Set `schedule_pause_status=UNPAUSED` only when the job is ready to run automatically.

