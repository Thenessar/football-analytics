from pathlib import Path
import re
import csv

import pytest

from football_analytics.databricks.config import DatabricksPipelineConfig, load_config_from_env
from football_analytics.databricks.tables import table_name
from football_analytics.league_scope import SENIOR_MENS_INTERNATIONAL_LEAGUES


ROOT = Path(__file__).resolve().parents[1]


def test_databricks_config_defaults_are_medallion_oriented(monkeypatch):
    for name in (
        "FOOTBALL_CATALOG",
        "FOOTBALL_BRONZE_SCHEMA",
        "FOOTBALL_SILVER_SCHEMA",
        "FOOTBALL_GOLD_SCHEMA",
        "FOOTBALL_LEAGUE_ID",
        "FOOTBALL_SEASON",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_config_from_env()

    assert config.catalog == "football_analytics"
    assert config.bronze_schema == "bronze"
    assert config.silver_schema == "silver"
    assert config.gold_schema == "gold"
    assert not hasattr(config, "league_id")
    assert not hasattr(config, "season")


def test_databricks_table_names_use_layer_schemas():
    config = DatabricksPipelineConfig(catalog="fa", bronze_schema="b", silver_schema="s", gold_schema="g")

    assert table_name(config, "bronze", "raw_fixture_payloads") == "fa.b.raw_fixture_payloads"
    assert table_name(config, "silver", "fixtures") == "fa.s.fixtures"
    assert table_name(config, "gold", "player_sapm") == "fa.g.player_sapm"
    with pytest.raises(ValueError, match="Unsupported medallion layer"):
        table_name(config, "platinum", "x")


def test_bundle_passes_catalog_schema_parameters_to_table_aware_tasks():
    bundle = (ROOT / "resources" / "international_medallion_pipeline.yml").read_text(encoding="utf-8")

    for task_name in ("prepare_run", "bronze_ingest"):
        match = re.search(
            rf"- task_key: {task_name}\b(?P<task>.*?)(?=\n        - task_key:|\Z)",
            bundle,
            flags=re.S,
        )
        assert match is not None
        task_block = match.group("task")
        for parameter in ("catalog", "bronze_schema", "silver_schema", "gold_schema"):
            assert f"{parameter}: \"{{{{job.parameters.{parameter}}}}}\"" in task_block

def test_bundle_runs_dbt_after_bronze_ingestion():
    bundle = (ROOT / "resources" / "international_medallion_pipeline.yml").read_text(encoding="utf-8")

    for task_name in (
        "dbt_deps",
        "dbt_seed",
        "dbt_build",
        "dbt_python_models",
        "dbt_build_python_dependents",
    ):
        assert f"task_key: {task_name}" in bundle

    assert "dbt deps" in bundle
    assert "dbt seed" in bundle
    assert "dbt build" in bundle
    assert "warehouse_id: ${var.sql_warehouse_id}" in bundle
    assert "existing_cluster_id:" not in bundle
    assert "dbt_python_cluster_id" not in bundle
    assert "environment_key: dbt_serverless" in bundle
    assert "environment_version: ${var.serverless_environment_version}" in bundle
    assert "dbt-databricks>=1.8.0" in bundle
    assert "bronze_schema: \\\"{{job.parameters.bronze_schema}}\\\"" in bundle
    assert "silver_schema: \\\"{{job.parameters.silver_schema}}\\\"" in bundle
    assert "gold_schema: \\\"{{job.parameters.gold_schema}}\\\"" in bundle
    assert bundle.count("catalog: ${var.catalog}") == 4
    assert bundle.count("schema: ${var.silver_schema}") == 4
    assert "dbt build --exclude resource_type:seed tag:python+" in bundle
    assert "dbt build --select tag:python+ --exclude tag:python resource_type:seed" in bundle
    assert "../notebooks/02_dbt_python_models.py" in bundle

    python_task = re.search(
        r"- task_key: dbt_python_models\b(?P<task>.*?)(?=\n        - task_key:|\Z)",
        bundle,
        flags=re.S,
    ).group("task")
    assert "- task_key: dbt_build" in python_task
    assert "notebook_task:" in python_task
    assert "sql_warehouse_id: ${var.sql_warehouse_id}" in python_task
    assert "environment_key: dbt_serverless" in python_task
    assert "libraries:" not in python_task
    assert "existing_cluster_id:" not in python_task

    downstream_task = re.search(
        r"- task_key: dbt_build_python_dependents\b(?P<task>.*?)(?=\n      environments:|\Z)",
        bundle,
        flags=re.S,
    ).group("task")
    assert "- task_key: dbt_python_models" in downstream_task

    for task_name in ("dbt_deps", "dbt_seed", "dbt_build", "dbt_build_python_dependents"):
        task_block = re.search(
            rf"- task_key: {task_name}\b(?P<task>.*?)(?=\n        - task_key:|\Z)",
            bundle,
            flags=re.S,
        ).group("task")
        assert "catalog: ${var.catalog}" in task_block
        assert "schema: ${var.silver_schema}" in task_block
        assert "catalog: \"{{job.parameters.catalog}}\"" not in task_block
        assert "schema: \"{{job.parameters.silver_schema}}\"" not in task_block
    assert "task_key: quality_checks" not in bundle


def test_bundle_passes_parallelism_parameters_to_bronze_ingest():
    bundle = (ROOT / "resources" / "international_medallion_pipeline.yml").read_text(encoding="utf-8")
    match = re.search(
        r"- task_key: bronze_ingest\b(?P<task>.*?)(?=\n        - task_key:|\Z)",
        bundle,
        flags=re.S,
    )

    assert match is not None
    task_block = match.group("task")
    for parameter in ("endpoint_max_workers", "api_rate_limit_per_minute"):
        assert f"{parameter}: \"{{{{job.parameters.{parameter}}}}}\"" in task_block


def test_bundle_passes_statistics_toggle_to_bronze_ingest():
    bundle = (ROOT / "resources" / "international_medallion_pipeline.yml").read_text(encoding="utf-8")
    notebook = (ROOT / "notebooks" / "01_bronze_ingest.py").read_text(encoding="utf-8")
    match = re.search(
        r"- task_key: bronze_ingest\b(?P<task>.*?)(?=\n        - task_key:|\Z)",
        bundle,
        flags=re.S,
    )

    assert match is not None
    assert 'include_statistics: "{{job.parameters.include_statistics}}"' in match.group("task")
    assert "- name: include_statistics" in bundle
    assert "ingest_fixture_statistics_for_fixtures_to_bronze" in notebook
    assert 'table_name(config, "bronze", "football_fixture_statistics_raw")' in notebook


def test_bundle_passes_default_lookahead_to_notebook_tasks():
    bundle_config = (ROOT / "databricks.yml").read_text(encoding="utf-8")
    bundle = (ROOT / "resources" / "international_medallion_pipeline.yml").read_text(encoding="utf-8")

    assert "lookahead_days:" in bundle_config
    assert "default: \"7\"" in bundle_config
    assert "default: ${var.lookahead_days}" in bundle

    for task_name in ("prepare_run", "bronze_ingest"):
        match = re.search(
            rf"- task_key: {task_name}\b(?P<task>.*?)(?=\n        - task_key:|\Z)",
            bundle,
            flags=re.S,
        )
        assert match is not None
        assert 'lookahead_days: "{{job.parameters.lookahead_days}}"' in match.group("task")


def test_bundle_installs_project_wheel_for_serverless_notebook_tasks():
    bundle_config = (ROOT / "databricks.yml").read_text(encoding="utf-8")
    job = (ROOT / "resources" / "international_medallion_pipeline.yml").read_text(encoding="utf-8")

    assert "football_analytics_wheel:" in bundle_config
    assert "type: whl" in bundle_config
    assert "python -m pip wheel . --wheel-dir dist --no-deps" in bundle_config
    assert "path: ." in bundle_config
    assert "environment_key: notebook_serverless" in job
    assert "${workspace.root_path}/artifacts/.internal/football_analytics-0.1.0-py3-none-any.whl" in job
    assert "environment_key: dbt_serverless" in job

    for task_name in ("prepare_run", "bronze_ingest"):
        match = re.search(
            rf"- task_key: {task_name}\b(?P<task>.*?)(?=\n        - task_key:|\Z)",
            job,
            flags=re.S,
        )
        assert match is not None
        assert "environment_key: notebook_serverless" in match.group("task")
        assert "libraries:" not in match.group("task")

    python_task = re.search(
        r"- task_key: dbt_python_models\b(?P<task>.*?)(?=\n        - task_key:|\Z)",
        job,
        flags=re.S,
    )
    assert python_task is not None
    assert "environment_key: dbt_serverless" in python_task.group("task")
    assert "libraries:" not in python_task.group("task")

    dbt_environment = re.search(
        r"- environment_key: dbt_serverless\b(?P<environment>.*?)(?=\n        - environment_key:|\Z)",
        job,
        flags=re.S,
    )
    assert dbt_environment is not None
    assert "${workspace.root_path}/artifacts/.internal/football_analytics-0.1.0-py3-none-any.whl" in dbt_environment.group("environment")


def test_production_bundle_uses_stable_workspace_root_path():
    bundle_config = (ROOT / "databricks.yml").read_text(encoding="utf-8")

    assert "mode: production" in bundle_config
    assert "root_path: /Workspace/Users/adam.r.denes@gmail.com/.bundle/${bundle.name}/${bundle.target}" in bundle_config


def test_databricks_notebook_files_match_medallion_order():
    notebook_names = sorted(path.name for path in (ROOT / "notebooks").glob("*.py"))

    assert notebook_names == [
        "00_prepare_run.py",
        "01_bronze_ingest.py",
        "02_dbt_python_models.py",
    ]


def test_bundle_references_renamed_notebooks_and_job():
    bundle = (ROOT / "resources" / "international_medallion_pipeline.yml").read_text(encoding="utf-8")

    for expected in (
        "international_medallion_pipeline",
        "../notebooks/00_prepare_run.py",
        "../notebooks/01_bronze_ingest.py",
        "../notebooks/02_dbt_python_models.py",
    ):
        assert expected in bundle

    for stale in (
        "worldcup_pipeline",
        "orchestrate_worldcup_run",
        "05_orchestrate_worldcup_run.py",
        "01_bronze_api_ingest.py",
        "02_silver_validate_normalize.py",
        "03_gold_feature_build.py",
        "04_pipeline_quality_checks.py",
        "04_quality_checks.py",
        "../notebooks/02_silver_normalize.py",
        "../notebooks/03_gold_build.py",
        "../notebooks/04_quality_checks.py",
        "existing_cluster_id",
        "dbt_python_cluster_id",
        "league_id",
        "season",
        "load_rankings_seed",
    ):
        assert stale not in bundle


def test_dbt_python_models_are_tagged_for_serverless_execution():
    schema = (ROOT / "dbt" / "models" / "marts" / "schema.yml").read_text(encoding="utf-8")

    for model_name in (
        "fct_football__team_elo_history",
        "fct_football__player_elo_history",
    ):
        match = re.search(
            rf"- name: {model_name}\b(?P<model>.*?)(?=\n  - name:|\Z)",
            schema,
            flags=re.S,
        )
        assert match is not None
        model_block = match.group("model")
        assert "tags:" in model_block
        assert "- python" in model_block
        assert "submission_method: serverless_cluster" in model_block
        assert "environment_key: dbt_python_serverless" in model_block
        assert "DBT_WORKSPACE_ROOT_PATH" in model_block
        assert "cluster_id:" not in model_block


def test_player_match_feature_mart_keeps_only_played_appearances():
    model_sql = (
        ROOT / "dbt" / "models" / "marts" / "fct_football__player_match_features.sql"
    ).read_text(encoding="utf-8")

    assert "where coalesce(games_minutes, 0) between 1 and 130" in model_sql


def test_dbt_project_contains_expected_models_and_seed():
    dbt_root = ROOT / "dbt"

    for expected in (
        "dbt_project.yml",
        "profiles.yml.example",
        "packages.yml",
        "macros/normalize_name.sql",
        "models/sources.yml",
        "models/staging/stg_football__fixtures.sql",
        "models/staging/stg_football__player_match_stats.sql",
        "models/staging/stg_football__lineups.sql",
        "models/staging/stg_football__team_match_stats.sql",
        "models/marts/fct_football__team_match_context.sql",
        "models/marts/dim_football__rating_baseline.sql",
        "models/marts/fct_football__team_elo_history.py",
        "models/marts/fct_football__player_elo_history.py",
        "models/marts/fct_football__lineup_elo_strength.sql",
        "models/marts/fct_football__player_shot_features.sql",
        "models/marts/fct_football__formation_matchup_history.sql",
        "models/marts/fct_football__player_sapm.sql",
        "seeds/senior_mens_international_leagues.csv",
        "seeds/fifa_mens_world_ranking_december_2022.csv",
    ):
        assert (dbt_root / expected).exists()

    model_sql = "\n".join(
        path.read_text(encoding="utf-8")
        for pattern in ("*.sql", "*.py")
        for path in (dbt_root / "models").rglob(pattern)
    )
    for expected_ref in (
        "source('bronze', 'football_fixtures_raw')",
        "source('bronze', 'football_match_raw')",
        "source('bronze', 'football_lineups_raw')",
        "source('bronze', 'football_fixture_statistics_raw')",
        "ref('stg_football__fixtures')",
        "ref('stg_football__player_match_stats')",
        "ref('fct_football__team_match_context')",
        "ref('fct_football__team_elo_history')",
        "ref('fct_football__player_elo_history')",
        "ref('fct_football__lineup_elo_strength')",
        "ref('fct_football__formation_matchup_history')",
        "ref('senior_mens_international_leagues')",
    ):
        assert expected_ref in model_sql


def test_dbt_league_seed_matches_python_source_of_truth():
    seed_path = ROOT / "dbt" / "seeds" / "senior_mens_international_leagues.csv"

    with seed_path.open(encoding="utf-8", newline="") as handle:
        rows = [
            (int(row["league_id"]), row["league_name"])
            for row in csv.DictReader(handle)
        ]

    assert rows == list(SENIOR_MENS_INTERNATIONAL_LEAGUES)


def test_current_databricks_docs_do_not_describe_pipeline_as_world_cup_only():
    docs = (ROOT / "DATABRICKS_BUNDLE.md").read_text(encoding="utf-8")

    assert "World Cup ingestion pipeline" not in docs
    assert "senior men's international medallion ingestion pipeline" in docs
    for flow_step in (
        "00_prepare_run.py",
        "01_bronze_ingest.py",
        "dbt deps",
        "dbt seed",
        "dbt build --exclude tag:python+",
        "02_dbt_python_models.py",
        "dbt build --select tag:python+ --exclude tag:python",
    ):
        assert flow_step in docs

    assert "`00_prepare_run.py` creates target schemas only" in docs
    assert "Silver staging models and Gold mart models live under `dbt/models`" in docs
    assert "serverless workflow execution" in docs
    assert "no all-purpose cluster ID is required" in docs
