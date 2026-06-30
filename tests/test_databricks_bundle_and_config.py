from pathlib import Path
import re

import pytest

from football_analytics.databricks.config import DatabricksPipelineConfig, load_config_from_env
from football_analytics.databricks.tables import table_name


ROOT = Path(__file__).resolve().parents[1]


def test_databricks_config_defaults_are_medallion_oriented(monkeypatch):
    for name in (
        "FOOTBALL_CATALOG",
        "FOOTBALL_BRONZE_SCHEMA",
        "FOOTBALL_SILVER_SCHEMA",
        "FOOTBALL_GOLD_SCHEMA",
        "FOOTBALL_OPS_SCHEMA",
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
    assert table_name(config, "ops", "audit") == "fa.ops.audit"
    with pytest.raises(ValueError, match="Unsupported medallion layer"):
        table_name(config, "platinum", "x")


def test_bundle_passes_catalog_schema_parameters_to_table_aware_tasks():
    bundle = (ROOT / "resources" / "international_medallion_pipeline.yml").read_text(encoding="utf-8")

    for task_name in ("prepare_run", "bronze_ingest", "quality_checks"):
        match = re.search(
            rf"- task_key: {task_name}\b(?P<task>.*?)(?=\n        - task_key:|\Z)",
            bundle,
            flags=re.S,
        )
        assert match is not None
        task_block = match.group("task")
        for parameter in ("catalog", "bronze_schema", "silver_schema", "gold_schema", "ops_schema"):
            assert f"{parameter}: \"{{{{job.parameters.{parameter}}}}}\"" in task_block


def test_bundle_runs_dbt_after_bronze_ingestion():
    bundle = (ROOT / "resources" / "international_medallion_pipeline.yml").read_text(encoding="utf-8")

    for task_name in ("dbt_deps", "dbt_seed", "dbt_build"):
        assert f"task_key: {task_name}" in bundle

    assert "dbt deps" in bundle
    assert "dbt seed" in bundle
    assert "dbt build" in bundle
    assert "warehouse_id: ${var.sql_warehouse_id}" in bundle
    assert "environment_key: dbt_serverless" in bundle
    assert "dbt-databricks>=1.8.0" in bundle
    assert "bronze_schema: \\\"{{job.parameters.bronze_schema}}\\\"" in bundle
    assert "silver_schema: \\\"{{job.parameters.silver_schema}}\\\"" in bundle
    assert "gold_schema: \\\"{{job.parameters.gold_schema}}\\\"" in bundle
    assert "task_key: dbt_build" in re.search(
        r"- task_key: quality_checks\b(?P<task>.*?)(?=\n        - task_key:|\Z)",
        bundle,
        flags=re.S,
    ).group("task")


def test_databricks_notebook_files_match_medallion_order():
    notebook_names = sorted(path.name for path in (ROOT / "notebooks").glob("*.py"))

    assert notebook_names == [
        "00_prepare_run.py",
        "01_bronze_ingest.py",
        "02_silver_normalize.py",
        "03_gold_build.py",
        "04_quality_checks.py",
    ]


def test_bundle_references_renamed_notebooks_and_job():
    bundle = (ROOT / "resources" / "international_medallion_pipeline.yml").read_text(encoding="utf-8")

    for expected in (
        "international_medallion_pipeline",
        "../notebooks/00_prepare_run.py",
        "../notebooks/01_bronze_ingest.py",
        "../notebooks/04_quality_checks.py",
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
        "../notebooks/02_silver_normalize.py",
        "../notebooks/03_gold_build.py",
        "existing_cluster_id",
        "league_id",
        "season",
    ):
        assert stale not in bundle


def test_dbt_project_contains_expected_models_and_seed():
    dbt_root = ROOT / "dbt"

    for expected in (
        "dbt_project.yml",
        "profiles.yml.example",
        "packages.yml",
        "macros/normalize_name.sql",
        "models/sources.yml",
        "models/staging/stg_football_fixtures.sql",
        "models/staging/stg_football_player_match_stats.sql",
        "models/staging/stg_football_lineups.sql",
        "models/marts/football_team_match_context.sql",
        "models/marts/football_rating_baseline.sql",
        "models/marts/football_player_shot_features.sql",
        "models/marts/football_player_sapm.sql",
        "seeds/fifa_mens_world_ranking_december_2022.csv",
    ):
        assert (dbt_root / expected).exists()

    model_sql = "\n".join(path.read_text(encoding="utf-8") for path in (dbt_root / "models").rglob("*.sql"))
    for expected_ref in (
        "source('bronze', 'football_fixtures_raw')",
        "source('bronze', 'football_match_raw')",
        "source('bronze', 'football_lineups_raw')",
        "ref('stg_football_fixtures')",
        "ref('stg_football_player_match_stats')",
        "ref('football_team_match_context')",
    ):
        assert expected_ref in model_sql


def test_current_databricks_docs_do_not_describe_pipeline_as_world_cup_only():
    docs = (ROOT / "DATABRICKS_BUNDLE.md").read_text(encoding="utf-8")

    assert "World Cup ingestion pipeline" not in docs
    assert "senior men's international medallion ingestion pipeline" in docs
    for flow_step in (
        "00_prepare_run.py",
        "01_bronze_ingest.py",
        "dbt deps",
        "dbt seed",
        "dbt build",
        "04_quality_checks.py",
    ):
        assert flow_step in docs

    assert "Silver staging models and Gold mart models live under `dbt/models`" in docs
    assert "Free Edition/serverless-style execution" in docs
