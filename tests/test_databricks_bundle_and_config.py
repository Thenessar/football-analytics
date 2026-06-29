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

    for task_name in ("prepare_run", "bronze_ingest", "silver_normalize", "gold_build", "quality_checks"):
        match = re.search(
            rf"- task_key: {task_name}\b(?P<task>.*?)(?=\n        - task_key:|\Z)",
            bundle,
            flags=re.S,
        )
        assert match is not None
        task_block = match.group("task")
        for parameter in ("catalog", "bronze_schema", "silver_schema", "gold_schema", "ops_schema"):
            assert f"{parameter}: \"{{{{job.parameters.{parameter}}}}}\"" in task_block


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
        "../notebooks/02_silver_normalize.py",
        "../notebooks/03_gold_build.py",
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
        "league_id",
        "season",
    ):
        assert stale not in bundle


def test_current_databricks_docs_do_not_describe_pipeline_as_world_cup_only():
    docs = (ROOT / "DATABRICKS_BUNDLE.md").read_text(encoding="utf-8")

    assert "World Cup ingestion pipeline" not in docs
    assert "senior men's international medallion ingestion pipeline" in docs
    for notebook_name in (
        "00_prepare_run.py",
        "01_bronze_ingest.py",
        "02_silver_normalize.py",
        "03_gold_build.py",
        "04_quality_checks.py",
    ):
        assert notebook_name in docs
