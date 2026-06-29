from football_analytics.databricks.config import DatabricksPipelineConfig
from football_analytics.databricks.ops_views import bronze_ops_view_names, bronze_ops_view_sql


def test_bronze_ops_view_names_use_ops_schema():
    config = DatabricksPipelineConfig(catalog="fa", ops_schema="operations")

    views = bronze_ops_view_names(config)

    assert views.run_summary == "fa.operations.v_bronze_ingestion_run_summary"
    assert views.failures == "fa.operations.v_bronze_ingestion_failures"
    assert views.pending == "fa.operations.v_bronze_ingestion_pending"


def test_bronze_ops_view_sql_summarizes_failures_and_pending_rows():
    config = DatabricksPipelineConfig(catalog="fa", ops_schema="ops")

    sql_by_view = bronze_ops_view_sql(config, checkpoint_table="fa.ops.ingestion_state_checkpoint")
    combined_sql = "\n".join(sql_by_view.values())

    assert "CREATE OR REPLACE VIEW fa.ops.v_bronze_ingestion_run_summary" in combined_sql
    assert "FROM fa.ops.ingestion_state_checkpoint" in combined_sql
    assert "failed_count" in combined_sql
    assert "WHERE status = 'FAILED'" in combined_sql
    assert "CREATE OR REPLACE VIEW fa.ops.v_bronze_ingestion_pending" in combined_sql
    assert "timestampdiff(MINUTE, started_at_utc, current_timestamp()) AS minutes_pending" in combined_sql
