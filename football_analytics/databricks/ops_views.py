from dataclasses import dataclass
from typing import Optional

from football_analytics.databricks.config import DatabricksPipelineConfig
from football_analytics.databricks.tables import table_name
from football_analytics.databricks_ingestion import (
    CHECKPOINT_COMPLETED,
    CHECKPOINT_FAILED,
    CHECKPOINT_PENDING,
    CHECKPOINT_SKIPPED,
    ensure_fixture_endpoint_checkpoint_table,
)


@dataclass(frozen=True)
class BronzeOpsViewNames:
    run_summary: str
    failures: str
    pending: str

    def as_dict(self) -> dict[str, str]:
        return {
            "run_summary": self.run_summary,
            "failures": self.failures,
            "pending": self.pending,
        }


def bronze_ops_view_names(config: DatabricksPipelineConfig) -> BronzeOpsViewNames:
    return BronzeOpsViewNames(
        run_summary=table_name(config, "ops", "v_bronze_ingestion_run_summary"),
        failures=table_name(config, "ops", "v_bronze_ingestion_failures"),
        pending=table_name(config, "ops", "v_bronze_ingestion_pending"),
    )


def bronze_ops_view_sql(
    config: DatabricksPipelineConfig,
    *,
    checkpoint_table: Optional[str] = None,
) -> dict[str, str]:
    checkpoint = checkpoint_table or table_name(config, "ops", "ingestion_state_checkpoint")
    views = bronze_ops_view_names(config)
    return {
        views.run_summary: f"""
            CREATE OR REPLACE VIEW {views.run_summary} AS
            SELECT
                run_id,
                target_date,
                endpoint,
                COUNT(*) AS checkpoint_rows,
                SUM(CASE WHEN fixture_id IS NOT NULL THEN 1 ELSE 0 END) AS fixture_rows,
                SUM(CASE WHEN status = '{CHECKPOINT_PENDING}' THEN 1 ELSE 0 END) AS pending_count,
                SUM(CASE WHEN status = '{CHECKPOINT_COMPLETED}' THEN 1 ELSE 0 END) AS completed_count,
                SUM(CASE WHEN status = '{CHECKPOINT_SKIPPED}' THEN 1 ELSE 0 END) AS skipped_count,
                SUM(CASE WHEN status = '{CHECKPOINT_FAILED}' THEN 1 ELSE 0 END) AS failed_count,
                MIN(started_at_utc) AS first_started_at_utc,
                MAX(started_at_utc) AS last_started_at_utc,
                MAX(completed_at_utc) AS last_completed_at_utc
            FROM {checkpoint}
            GROUP BY run_id, target_date, endpoint
        """,
        views.failures: f"""
            CREATE OR REPLACE VIEW {views.failures} AS
            SELECT
                run_id,
                target_date,
                endpoint,
                fixture_id,
                status,
                attempt_count,
                last_error,
                started_at_utc,
                completed_at_utc
            FROM {checkpoint}
            WHERE status = '{CHECKPOINT_FAILED}'
        """,
        views.pending: f"""
            CREATE OR REPLACE VIEW {views.pending} AS
            SELECT
                run_id,
                target_date,
                endpoint,
                fixture_id,
                status,
                attempt_count,
                last_error,
                started_at_utc,
                timestampdiff(MINUTE, started_at_utc, current_timestamp()) AS minutes_pending
            FROM {checkpoint}
            WHERE status = '{CHECKPOINT_PENDING}'
        """,
    }


def create_bronze_ops_views(
    spark,
    config: DatabricksPipelineConfig,
    *,
    checkpoint_table: Optional[str] = None,
) -> BronzeOpsViewNames:
    checkpoint = checkpoint_table or table_name(config, "ops", "ingestion_state_checkpoint")
    ensure_fixture_endpoint_checkpoint_table(spark, checkpoint_table=checkpoint)
    view_sql = bronze_ops_view_sql(config, checkpoint_table=checkpoint)
    for sql in view_sql.values():
        spark.sql(sql)
    return bronze_ops_view_names(config)
