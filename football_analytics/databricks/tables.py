from football_analytics.databricks.config import DatabricksPipelineConfig


def table_name(config: DatabricksPipelineConfig, layer: str, name: str) -> str:
    return f"{config.catalog}.{config.schema_for_layer(layer)}.{name}"


def audit_table(config: DatabricksPipelineConfig) -> str:
    return table_name(config, "ops", "audit")


def dead_letter_table(config: DatabricksPipelineConfig) -> str:
    return table_name(config, "ops", "dead_letter")

