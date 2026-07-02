from football_analytics.databricks.config import DatabricksPipelineConfig


def table_name(config: DatabricksPipelineConfig, layer: str, name: str) -> str:
    return f"{config.catalog}.{config.schema_for_layer(layer)}.{name}"

