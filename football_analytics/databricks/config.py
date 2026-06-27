import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatabricksPipelineConfig:
    catalog: str = "football_analytics"
    bronze_schema: str = "bronze"
    silver_schema: str = "silver"
    gold_schema: str = "gold"
    ops_schema: str = "ops"
    api_key: Optional[str] = None

    @property
    def namespace(self) -> str:
        """Default namespace for operational tables, not a competition filter."""
        return f"{self.catalog}.{self.ops_schema}"

    def schema_for_layer(self, layer: str) -> str:
        schemas = {
            "bronze": self.bronze_schema,
            "silver": self.silver_schema,
            "gold": self.gold_schema,
            "ops": self.ops_schema,
        }
        try:
            return schemas[layer]
        except KeyError as error:
            raise ValueError(f"Unsupported medallion layer: {layer}") from error


def load_config_from_env() -> DatabricksPipelineConfig:
    return DatabricksPipelineConfig(
        catalog=os.getenv("FOOTBALL_CATALOG", "football_analytics"),
        bronze_schema=os.getenv("FOOTBALL_BRONZE_SCHEMA", "bronze"),
        silver_schema=os.getenv("FOOTBALL_SILVER_SCHEMA", "silver"),
        gold_schema=os.getenv("FOOTBALL_GOLD_SCHEMA", "gold"),
        ops_schema=os.getenv("FOOTBALL_OPS_SCHEMA", "ops"),
        api_key=os.getenv("FOOTBALL_API_KEY"),
    )

