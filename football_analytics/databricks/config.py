import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DatabricksPipelineConfig:
    catalog: str = "main"
    schema: str = "football"
    league_id: int = 1
    season: int = 2026
    api_key: Optional[str] = None

    @property
    def namespace(self) -> str:
        return f"{self.catalog}.{self.schema}"


def load_config_from_env() -> DatabricksPipelineConfig:
    return DatabricksPipelineConfig(
        catalog=os.getenv("FOOTBALL_CATALOG", "main"),
        schema=os.getenv("FOOTBALL_SCHEMA", "football"),
        league_id=int(os.getenv("FOOTBALL_LEAGUE_ID", "1")),
        season=int(os.getenv("FOOTBALL_SEASON", "2026")),
        api_key=os.getenv("FOOTBALL_API_KEY"),
    )

