import os

# ==============================================================================
# Football API Ingestion Settings
# ==============================================================================
API_KEY = os.getenv("FOOTBALL_API_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {
    "x-rapidapi-host": "v3.football.api-sports.io",
}
if API_KEY:
    HEADERS["x-rapidapi-key"] = API_KEY

# Local cache settings for testing & minimizing redundant API requests
CACHE_FILE = "world_cup_2026_completed_data.json"
INTERNATIONAL_MATCH_ANCHOR_DATE = "2022-12-23"
HISTORICAL_ANCHOR_DATE = INTERNATIONAL_MATCH_ANCHOR_DATE
RECENCY_HALF_LIFE_DAYS = 365
FIFA_RANKINGS_SEED_FILE = "data/seeds/fifa_mens_world_ranking_december_2022.csv"
FIFA_RANKINGS_SEED_AS_OF_DATE = "2022-12-22"

# ==============================================================================
# Quantitative Modeling & Prior Configs
# ==============================================================================
# Global positional baseline priors per 90 minutes
POSITIONAL_PRIORS = {
    "F": {"shots": 2.6, "sot_pct": 0.42, "conversion": 0.16},
    "M": {"shots": 1.2, "sot_pct": 0.35, "conversion": 0.10},
    "D": {"shots": 0.5, "sot_pct": 0.25, "conversion": 0.05},
    "G": {"shots": 0.0, "sot_pct": 0.00, "conversion": 0.00}
}
