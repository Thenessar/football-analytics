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
