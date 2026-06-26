import pytest

from football_analytics import databricks_ingestion as ingestion


def test_delta_paths_match_bronze_and_silver_contract():
    assert ingestion.BRONZE_FOOTBALL_MATCH_RAW_PATH == "/mnt/syndicate/bronze/football_match_raw"
    assert ingestion.SILVER_PLAYER_MATCH_STATS_PATH == "/mnt/syndicate/silver/football_player_match_stats"
    assert ingestion.INGESTION_STATE_CHECKPOINT_TABLE == "default.ingestion_state_checkpoint"


def test_weekly_windows_are_deterministic_from_anchor():
    assert ingestion.iter_weekly_windows("2022-11-07", "2022-11-20") == [
        ("2022-11-07", "2022-11-13"),
        ("2022-11-14", "2022-11-20"),
    ]


def test_accent_translation_map_is_valid_for_pyspark_translate():
    assert len(ingestion.ACCENTED_CHARS) == len(ingestion.ASCII_CHARS)
    assert "Á" in ingestion.ACCENTED_CHARS
    assert "ç" in ingestion.ACCENTED_CHARS
    assert "ã" in ingestion.ACCENTED_CHARS


def test_pyspark_is_loaded_lazily():
    with pytest.raises(RuntimeError, match="PySpark is required"):
        ingestion.normalized_name_sql("player_name")


def test_fetch_football_api_payload_preserves_full_response_envelope(monkeypatch):
    captured = {}

    class FootballApiResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "get": "fixtures/players",
                "parameters": {"fixture": "1489437"},
                "errors": [],
                "response": [{"team": {"name": "Curaçao"}, "players": []}],
            }

    def fake_get(url, headers, params, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        captured["timeout"] = timeout
        return FootballApiResponse()

    monkeypatch.setattr("football_analytics.databricks_ingestion.requests.get", fake_get)

    payload = ingestion.fetch_football_api_payload(
        "fixtures/players",
        {"fixture": 1489437},
        api_key="secret",
    )

    assert payload["response"][0]["team"]["name"] == "Curaçao"
    assert payload["parameters"] == {"fixture": "1489437"}
    assert captured["url"].endswith("/fixtures/players")
    assert captured["headers"]["x-rapidapi-key"] == "secret"
    assert captured["headers"]["x-apisports-key"] == "secret"
    assert captured["params"] == {"fixture": 1489437}
    assert captured["timeout"] == 30


def test_fetch_football_api_payload_fails_on_provider_errors(monkeypatch):
    class FootballApiErrorResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"errors": {"fixture": "required"}, "response": []}

    monkeypatch.setattr(
        "football_analytics.databricks_ingestion.requests.get",
        lambda *args, **kwargs: FootballApiErrorResponse(),
    )

    with pytest.raises(RuntimeError, match="Football-API returned errors"):
        ingestion.fetch_football_api_payload("fixtures/players", {})


def test_fetch_football_api_payload_raises_quota_error_on_http_429(monkeypatch):
    class RateLimitedResponse:
        status_code = 429

        def raise_for_status(self):
            raise AssertionError("quota detection should happen before raise_for_status")

    monkeypatch.setattr(
        "football_analytics.databricks_ingestion.requests.get",
        lambda *args, **kwargs: RateLimitedResponse(),
    )

    with pytest.raises(ingestion.FootballApiQuotaError, match="429"):
        ingestion.fetch_football_api_payload("fixtures/players", {})


def test_fetch_football_api_payload_raises_quota_error_on_provider_payload(monkeypatch):
    class QuotaEnvelopeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"errors": {"requests": "You have exceeded your daily quota"}, "response": []}

    monkeypatch.setattr(
        "football_analytics.databricks_ingestion.requests.get",
        lambda *args, **kwargs: QuotaEnvelopeResponse(),
    )

    with pytest.raises(ingestion.FootballApiQuotaError, match="quota"):
        ingestion.fetch_football_api_payload("fixtures/players", {})
