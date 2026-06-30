import pytest

from football_analytics import databricks_ingestion as ingestion


def test_delta_paths_match_bronze_and_silver_contract():
    assert ingestion.BRONZE_FOOTBALL_MATCH_RAW_PATH == "/mnt/syndicate/bronze/football_match_raw"
    assert ingestion.SILVER_PLAYER_MATCH_STATS_PATH == "/mnt/syndicate/silver/football_player_match_stats"
    assert ingestion.INGESTION_STATE_CHECKPOINT_TABLE == "default.ingestion_state_checkpoint"


def test_delta_target_detection_distinguishes_tables_from_paths():
    assert ingestion.is_delta_table_target("football_analytics.bronze.football_match_raw")
    assert not ingestion.is_delta_table_target("/mnt/syndicate/bronze/football_match_raw")
    assert not ingestion.is_delta_table_target("dbfs:/mnt/syndicate/bronze/football_match_raw")


def test_weekly_windows_are_deterministic_from_anchor():
    assert ingestion.iter_weekly_windows("2022-11-07", "2022-11-20") == [
        ("2022-11-07", "2022-11-13"),
        ("2022-11-14", "2022-11-20"),
    ]


def test_weekly_windows_default_to_post_2022_world_cup_anchor():
    assert ingestion.iter_weekly_windows(end_date="2023-01-05") == [
        ("2022-12-23", "2022-12-29"),
        ("2022-12-30", "2023-01-05"),
    ]


def test_fifa_rankings_seed_file_is_versioned_with_expected_source_columns():
    seed_path = "data/seeds/fifa_mens_world_ranking_december_2022.csv"
    with open(seed_path, "r", encoding="utf-8") as seed_file:
        assert seed_file.readline().strip() == "Rank,Team,Raiting"
        assert seed_file.readline().strip().startswith("1,Brazil,")


def test_daily_dates_are_inclusive():
    assert ingestion.iter_daily_dates("2026-06-25", "2026-06-27") == [
        "2026-06-25",
        "2026-06-26",
        "2026-06-27",
    ]


def test_legacy_world_cup_fixture_wrapper_filters_non_world_cup(monkeypatch):
    def fake_fetch(endpoint, params, *, api_key=None):
        assert endpoint == "fixtures"
        assert params == {"date": "2026-06-25", "timezone": "UTC"}
        return {
            "response": [
                {
                    "fixture": {"id": 1489437, "status": {"short": "FT"}},
                    "league": {"id": 1, "season": 2026},
                },
                {
                    "fixture": {"id": 1036663, "status": {"short": "FT"}},
                    "league": {"id": 667, "season": 2023},
                },
            ]
        }

    monkeypatch.setattr(ingestion, "fetch_football_api_payload", fake_fetch)

    fixtures = ingestion.fetch_world_cup_fixtures_for_date("2026-06-25")

    assert [(fixture["fixture"]["id"]) for fixture in fixtures] == [1489437]


def test_fetch_senior_mens_international_fixtures_for_date_keeps_national_competitions(monkeypatch):
    def fake_fetch(endpoint, params, *, api_key=None):
        return {
            "response": [
                {
                    "fixture": {"id": 1, "status": {"short": "FT"}},
                    "league": {"id": 10, "season": 2024, "name": "Friendlies"},
                },
                {
                    "fixture": {"id": 2, "status": {"short": "FT"}},
                    "league": {"id": 960, "season": 2024, "name": "Euro Championship - Qualification"},
                },
                {
                    "fixture": {"id": 3, "status": {"short": "FT"}},
                    "league": {"id": 667, "season": 2024, "name": "Friendlies Clubs"},
                },
            ]
        }

    monkeypatch.setattr(ingestion, "fetch_football_api_payload", fake_fetch)

    fixtures = ingestion.fetch_senior_mens_international_fixtures_for_date("2024-06-01")

    assert [fixture["fixture"]["id"] for fixture in fixtures] == [1, 2]


def test_senior_mens_international_player_stats_bronze_discovers_fixture_range(monkeypatch):
    discovered_dates = []
    ingested_fixture_ids = []

    def fake_discover(match_date, *, api_key=None, completed_only=True):
        discovered_dates.append(match_date)
        return [{
            "fixture": {"id": 1000 + len(discovered_dates), "status": {"short": "FT"}},
            "league": {"id": 1, "season": 2026},
        }]

    def fake_ingest(spark, fixture_ids, *, api_key=None, bronze_path=ingestion.BRONZE_FOOTBALL_MATCH_RAW_PATH):
        ingested_fixture_ids.extend(fixture_ids)
        return ingestion.BronzeIngestionSummary(
            requested_dates=(),
            discovered_fixtures=len(fixture_ids),
            ingested_fixtures=len(fixture_ids),
            skipped_fixtures=0,
            failed_fixtures=0,
            fixture_ids=tuple(fixture_ids),
        )

    monkeypatch.setattr(ingestion, "fetch_senior_mens_international_fixtures_for_date", fake_discover)
    monkeypatch.setattr(ingestion, "ingest_player_stats_for_fixtures_to_bronze", fake_ingest)

    summary = ingestion.ingest_senior_mens_international_player_stats_bronze(
        spark=object(),
        date_from="2026-06-25",
        date_to="2026-06-26",
    )

    assert discovered_dates == ["2026-06-25", "2026-06-26"]
    assert ingested_fixture_ids == [1001, 1002]
    assert summary.as_dict()["ingested_fixtures"] == 2


def test_legacy_world_cup_player_stats_wrapper_delegates_to_senior_international_loader(monkeypatch):
    captured = {}

    def fake_loader(spark, **kwargs):
        captured["spark"] = spark
        captured["kwargs"] = kwargs
        return ingestion.BronzeIngestionSummary(
            requested_dates=("2026-06-25",),
            discovered_fixtures=0,
            ingested_fixtures=0,
            skipped_fixtures=0,
            failed_fixtures=0,
            fixture_ids=(),
        )

    monkeypatch.setattr(ingestion, "ingest_senior_mens_international_player_stats_bronze", fake_loader)

    summary = ingestion.ingest_world_cup_player_stats_bronze(
        spark=object(),
        target_date="2026-06-25",
        completed_only=False,
    )

    assert summary.requested_dates == ("2026-06-25",)
    assert captured["kwargs"]["target_date"] == "2026-06-25"
    assert captured["kwargs"]["completed_only"] is False


def test_endpoint_ingestion_plan_skips_completed_unless_forced():
    plan = ingestion.endpoint_ingestion_plan([101, 102, 102, 103], completed_fixture_ids=[102])

    assert plan.fixture_ids_to_fetch == (101, 103)
    assert plan.skipped_fixture_ids == (102,)

    forced = ingestion.endpoint_ingestion_plan([101, 102], completed_fixture_ids=[102], force_refresh=True)

    assert forced.fixture_ids_to_fetch == (101, 102)
    assert forced.skipped_fixture_ids == ()


def test_senior_mens_fixture_filter_excludes_club_women_youth_and_keeps_allowed():
    payload = {
        "response": [
            {"fixture": {"id": 1, "status": {"short": "FT"}}, "league": {"id": 1, "name": "World Cup"}},
            {"fixture": {"id": 2, "status": {"short": "FT"}}, "league": {"id": 667, "name": "Club Friendlies"}},
            {"fixture": {"id": 3, "status": {"short": "FT"}}, "league": {"name": "World Cup - Women"}},
            {"fixture": {"id": 4, "status": {"short": "FT"}}, "league": {"id": 5, "name": "UEFA Nations League"}},
            {"fixture": {"id": 5, "status": {"short": "FT"}}, "league": {"name": "U21 Championship"}},
        ]
    }

    eligible, skipped = ingestion.split_senior_mens_international_fixtures(payload)

    assert [item["fixture"]["id"] for item in eligible] == [1, 4]
    assert [item["fixture"]["id"] for item in skipped] == [2, 3, 5]


def test_bronze_fixture_metadata_rows_include_request_hash_and_run_context():
    payload = {"response": [{"fixture": {"id": 1}}]}

    rows = ingestion._json_payload_rows(
        [payload],
        run_id="run-1",
        source_endpoint=ingestion.FIXTURES_ENDPOINT,
        request_params={"date": "2026-06-25", "timezone": "UTC"},
        target_date="2026-06-25",
    )

    assert rows[0][0] == "run-1"
    assert rows[0][2] == "fixtures"
    assert rows[0][4] == "2026-06-25"
    assert rows[0][6] == ingestion.payload_hash(payload)


def test_player_stats_skips_completed_fixture_ids(monkeypatch):
    called_fixture_ids = []

    def fake_fetch(endpoint, params, *, api_key=None):
        called_fixture_ids.append(params["fixture"])
        return {"response": [{"team": {"id": 1}, "players": []}]}

    monkeypatch.setattr(ingestion, "fetch_football_api_payload", fake_fetch)

    summary = ingestion.ingest_player_stats_for_fixtures_to_bronze(
        spark=None,
        fixture_ids=[10, 11],
        completed_fixture_ids=[10],
    )

    assert called_fixture_ids == [11]
    assert summary.skipped_fixtures == 1
    assert summary.player_stat_payloads_ingested == 1


def test_player_stats_force_refresh_refetches_completed_fixture_ids(monkeypatch):
    called_fixture_ids = []

    def fake_fetch(endpoint, params, *, api_key=None):
        called_fixture_ids.append(params["fixture"])
        return {"response": [{"team": {"id": 1}, "players": []}]}

    monkeypatch.setattr(ingestion, "fetch_football_api_payload", fake_fetch)

    ingestion.ingest_player_stats_for_fixtures_to_bronze(
        spark=None,
        fixture_ids=[10, 11],
        completed_fixture_ids=[10],
        force_refresh=True,
    )

    assert called_fixture_ids == [10, 11]


def test_player_stats_logs_progress_and_pending_checkpoints(monkeypatch):
    log_records = []
    checkpoint_records = []

    class FakeLogger:
        def log(self, level, message, extra=None):
            log_records.append((level, message, extra or {}))

    def fake_fetch(endpoint, params, *, api_key=None, logger=None):
        if params["fixture"] == 12:
            raise RuntimeError("temporary failure\nwithout payload")
        return {"response": [{"team": {"id": 1}, "players": []}]}

    def fake_upsert(spark, **kwargs):
        checkpoint_records.append((kwargs["fixture_id"], kwargs["endpoint"], kwargs["status"]))

    monkeypatch.setattr(ingestion, "_supports_spark_sql", lambda spark: True)
    monkeypatch.setattr(ingestion, "fetch_football_api_payload", fake_fetch)
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoint", fake_upsert)
    monkeypatch.setattr(ingestion, "write_bronze_raw_envelopes", lambda *args, **kwargs: None)

    summary = ingestion.ingest_player_stats_for_fixtures_to_bronze(
        spark=object(),
        fixture_ids=[10, 11, 12],
        completed_fixture_ids=[10],
        target_date="2026-06-25",
        run_id="run-1",
        logger=FakeLogger(),
    )

    events = [record[1] for record in log_records]
    plan = next(record[2] for record in log_records if record[1] == "endpoint_plan_created")
    failed = next(record[2] for record in log_records if record[1] == "fixture_endpoint_failed")

    assert plan["endpoint"] == ingestion.PLAYER_STATS_ENDPOINT
    assert plan["total"] == 3
    assert plan["skipped_fixtures"] == 1
    assert plan["fixtures_to_fetch"] == 2
    assert "fixture_endpoint_skipped" in events
    assert "fixture_endpoint_started" in events
    assert "fixture_endpoint_completed" in events
    assert failed["fixture_id"] == 12
    assert failed["error"] == "temporary failure without payload"
    assert checkpoint_records == [
        (10, ingestion.PLAYER_STATS_ENDPOINT, ingestion.CHECKPOINT_SKIPPED),
        (11, ingestion.PLAYER_STATS_ENDPOINT, ingestion.CHECKPOINT_PENDING),
        (11, ingestion.PLAYER_STATS_ENDPOINT, ingestion.CHECKPOINT_COMPLETED),
        (12, ingestion.PLAYER_STATS_ENDPOINT, ingestion.CHECKPOINT_PENDING),
        (12, ingestion.PLAYER_STATS_ENDPOINT, ingestion.CHECKPOINT_FAILED),
    ]
    assert summary.player_stat_payloads_ingested == 1
    assert summary.failed_fixtures == 1


def test_medallion_bronze_calls_player_stats_only_for_filtered_fixtures(monkeypatch):
    discovery = ingestion.FixtureDiscoveryResult(
        target_date="2026-06-25",
        raw_payload={"response": [{"fixture": {"id": 100}}, {"fixture": {"id": 200}}]},
        eligible_fixtures=({"fixture": {"id": 100}},),
        skipped_fixtures=({"fixture": {"id": 200}},),
    )
    player_fixture_ids = []

    monkeypatch.setattr(
        ingestion,
        "discover_senior_mens_fixtures_for_date",
        lambda *args, **kwargs: discovery,
    )

    def fake_player_ingest(spark, fixture_ids, **kwargs):
        player_fixture_ids.extend(fixture_ids)
        return ingestion.BronzeIngestionSummary(
            requested_dates=(),
            discovered_fixtures=len(fixture_ids),
            ingested_fixtures=len(fixture_ids),
            skipped_fixtures=0,
            failed_fixtures=0,
            fixture_ids=tuple(fixture_ids),
            player_stat_payloads_ingested=len(fixture_ids),
        )

    monkeypatch.setattr(ingestion, "ingest_player_stats_for_fixtures_to_bronze", fake_player_ingest)

    summary = ingestion.ingest_senior_mens_international_bronze(
        spark=None,
        target_date="2026-06-25",
        include_lineups=False,
    )

    assert player_fixture_ids == [100]
    assert summary.discovered_fixtures == 2
    assert summary.eligible_fixtures == 1


def test_lineup_empty_response_is_skipped_not_failed(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "fetch_football_api_payload",
        lambda endpoint, params, *, api_key=None: {"response": []},
    )

    summary = ingestion.ingest_lineups_for_fixtures_to_bronze(
        spark=None,
        fixture_ids=[1489437],
    )

    assert summary.lineups_ingested == 0
    assert summary.lineups_skipped == 1
    assert summary.failed_fixtures == 0


def test_lineups_write_pending_checkpoint_before_fetch(monkeypatch):
    operations = []

    def fake_fetch(endpoint, params, *, api_key=None, logger=None):
        operations.append(("fetch", params["fixture"]))
        return {"response": [{"team": {"id": 1}, "formation": "4-3-3"}]}

    def fake_upsert(spark, **kwargs):
        operations.append(("checkpoint", kwargs["fixture_id"], kwargs["status"]))

    monkeypatch.setattr(ingestion, "_supports_spark_sql", lambda spark: True)
    monkeypatch.setattr(ingestion, "fetch_football_api_payload", fake_fetch)
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoint", fake_upsert)
    monkeypatch.setattr(ingestion, "write_lineups_bronze", lambda *args, **kwargs: None)

    ingestion.ingest_lineups_for_fixtures_to_bronze(
        spark=object(),
        fixture_ids=[1489437],
        completed_fixture_ids=[],
    )

    assert operations[:2] == [
        ("checkpoint", 1489437, ingestion.CHECKPOINT_PENDING),
        ("fetch", 1489437),
    ]
    assert operations[-1] == ("checkpoint", 1489437, ingestion.CHECKPOINT_COMPLETED)


def test_delta_merge_sql_uses_natural_key_predicate_and_updates_non_keys():
    sql = ingestion.build_delta_merge_sql(
        "delta.`/tmp/silver`",
        "_updates",
        ("fixture_id", "team_id", "player_id"),
        ("fixture_id", "team_id", "player_id", "shots_total", "updated_at_utc"),
    )

    assert "target.fixture_id <=> source.fixture_id" in sql
    assert "shots_total = source.shots_total" in sql
    assert "WHEN NOT MATCHED THEN INSERT" in sql


def test_managed_table_merge_bootstrap_checks_catalog_before_reading():
    class ExplodingSpark:
        class Catalog:
            def tableExists(self, table_name):
                assert table_name == "football_analytics.silver.football_lineups"
                return False

        catalog = Catalog()

        def table(self, table_name):
            raise AssertionError("missing managed tables should not be read before bootstrap")

        def sql(self, statement):
            raise AssertionError("first-run bootstrap should write, not merge")

    class CapturingWriter:
        def __init__(self):
            self.saved_table = None
            self.options = {}

        def format(self, value):
            assert value == "delta"
            return self

        def mode(self, value):
            assert value == "overwrite"
            return self

        def option(self, key, value):
            self.options[key] = value
            return self

        def saveAsTable(self, table_name):
            self.saved_table = table_name

    class CapturingDataFrame:
        def __init__(self):
            self.write = CapturingWriter()

        @property
        def columns(self):
            return ["fixture_id", "team_id", "player_id"]

        def createOrReplaceTempView(self, name):
            raise AssertionError("first-run bootstrap should not create a merge view")

    spark = ExplodingSpark()
    dataframe = CapturingDataFrame()

    ingestion.merge_dataframe_to_delta_path(
        spark,
        dataframe,
        target_path="football_analytics.silver.football_lineups",
        keys=("fixture_id", "team_id", "player_id"),
        temp_view="_silver_lineups_updates",
    )

    assert dataframe.write.saved_table == "football_analytics.silver.football_lineups"
    assert dataframe.write.options == {"overwriteSchema": "true"}


def test_fifa_seed_rows_are_typed_and_rating_typo_is_normalized():
    rows = ingestion.read_fifa_rankings_seed_rows()

    assert rows[0]["rank"] == 1
    assert rows[0]["team_name"] == "Brazil"
    assert isinstance(rows[0]["rating"], float)
    assert rows[0]["ranking_as_of_date"] == "2022-12-22"


def test_fifa_seed_table_writer_uses_python_rows_not_spark_csv():
    class ExplodingSparkReader:
        def option(self, *args, **kwargs):
            raise AssertionError("seed writer should not call spark.read.csv")

    class CapturingWriter:
        def __init__(self):
            self.saved_table = None

        def format(self, value):
            assert value == "delta"
            return self

        def mode(self, value):
            assert value == "overwrite"
            return self

        def option(self, key, value):
            assert (key, value) == ("overwriteSchema", "true")
            return self

        def saveAsTable(self, table_name):
            self.saved_table = table_name

    class CapturingDataFrame:
        def __init__(self, rows):
            self.rows = rows
            self.write = CapturingWriter()

    class CapturingSpark:
        read = ExplodingSparkReader()

        def __init__(self):
            self.created = None
            self.schema = None

        def createDataFrame(self, rows, schema):
            self.created = CapturingDataFrame(rows)
            self.schema = schema
            return self.created

    spark = CapturingSpark()

    seed_df = ingestion.write_fifa_rankings_seed_table(spark, table_name="default.seed_test")

    assert seed_df.write.saved_table == "default.seed_test"
    assert spark.schema == "rank int, team_name string, rating double, ranking_as_of_date date"
    assert spark.created.rows[0]["team_name"] == "Brazil"


def test_ingest_fixture_metadata_lands_fixture_id_bronze_payload(monkeypatch):
    captured = {"checkpoints": []}
    payload = {"response": [{"fixture": {"id": 1489437}}]}

    def fake_fetch(endpoint, params, *, api_key=None, logger=None):
        captured["fetch"] = {
            "endpoint": endpoint,
            "params": params,
            "api_key": api_key,
            "logger": logger,
        }
        return payload

    def fake_write(spark, api_payloads, **kwargs):
        captured["write"] = {
            "spark": spark,
            "api_payloads": tuple(api_payloads),
            **kwargs,
        }

    def fake_checkpoint(spark, **kwargs):
        captured["checkpoints"].append(kwargs)

    monkeypatch.setattr(ingestion, "_fetch_payload_with_optional_logger", fake_fetch)
    monkeypatch.setattr(ingestion, "write_bronze_raw_envelopes", fake_write)
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoint", fake_checkpoint)

    result = ingestion.ingest_fixture_metadata_to_bronze(
        spark=object(),
        fixture_id=1489437,
        api_key="secret",
        run_id="run-1",
        target_date="2026-06-25",
        bronze_path="football_analytics.bronze.football_fixtures_raw",
        checkpoint_table="football_analytics.ops.ingestion_state_checkpoint",
    )

    assert result == payload
    assert captured["fetch"]["endpoint"] == ingestion.FIXTURES_ENDPOINT
    assert captured["fetch"]["params"] == {"id": 1489437}
    assert captured["write"]["api_payloads"] == (payload,)
    assert captured["write"]["request_params"] == {"id": 1489437}
    assert captured["write"]["fixture_id"] == 1489437
    assert captured["write"]["target_date"] == "2026-06-25"
    assert captured["write"]["bronze_path"] == "football_analytics.bronze.football_fixtures_raw"
    assert [row["status"] for row in captured["checkpoints"]] == [
        ingestion.CHECKPOINT_PENDING,
        ingestion.CHECKPOINT_COMPLETED,
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
