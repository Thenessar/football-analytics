import pytest
import threading
import time

from football_analytics import databricks_ingestion as ingestion


def _player_stats_payload(team_id=1):
    return {
        "response": [
            {"team": {"id": team_id}, "players": [{"player": {"id": team_id * 100}}]},
            {"team": {"id": team_id + 1}, "players": []},
        ]
    }


def test_delta_paths_match_bronze_and_silver_contract():
    assert ingestion.BRONZE_FOOTBALL_MATCH_RAW_PATH == "/mnt/syndicate/bronze/football_match_raw"
    assert ingestion.BRONZE_LINEUPS_RAW_PATH == "/mnt/syndicate/bronze/football_lineups_raw"
    assert ingestion.BRONZE_FIXTURE_STATISTICS_RAW_PATH == "/mnt/syndicate/bronze/football_fixture_statistics_raw"
    assert ingestion.INGESTION_STATE_CHECKPOINT_TABLE == "default.bronze_ingestion_state_checkpoint"


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


def test_daily_dates_are_inclusive():
    assert ingestion.iter_daily_dates("2026-06-25", "2026-06-27") == [
        "2026-06-25",
        "2026-06-26",
        "2026-06-27",
    ]


def test_default_ingestion_dates_use_last_checkpoint_through_lookahead(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "latest_completed_checkpoint_date",
        lambda spark, *, checkpoint_table=ingestion.INGESTION_STATE_CHECKPOINT_TABLE, max_target_date=None: "2026-06-25",
    )

    selection = ingestion.resolve_ingestion_dates(object(), today="2026-06-27")

    assert selection.dates == (
        "2026-06-25",
        "2026-06-26",
        "2026-06-27",
        "2026-06-28",
        "2026-06-29",
        "2026-06-30",
        "2026-07-01",
        "2026-07-02",
        "2026-07-03",
        "2026-07-04",
    )
    assert selection.mode == "checkpoint_to_lookahead"


def test_default_ingestion_dates_fall_back_to_today_without_checkpoint(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "latest_completed_checkpoint_date",
        lambda spark, *, checkpoint_table=ingestion.INGESTION_STATE_CHECKPOINT_TABLE, max_target_date=None: None,
    )

    selection = ingestion.resolve_ingestion_dates(None, today="2026-06-27")

    assert selection.dates == (
        "2026-06-27",
        "2026-06-28",
        "2026-06-29",
        "2026-06-30",
        "2026-07-01",
        "2026-07-02",
        "2026-07-03",
        "2026-07-04",
    )


def test_default_ingestion_dates_use_configured_lookahead(monkeypatch):
    monkeypatch.setattr(
        ingestion,
        "latest_completed_checkpoint_date",
        lambda spark, *, checkpoint_table=ingestion.INGESTION_STATE_CHECKPOINT_TABLE, max_target_date=None: None,
    )

    selection = ingestion.resolve_ingestion_dates(None, today="2026-06-27", lookahead_days=2)

    assert selection.dates == ("2026-06-27", "2026-06-28", "2026-06-29")


def test_default_checkpoint_lookup_ignores_future_target_dates(monkeypatch):
    captured = {}

    def fake_latest(spark, *, checkpoint_table=ingestion.INGESTION_STATE_CHECKPOINT_TABLE, max_target_date=None):
        captured["max_target_date"] = max_target_date
        return "2026-06-27"

    monkeypatch.setattr(ingestion, "latest_completed_checkpoint_date", fake_latest)

    ingestion.resolve_ingestion_dates(object(), today="2026-06-27")

    assert captured["max_target_date"] == "2026-06-27"


class _FakeSparkExpression:
    def __eq__(self, other):
        return self

    def __le__(self, other):
        return self

    def __and__(self, other):
        return self

    def isNotNull(self):
        return self

    def isin(self, values):
        return self

    def alias(self, name):
        return self


class _FakeSparkFunctions:
    def col(self, name):
        return _FakeSparkExpression()

    def lit(self, value):
        return _FakeSparkExpression()

    def to_date(self, value):
        return _FakeSparkExpression()

    def max(self, value):
        return _FakeSparkExpression()


class _LazyMissingCheckpointTable:
    def where(self, condition):
        raise RuntimeError("TABLE_OR_VIEW_NOT_FOUND")


class _FakeSparkWithLazyMissingCheckpoint:
    def table(self, name):
        return _LazyMissingCheckpointTable()


def test_latest_completed_checkpoint_date_tolerates_lazy_missing_table(monkeypatch):
    monkeypatch.setattr(ingestion, "_require_pyspark", lambda: (_FakeSparkFunctions(),))

    assert ingestion.latest_completed_checkpoint_date(_FakeSparkWithLazyMissingCheckpoint()) is None


def test_completed_checkpoint_fixture_ids_tolerates_lazy_missing_table(monkeypatch):
    monkeypatch.setattr(ingestion, "_require_pyspark", lambda: (_FakeSparkFunctions(),))

    assert ingestion.completed_checkpoint_fixture_ids(
        _FakeSparkWithLazyMissingCheckpoint(),
        endpoint=ingestion.PLAYER_STATS_ENDPOINT,
    ) == set()


def test_date_scope_log_fields_use_single_date_or_range():
    assert ingestion._date_scope_log_fields(["2026-06-25"]) == {
        "date_scope": "single_day",
        "target_date": "2026-06-25",
        "requested_dates_count": 1,
    }
    assert ingestion._date_scope_log_fields(["2026-06-25", "2026-06-26"]) == {
        "date_scope": "date_range",
        "date_from": "2026-06-25",
        "date_to": "2026-06-26",
        "requested_dates_count": 2,
    }


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


def test_fixture_business_state_classification_matches_lifecycle():
    fixture = {"fixture": {"id": 1, "status": {"short": "NS"}}}
    confirmed_lineups = {
        "response": [
            {"team": {"id": 10}, "startXI": [{"player": {"id": player_id}} for player_id in range(11)]},
            {"team": {"id": 20}, "startXI": [{"player": {"id": player_id}} for player_id in range(11, 22)]},
        ]
    }
    player_stats = {
        "response": [
            {"team": {"id": 10}, "players": [{"player": {"id": 1}}]},
            {"team": {"id": 20}, "players": [{"player": {"id": 2}}]},
        ]
    }

    assert ingestion.classify_fixture_business_state(fixture) == ingestion.FIXTURE_STATE_SCHEDULED
    assert (
        ingestion.classify_fixture_business_state(fixture, lineups_payload=confirmed_lineups)
        == ingestion.FIXTURE_STATE_LINEUPS_CONFIRMED
    )
    assert (
        ingestion.classify_fixture_business_state({"fixture": {"status": {"short": "1H"}}})
        == ingestion.FIXTURE_STATE_LIVE
    )
    assert (
        ingestion.classify_fixture_business_state(
            {"fixture": {"status": {"short": "FT"}}},
            player_stats_payload=player_stats,
        )
        == ingestion.FIXTURE_STATE_COMPLETED
    )


def test_fixture_endpoint_selection_fetches_player_stats_only_for_completed_matches():
    fixtures = [
        {"fixture": {"id": 1, "status": {"short": "NS"}}},
        {"fixture": {"id": 2, "status": {"short": "1H"}}},
        {"fixture": {"id": 3, "status": {"short": "FT"}}},
        {"fixture": {"id": 4, "status": {"short": "CANC"}}},
    ]

    assert ingestion.fixture_ids_for_player_stats(fixtures) == (3,)
    assert ingestion.fixture_ids_for_lineups(fixtures) == (1, 2, 3)
    assert ingestion.fixture_ids_for_fixture_statistics(fixtures) == (3,)


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


def test_fixture_statistics_bronze_writes_raw_envelope_with_run_context(monkeypatch):
    captured = {}
    payload = {
        "response": [
            {
                "team": {"id": 26, "name": "Argentina"},
                "statistics": [{"type": "Ball Possession", "value": "61%"}],
            }
        ]
    }

    def fake_write(spark, api_payloads, **kwargs):
        captured["api_payloads"] = tuple(api_payloads)
        captured.update(kwargs)

    monkeypatch.setattr(ingestion, "write_bronze_raw_envelopes", fake_write)

    ingestion.write_fixture_statistics_bronze(
        spark=object(),
        api_payloads=[payload],
        fixture_id=1489437,
        run_id="run-1",
    )

    assert captured["api_payloads"] == (payload,)
    assert captured["run_id"] == "run-1"
    assert captured["source_endpoint"] == ingestion.STATISTICS_ENDPOINT
    assert captured["request_params"] == {"fixture": 1489437}
    assert captured["fixture_id"] == 1489437
    assert captured["bronze_path"] == ingestion.BRONZE_FIXTURE_STATISTICS_RAW_PATH

    rows = ingestion._json_payload_rows(
        [payload],
        run_id="run-1",
        source_endpoint=ingestion.STATISTICS_ENDPOINT,
        request_params={"fixture": 1489437},
        fixture_id=1489437,
    )
    assert rows[0][0] == "run-1"
    assert rows[0][1] == 1489437
    assert rows[0][2] == "fixtures/statistics"
    assert rows[0][6] == ingestion.payload_hash(payload)


def test_player_stats_skips_completed_fixture_ids(monkeypatch):
    called_fixture_ids = []

    def fake_fetch(endpoint, params, *, api_key=None):
        called_fixture_ids.append(params["fixture"])
        return _player_stats_payload()

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
        return _player_stats_payload()

    monkeypatch.setattr(ingestion, "fetch_football_api_payload", fake_fetch)

    ingestion.ingest_player_stats_for_fixtures_to_bronze(
        spark=None,
        fixture_ids=[10, 11],
        completed_fixture_ids=[10],
        force_refresh=True,
    )

    assert called_fixture_ids == [10, 11]


def _fixture_statistics_payload(team_id=1):
    return {
        "response": [
            {
                "team": {"id": team_id, "name": "Home"},
                "statistics": [{"type": "Ball Possession", "value": "61%"}],
            },
            {
                "team": {"id": team_id + 1, "name": "Away"},
                "statistics": [{"type": "Ball Possession", "value": "39%"}],
            },
        ]
    }


def test_fixture_statistics_skips_completed_fixture_ids(monkeypatch):
    called_fixture_ids = []

    def fake_fetch(endpoint, params, *, api_key=None):
        assert endpoint == ingestion.STATISTICS_ENDPOINT
        called_fixture_ids.append(params["fixture"])
        return _fixture_statistics_payload()

    monkeypatch.setattr(ingestion, "fetch_football_api_payload", fake_fetch)

    summary = ingestion.ingest_fixture_statistics_for_fixtures_to_bronze(
        spark=None,
        fixture_ids=[10, 11],
        completed_fixture_ids=[10],
    )

    assert called_fixture_ids == [11]
    assert summary.skipped_fixtures == 1
    assert summary.statistics_ingested == 1


def test_fixture_statistics_force_refresh_refetches_completed_fixture_ids(monkeypatch):
    called_fixture_ids = []

    def fake_fetch(endpoint, params, *, api_key=None):
        called_fixture_ids.append(params["fixture"])
        return _fixture_statistics_payload()

    monkeypatch.setattr(ingestion, "fetch_football_api_payload", fake_fetch)

    ingestion.ingest_fixture_statistics_for_fixtures_to_bronze(
        spark=None,
        fixture_ids=[10, 11],
        completed_fixture_ids=[10],
        force_refresh=True,
    )

    assert called_fixture_ids == [10, 11]


def test_fixture_statistics_plan_excludes_uncompleted_fixtures():
    fixtures = [
        {"fixture": {"id": 1, "status": {"short": "NS"}}},
        {"fixture": {"id": 2, "status": {"short": "HT"}}},
        {"fixture": {"id": 3, "status": {"short": "FT"}}},
        {"fixture": {"id": 4, "status": {"short": "AET"}}},
        {"fixture": {"id": 5, "status": {"short": "PEN"}}},
        {"fixture": {"id": 6, "status": {"short": "PST"}}},
    ]

    assert ingestion.fixture_ids_for_fixture_statistics(fixtures) == (3, 4, 5)


def test_fixture_statistics_completed_checkpoint_rows_skip_api_calls(monkeypatch):
    """A COMPLETED checkpoint row for the statistics endpoint suppresses refetching."""
    called_fixture_ids = []
    checkpoint_queries = []

    def fake_completed_ids(spark, *, endpoint, target_date=None, checkpoint_table=None):
        checkpoint_queries.append({"endpoint": endpoint, "target_date": target_date})
        return {10}

    def fake_fetch(endpoint, params, *, api_key=None, logger=None):
        called_fixture_ids.append(params["fixture"])
        return _fixture_statistics_payload()

    monkeypatch.setattr(ingestion, "_supports_spark_sql", lambda spark: True)
    monkeypatch.setattr(ingestion, "completed_checkpoint_fixture_ids", fake_completed_ids)
    monkeypatch.setattr(ingestion, "_fetch_payload_with_optional_logger", fake_fetch)
    monkeypatch.setattr(ingestion, "write_fixture_statistics_bronze", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoint", lambda spark, **kwargs: None)
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoints", lambda *args, **kwargs: None)

    summary = ingestion.ingest_fixture_statistics_for_fixtures_to_bronze(
        spark=object(),
        fixture_ids=[10, 11],
        target_date="2026-06-25",
    )

    assert checkpoint_queries == [
        {"endpoint": ingestion.STATISTICS_ENDPOINT, "target_date": "2026-06-25"}
    ]
    assert called_fixture_ids == [11]
    assert summary.statistics_skipped == 1
    assert summary.statistics_ingested == 1


def test_fixture_statistics_refresh_lands_changed_payload_and_new_hash(monkeypatch):
    """force_refresh refetches a completed fixture; a changed payload is rewritten
    to Bronze and the checkpoint hash is updated to the new response_hash."""
    old_payload = _fixture_statistics_payload(team_id=1)
    new_payload = _fixture_statistics_payload(team_id=99)
    writes = []
    checkpoint_records = []

    monkeypatch.setattr(ingestion, "_supports_spark_sql", lambda spark: True)
    monkeypatch.setattr(
        ingestion,
        "_fetch_payload_with_optional_logger",
        lambda endpoint, params, *, api_key=None, logger=None: new_payload,
    )
    monkeypatch.setattr(
        ingestion,
        "write_fixture_statistics_bronze",
        lambda spark, api_payloads, **kwargs: writes.append((tuple(api_payloads), kwargs)),
    )
    monkeypatch.setattr(
        ingestion,
        "upsert_endpoint_checkpoint",
        lambda spark, **kwargs: checkpoint_records.append(kwargs),
    )
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoints", lambda *args, **kwargs: None)

    summary = ingestion.ingest_fixture_statistics_for_fixtures_to_bronze(
        spark=object(),
        fixture_ids=[10],
        completed_fixture_ids=[10],
        force_refresh=True,
    )

    assert writes[0][0] == (new_payload,)
    completed = [
        record for record in checkpoint_records
        if record["status"] == ingestion.CHECKPOINT_COMPLETED
    ]
    assert completed[-1]["response_hash"] == ingestion.payload_hash(new_payload)
    assert completed[-1]["response_hash"] != ingestion.payload_hash(old_payload)
    assert summary.statistics_ingested == 1
    assert summary.skipped_fixtures == 0


def test_fixture_statistics_empty_payload_is_skipped_for_retry(monkeypatch):
    checkpoint_records = []

    monkeypatch.setattr(ingestion, "_supports_spark_sql", lambda spark: True)
    monkeypatch.setattr(
        ingestion,
        "fetch_football_api_payload",
        lambda endpoint, params, *, api_key=None, logger=None: {"response": []},
    )
    monkeypatch.setattr(ingestion, "write_fixture_statistics_bronze", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ingestion,
        "upsert_endpoint_checkpoint",
        lambda spark, **kwargs: checkpoint_records.append(kwargs),
    )
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoints", lambda *args, **kwargs: None)

    summary = ingestion.ingest_fixture_statistics_for_fixtures_to_bronze(
        spark=object(),
        fixture_ids=[10],
        completed_fixture_ids=[],
    )

    assert checkpoint_records[-1]["status"] == ingestion.CHECKPOINT_SKIPPED
    assert checkpoint_records[-1]["response_hash"] == ingestion.payload_hash({"response": []})
    assert summary.statistics_ingested == 0
    assert summary.statistics_skipped == 1


def test_player_stats_empty_payload_is_skipped_for_retry(monkeypatch):
    checkpoint_records = []

    monkeypatch.setattr(ingestion, "_supports_spark_sql", lambda spark: True)
    monkeypatch.setattr(
        ingestion,
        "fetch_football_api_payload",
        lambda endpoint, params, *, api_key=None, logger=None: {"response": []},
    )
    monkeypatch.setattr(ingestion, "write_bronze_raw_envelopes", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ingestion,
        "upsert_endpoint_checkpoint",
        lambda spark, **kwargs: checkpoint_records.append(kwargs),
    )
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoints", lambda *args, **kwargs: None)

    summary = ingestion.ingest_player_stats_for_fixtures_to_bronze(
        spark=object(),
        fixture_ids=[10],
        completed_fixture_ids=[],
    )

    assert checkpoint_records[-1]["status"] == ingestion.CHECKPOINT_SKIPPED
    assert summary.player_stat_payloads_ingested == 0
    assert summary.skipped_fixtures == 1


def test_player_stats_logs_progress_and_pending_checkpoints(monkeypatch):
    log_records = []
    checkpoint_records = []

    class FakeLogger:
        def log(self, level, message, extra=None):
            log_records.append((level, message, extra or {}))

    def fake_fetch(endpoint, params, *, api_key=None, logger=None):
        if params["fixture"] == 12:
            raise RuntimeError("temporary failure\nwithout payload")
        return _player_stats_payload()

    def fake_upsert(spark, **kwargs):
        checkpoint_records.append((kwargs["fixture_id"], kwargs["endpoint"], kwargs["status"]))

    def fake_batch_upsert(spark, checkpoint_rows, *, checkpoint_table=ingestion.INGESTION_STATE_CHECKPOINT_TABLE):
        for row in checkpoint_rows:
            checkpoint_records.append((row["fixture_id"], row["endpoint"], row["status"]))

    monkeypatch.setattr(ingestion, "_supports_spark_sql", lambda spark: True)
    monkeypatch.setattr(ingestion, "fetch_football_api_payload", fake_fetch)
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoint", fake_upsert)
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoints", fake_batch_upsert)
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
    assert "endpoint_fetch_batch_completed" in events
    assert failed["fixture_id"] == 12
    assert failed["error"] == "temporary failure without payload"
    batch_completed = next(record[2] for record in log_records if record[1] == "endpoint_fetch_batch_completed")
    assert batch_completed["target_date"] == "2026-06-25"
    assert batch_completed["endpoint"] == ingestion.PLAYER_STATS_ENDPOINT
    assert batch_completed["total"] == 2
    assert batch_completed["successful_fetches"] == 1
    assert batch_completed["failed_fetches"] == 1
    assert checkpoint_records == [
        (10, ingestion.PLAYER_STATS_ENDPOINT, ingestion.CHECKPOINT_SKIPPED),
        (11, ingestion.PLAYER_STATS_ENDPOINT, ingestion.CHECKPOINT_PENDING),
        (12, ingestion.PLAYER_STATS_ENDPOINT, ingestion.CHECKPOINT_PENDING),
        (11, ingestion.PLAYER_STATS_ENDPOINT, ingestion.CHECKPOINT_COMPLETED),
        (12, ingestion.PLAYER_STATS_ENDPOINT, ingestion.CHECKPOINT_FAILED),
    ]
    assert summary.player_stat_payloads_ingested == 1
    assert summary.failed_fixtures == 1


def test_player_stats_parallel_fetches_with_configured_workers(monkeypatch):
    active_fetches = 0
    peak_fetches = 0
    lock = threading.Lock()

    def fake_fetch(endpoint, params, *, api_key=None, logger=None):
        nonlocal active_fetches, peak_fetches
        with lock:
            active_fetches += 1
            peak_fetches = max(peak_fetches, active_fetches)
        time.sleep(0.02)
        with lock:
            active_fetches -= 1
        return _player_stats_payload(params["fixture"])

    monkeypatch.setattr(ingestion, "_supports_spark_sql", lambda spark: False)
    monkeypatch.setattr(ingestion, "_fetch_payload_with_optional_logger", fake_fetch)

    summary = ingestion.ingest_player_stats_for_fixtures_to_bronze(
        spark=object(),
        fixture_ids=[10, 11, 12, 13],
        endpoint_max_workers=4,
    )

    assert peak_fetches > 1
    assert summary.fixture_ids == (10, 11, 12, 13)
    assert summary.player_stat_payloads_ingested == 4


def test_parallel_fetch_batches_pending_checkpoints(monkeypatch):
    batch_sizes = []

    def fake_batch_upsert(spark, checkpoint_rows, *, checkpoint_table=ingestion.INGESTION_STATE_CHECKPOINT_TABLE):
        rows = list(checkpoint_rows)
        batch_sizes.append(len(rows))
        assert {row["status"] for row in rows} == {ingestion.CHECKPOINT_PENDING}

    monkeypatch.setattr(ingestion, "_supports_spark_sql", lambda spark: True)
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoints", fake_batch_upsert)
    monkeypatch.setattr(
        ingestion,
        "_fetch_payload_with_optional_logger",
        lambda endpoint, params, *, api_key=None, logger=None: {"response": []},
    )
    monkeypatch.setattr(ingestion, "write_bronze_raw_envelopes", lambda *args, **kwargs: None)
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoint", lambda *args, **kwargs: None)

    ingestion.ingest_player_stats_for_fixtures_to_bronze(
        spark=object(),
        fixture_ids=[10, 11, 12, 13],
        completed_fixture_ids=[],
        endpoint_max_workers=4,
    )

    assert batch_sizes == [4]


def test_api_rate_limiter_spaces_parallel_requests(monkeypatch):
    class FakeClock:
        def __init__(self):
            self.current = 0.0
            self.sleeps = []

        def monotonic(self):
            return self.current

        def sleep(self, delay_seconds):
            self.sleeps.append(delay_seconds)
            self.current += delay_seconds

    clock = FakeClock()
    monkeypatch.setattr(ingestion.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(ingestion.time, "sleep", clock.sleep)
    limiter = ingestion.ApiRateLimiter(calls_per_minute=6000)

    limiter.wait()
    limiter.wait()
    limiter.wait()

    assert clock.sleeps == [0.01, 0.01]


def test_medallion_bronze_calls_player_stats_only_for_filtered_fixtures(monkeypatch):
    discovery = ingestion.FixtureDiscoveryResult(
        target_date="2026-06-25",
        raw_payload={"response": [{"fixture": {"id": 100}}, {"fixture": {"id": 200}}]},
        eligible_fixtures=({"fixture": {"id": 100, "status": {"short": "FT"}}},),
        skipped_fixtures=({"fixture": {"id": 200}},),
    )
    player_fixture_ids = []
    statistics_fixture_ids = []

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

    def fake_statistics_ingest(spark, fixture_ids, **kwargs):
        statistics_fixture_ids.extend(fixture_ids)
        return ingestion.BronzeIngestionSummary(
            requested_dates=(),
            discovered_fixtures=len(fixture_ids),
            ingested_fixtures=len(fixture_ids),
            skipped_fixtures=0,
            failed_fixtures=0,
            fixture_ids=tuple(fixture_ids),
            statistics_ingested=len(fixture_ids),
        )

    monkeypatch.setattr(ingestion, "ingest_player_stats_for_fixtures_to_bronze", fake_player_ingest)
    monkeypatch.setattr(
        ingestion, "ingest_fixture_statistics_for_fixtures_to_bronze", fake_statistics_ingest
    )

    summary = ingestion.ingest_senior_mens_international_bronze(
        spark=None,
        target_date="2026-06-25",
        include_lineups=False,
    )

    assert player_fixture_ids == [100]
    assert statistics_fixture_ids == [100]
    assert summary.discovered_fixtures == 2
    assert summary.eligible_fixtures == 1
    assert summary.statistics_ingested == 1


def test_medallion_bronze_keeps_scheduled_fixture_and_skips_missing_lineups(monkeypatch):
    discovery_kwargs = {}
    discovery = ingestion.FixtureDiscoveryResult(
        target_date="2026-06-25",
        raw_payload={"response": [{"fixture": {"id": 100, "status": {"short": "NS"}}}]},
        eligible_fixtures=({"fixture": {"id": 100, "status": {"short": "NS"}}},),
        skipped_fixtures=(),
    )
    player_fixture_ids = []

    def fake_discover(spark, match_date, **kwargs):
        discovery_kwargs.update(kwargs)
        return discovery

    monkeypatch.setattr(ingestion, "discover_senior_mens_fixtures_for_date", fake_discover)

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
    monkeypatch.setattr(
        ingestion,
        "ingest_lineups_for_fixtures_to_bronze",
        lambda spark, fixture_ids, **kwargs: ingestion.BronzeIngestionSummary(
            requested_dates=(),
            discovered_fixtures=len(fixture_ids),
            ingested_fixtures=0,
            skipped_fixtures=len(fixture_ids),
            failed_fixtures=0,
            fixture_ids=(),
            eligible_fixtures=len(fixture_ids),
            lineups_ingested=0,
            lineups_skipped=len(fixture_ids),
        ),
    )

    summary = ingestion.ingest_senior_mens_international_bronze(
        spark=None,
        target_date="2026-06-25",
    )

    assert discovery_kwargs["completed_only"] is False
    assert player_fixture_ids == []
    assert summary.fixture_ids == (100,)
    assert summary.lineups_skipped == 1
    assert summary.failed_fixtures == 0


def test_medallion_bronze_logs_single_date_or_date_range(monkeypatch):
    log_records = []

    class FakeLogger:
        def log(self, level, message, extra=None):
            log_records.append((message, extra or {}))

    def fake_discover(spark, match_date, **kwargs):
        return ingestion.FixtureDiscoveryResult(
            target_date=match_date,
            raw_payload={"response": []},
            eligible_fixtures=(),
            skipped_fixtures=(),
        )

    monkeypatch.setattr(ingestion, "discover_senior_mens_fixtures_for_date", fake_discover)
    monkeypatch.setattr(
        ingestion,
        "ingest_player_stats_for_fixtures_to_bronze",
        lambda *args, **kwargs: ingestion.BronzeIngestionSummary(
            requested_dates=(),
            discovered_fixtures=0,
            ingested_fixtures=0,
            skipped_fixtures=0,
            failed_fixtures=0,
            fixture_ids=(),
        ),
    )

    ingestion.ingest_senior_mens_international_bronze(
        spark=None,
        date_from="2026-06-25",
        date_to="2026-06-26",
        include_lineups=False,
        logger=FakeLogger(),
    )

    started = next(extra for message, extra in log_records if message == "bronze_ingest_started")
    completed = next(extra for message, extra in log_records if message == "bronze_ingest_completed")

    assert started["date_scope"] == "date_range"
    assert started["date_from"] == "2026-06-25"
    assert started["date_to"] == "2026-06-26"
    assert started["requested_dates_count"] == 2
    assert completed["date_from"] == "2026-06-25"
    assert completed["date_to"] == "2026-06-26"


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

    def fake_batch_upsert(spark, checkpoint_rows, *, checkpoint_table=ingestion.INGESTION_STATE_CHECKPOINT_TABLE):
        for row in checkpoint_rows:
            operations.append(("checkpoint", row["fixture_id"], row["status"]))

    monkeypatch.setattr(ingestion, "_supports_spark_sql", lambda spark: True)
    monkeypatch.setattr(ingestion, "fetch_football_api_payload", fake_fetch)
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoint", fake_upsert)
    monkeypatch.setattr(ingestion, "upsert_endpoint_checkpoints", fake_batch_upsert)
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
        checkpoint_table="football_analytics.bronze.ingestion_state_checkpoint",
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
