import json
import logging

from football_analytics.databricks.logging import JsonFormatter


def test_json_formatter_preserves_safe_structured_fields():
    record = logging.LogRecord(
        name="football_analytics.bronze_ingest",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="fixture_endpoint_completed",
        args=(),
        exc_info=None,
    )
    record.event = "fixture_endpoint_completed"
    record.endpoint = "fixtures/players"
    record.fixture_id = 1489437
    record.latency_ms = 125
    record.status_code = 200
    record.raw_payload = {"response": ["do not log"]}

    payload = json.loads(JsonFormatter().format(record))

    assert payload["event"] == "fixture_endpoint_completed"
    assert payload["endpoint"] == "fixtures/players"
    assert payload["fixture_id"] == 1489437
    assert payload["latency_ms"] == 125
    assert payload["status_code"] == 200
    assert "raw_payload" not in payload
