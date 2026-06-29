import json
import logging
import sys
from typing import Optional


class JsonFormatter(logging.Formatter):
    SAFE_EXTRA_FIELDS = (
        "event",
        "run_id",
        "job_id",
        "task_key",
        "stage",
        "target_date",
        "endpoint",
        "fixture_id",
        "team_id",
        "index",
        "total",
        "status",
        "status_code",
        "latency_ms",
        "attempt",
        "discovered_fixtures",
        "eligible_fixtures",
        "skipped_fixtures",
        "failed_fixtures",
        "fixtures_to_fetch",
        "error",
        "params",
        "quota_headers",
    )

    def format(self, record):
        payload = {
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        for key in self.SAFE_EXTRA_FIELDS:
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_json_logging(level: int = logging.INFO, logger_name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.handlers = []
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger

