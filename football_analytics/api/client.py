import hashlib
import json
import logging
import time
from typing import Callable, Mapping, Optional

import requests

from football_analytics.api.exceptions import (
    FootballApiError,
    FootballApiPayloadError,
    FootballApiQuotaError,
    FootballApiTransientError,
)
from football_analytics.api.retry import RETRYABLE_STATUS_CODES, backoff_delay, sleep_before_retry
from football_analytics.config import BASE_URL, HEADERS

QUOTA_ERROR_TOKENS = (
    "rate limit",
    "too many request",
    "too many requests",
    "quota",
    "requests limit",
    "request limit",
    "subscription",
    "exceeded",
)


def payload_hash(payload: Mapping) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_quota_error_payload(payload: Mapping) -> bool:
    errors = payload.get("errors") if isinstance(payload, Mapping) else None
    if not errors:
        return False
    text = json.dumps(errors, ensure_ascii=False).casefold()
    return any(token in text for token in QUOTA_ERROR_TOKENS)


class FootballApiClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = BASE_URL,
        headers: Optional[Mapping[str, str]] = None,
        timeout_seconds: int = 30,
        max_attempts: int = 4,
        sleep: Callable[[float], None] = sleep_before_retry,
        request_get: Callable = requests.get,
        logger: Optional[logging.Logger] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.request_get = request_get
        self.logger = logger or logging.getLogger(__name__)
        self.headers = dict(headers or HEADERS)
        if api_key:
            self.headers["x-rapidapi-key"] = api_key
            self.headers["x-apisports-key"] = api_key

    def get(self, endpoint: str, params: Mapping) -> Mapping:
        endpoint_clean = endpoint.lstrip("/")
        url = f"{self.base_url}/{endpoint_clean}"
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            started = time.perf_counter()
            try:
                response = self.request_get(
                    url,
                    headers=self.headers,
                    params=dict(params),
                    timeout=self.timeout_seconds,
                )
                latency_ms = int((time.perf_counter() - started) * 1000)
                self._log_response(endpoint_clean, params, response, latency_ms, attempt)
                status_code = getattr(response, "status_code", None)
                if status_code in RETRYABLE_STATUS_CODES:
                    if status_code == 429 and attempt == self.max_attempts:
                        raise FootballApiQuotaError("API-Sports HTTP 429: too many requests")
                    if attempt < self.max_attempts:
                        headers = getattr(response, "headers", {}) or {}
                        self.sleep(backoff_delay(attempt, retry_after=headers.get("Retry-After")))
                        continue
                    raise FootballApiTransientError(
                        f"API-Sports {status_code} for {endpoint_clean} after {attempt} attempts"
                    )
                response.raise_for_status()
                payload = response.json()
                self._raise_for_provider_errors(endpoint_clean, payload)
                return payload
            except FootballApiError:
                raise
            except requests.RequestException as error:
                last_error = error
                self.logger.warning(
                    "football_api_request_exception",
                    extra={"endpoint": endpoint_clean, "params": dict(params), "attempt": attempt},
                )
                if attempt < self.max_attempts:
                    self.sleep(backoff_delay(attempt))
                    continue
                raise FootballApiTransientError(
                    f"API-Sports request failed for {endpoint_clean} after {attempt} attempts: {error}"
                ) from error
        raise FootballApiTransientError(f"API-Sports request failed for {endpoint_clean}: {last_error}")

    def _raise_for_provider_errors(self, endpoint: str, payload: Mapping) -> None:
        errors = payload.get("errors") if isinstance(payload, Mapping) else None
        if not errors:
            return
        if is_quota_error_payload(payload):
            raise FootballApiQuotaError(f"API-Sports quota/rate-limit error for {endpoint}: {errors}")
        raise FootballApiPayloadError(f"API-Sports returned errors for {endpoint}: {errors}")

    def _log_response(self, endpoint: str, params: Mapping, response, latency_ms: int, attempt: int) -> None:
        quota_headers = {
            key: value
            for key, value in getattr(response, "headers", {}).items()
            if "rate" in key.lower() or "quota" in key.lower() or "limit" in key.lower()
        }
        self.logger.info(
            "football_api_request",
            extra={
                "endpoint": endpoint,
                "params": dict(params),
                "status_code": getattr(response, "status_code", None),
                "latency_ms": latency_ms,
                "attempt": attempt,
                "quota_headers": quota_headers,
            },
        )
