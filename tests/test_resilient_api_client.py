import pytest

from football_analytics.api.client import FootballApiClient
from football_analytics.api.exceptions import FootballApiQuotaError, FootballApiTransientError


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {"errors": [], "response": []}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("retryable statuses should be handled before raise_for_status")

    def json(self):
        return self._payload


def test_one_time_503_then_success_is_retried():
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(kwargs["params"])
        if len(calls) == 1:
            return FakeResponse(status_code=503)
        return FakeResponse(payload={"errors": [], "response": [{"ok": True}]})

    client = FootballApiClient(request_get=fake_get, sleep=lambda _: None)

    payload = client.get("fixtures/players", {"fixture": 1})

    assert payload["response"] == [{"ok": True}]
    assert len(calls) == 2


def test_repeated_503_exhaustion_raises_transient_error():
    client = FootballApiClient(
        request_get=lambda *args, **kwargs: FakeResponse(status_code=503),
        sleep=lambda _: None,
        max_attempts=2,
    )

    with pytest.raises(FootballApiTransientError, match="503"):
        client.get("fixtures/players", {"fixture": 1})


def test_429_with_retry_after_sleeps_before_success():
    delays = []
    responses = [
        FakeResponse(status_code=429, headers={"Retry-After": "2"}),
        FakeResponse(payload={"errors": [], "response": [{"ok": True}]}),
    ]

    def fake_get(*args, **kwargs):
        return responses.pop(0)

    client = FootballApiClient(request_get=fake_get, sleep=delays.append)

    payload = client.get("fixtures/players", {"fixture": 1})

    assert payload["response"][0]["ok"] is True
    assert delays == [2.0]


def test_429_after_retries_raises_quota_error():
    client = FootballApiClient(
        request_get=lambda *args, **kwargs: FakeResponse(status_code=429),
        sleep=lambda _: None,
        max_attempts=1,
    )

    with pytest.raises(FootballApiQuotaError, match="429"):
        client.get("fixtures/players", {"fixture": 1})

