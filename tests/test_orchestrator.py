import datetime

import pytest

from football_analytics.orchestrator import (
    resolve_starting_lineups,
    poll_and_orchestrate,
)
from football_analytics.pipeline import ConfirmedLineupDataError


def test_offline_resolution_uses_configured_active_mock_xi():
    home, away, home_source, away_source = resolve_starting_lineups(
        "Germany",
        "Ecuador",
        lineups_data=None,
        offline=True,
    )

    home_names = {player["name"] for player in home}
    assert len(home) == 11
    assert len(away) == 11
    assert {"Deniz Undav", "Nadiem Amiri", "Maximilian Beier", "Jamie Leweling"} <= home_names
    assert {"Kai Havertz", "Felix Nmecha", "Leroy Sané"}.isdisjoint(home_names)
    assert home_source == "explicit offline sample XI"
    assert away_source == "explicit offline sample XI"


def test_live_resolution_fails_closed_without_confirmed_lineups():
    with pytest.raises(ConfirmedLineupDataError, match="Confirmed starting XI"):
        resolve_starting_lineups(
            "Germany",
            "Ecuador",
            lineups_data=None,
            offline=False,
        )


def test_t60_live_monitor_passes_exact_verified_payload(monkeypatch):
    verified_payload = [{"team": {"name": "Germany"}, "startXI": []}]
    captured = {}

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            pass

        def get_fixture_details(self, fixture_id):
            return {
                "fixture": {
                    "date": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                }
            }

        def fetch_confirmed_lineups(self, fixture_id):
            return verified_payload

    def fake_run(*args, **kwargs):
        captured["confirmed_lineups"] = kwargs["confirmed_lineups"]

    monkeypatch.setattr(
        "football_analytics.orchestrator.FootballDataPipeline",
        FakePipeline,
    )
    monkeypatch.setattr(
        "football_analytics.orchestrator.run_matchup_pipeline",
        fake_run,
    )

    poll_and_orchestrate("Germany", "Ecuador", 1489437, offline=False)

    assert captured["confirmed_lineups"] is verified_payload


def test_t60_live_monitor_raises_instead_of_retrying(monkeypatch):
    class MissingLineupPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def get_fixture_details(self, fixture_id):
            return {
                "fixture": {
                    "date": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat()
                }
            }

        def fetch_confirmed_lineups(self, fixture_id):
            raise ConfirmedLineupDataError("official lineup missing")

    monkeypatch.setattr(
        "football_analytics.orchestrator.FootballDataPipeline",
        MissingLineupPipeline,
    )

    with pytest.raises(ConfirmedLineupDataError, match="official lineup missing"):
        poll_and_orchestrate("Germany", "Ecuador", 1489437, offline=False)


def test_live_monitor_wraps_fixture_metadata_failure(monkeypatch):
    class BrokenMetadataPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def get_fixture_details(self, fixture_id):
            raise ConnectionError("network unavailable")

    monkeypatch.setattr(
        "football_analytics.orchestrator.FootballDataPipeline",
        BrokenMetadataPipeline,
    )

    with pytest.raises(ConfirmedLineupDataError, match="metadata fetch failed"):
        poll_and_orchestrate("Germany", "Ecuador", 1489437, offline=False)
