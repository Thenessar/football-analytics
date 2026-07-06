{{ config(materialized='view') }}

-- Per-fixture operational coverage (§16.3): confirmed-starter prediction
-- coverage, missing expected-minutes counts, and the lineup source of the
-- active prediction set. Fixtures without an active prediction set are
-- included with zero coverage so gaps are visible.

with confirmed_starters as (
    select fixture_id, team_id, player_id
    from {{ ref('stg_football__lineups') }}
    where is_starting
),

active_predictions as (
    select *
    from {{ source('gold_predictions', 'pred_football__player_event_predictions') }}
    where is_active_prediction
),

predicted_players as (
    select distinct
        fixture_id,
        team_id,
        player_id,
        lineup_source,
        case when expected_minutes is null then 1 else 0 end as missing_expected_minutes
    from active_predictions
),

fixture_lineups as (
    select
        confirmed_starters.fixture_id,
        count(*) as confirmed_starters,
        count(predicted_players.player_id) as starters_with_predictions,
        sum(coalesce(predicted_players.missing_expected_minutes, 0)) as starters_missing_expected_minutes,
        max(predicted_players.lineup_source) as lineup_source
    from confirmed_starters
    left join predicted_players
        on confirmed_starters.fixture_id = predicted_players.fixture_id
       and confirmed_starters.team_id = predicted_players.team_id
       and confirmed_starters.player_id = predicted_players.player_id
    group by confirmed_starters.fixture_id
)

select
    fixtures.fixture_id,
    fixtures.fixture_date_utc,
    fixtures.league_id,
    fixtures.league_name,
    fixtures.status_short,
    coalesce(fixture_lineups.confirmed_starters, 0) as confirmed_starters,
    coalesce(fixture_lineups.starters_with_predictions, 0) as starters_with_predictions,
    case
        when coalesce(fixture_lineups.confirmed_starters, 0) > 0
        then coalesce(fixture_lineups.starters_with_predictions, 0) * 1.0 / fixture_lineups.confirmed_starters
        else 0.0
    end as starter_prediction_coverage,
    coalesce(fixture_lineups.starters_missing_expected_minutes, 0) as starters_missing_expected_minutes,
    fixture_lineups.lineup_source
from {{ ref('stg_football__fixtures') }} as fixtures
left join fixture_lineups
    on fixtures.fixture_id = fixture_lineups.fixture_id
