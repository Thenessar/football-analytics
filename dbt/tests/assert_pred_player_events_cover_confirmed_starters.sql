-- Every confirmed starter must have an active prediction row for every
-- target event that was predicted for that fixture (business_logic.md §16.2).
-- Fixtures without any active prediction set are out of scope, so this test
-- passes on an empty prediction table.
with active_predictions as (
    select *
    from {{ source('gold_predictions', 'pred_football__player_event_predictions') }}
    where is_active_prediction
),

fixture_targets as (
    select distinct fixture_id, target_event
    from active_predictions
),

confirmed_starters as (
    select fixture_id, team_id, player_id
    from {{ ref('stg_football__lineups') }}
    where is_starting
),

expected_rows as (
    select
        confirmed_starters.fixture_id,
        confirmed_starters.team_id,
        confirmed_starters.player_id,
        fixture_targets.target_event
    from confirmed_starters
    inner join fixture_targets
        on confirmed_starters.fixture_id = fixture_targets.fixture_id
)

select expected_rows.*
from expected_rows
left join active_predictions
    on expected_rows.fixture_id = active_predictions.fixture_id
   and expected_rows.team_id = active_predictions.team_id
   and expected_rows.player_id = active_predictions.player_id
   and expected_rows.target_event = active_predictions.target_event
where active_predictions.player_id is null
