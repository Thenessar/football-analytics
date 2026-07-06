{{ config(materialized='view') }}

-- Row-level actual-vs-predicted for completed fixtures (§16.3 groundwork).
-- Unpivots the 11 target-event labels from the player event feature mart and
-- joins the active prediction rows, so downstream monitoring can compute
-- accuracy, bias, and calibration at any aggregation level.

with completed_labels as (
    select
        fixture_id,
        fixture_date_utc,
        league_id,
        league_name,
        league_season,
        team_id,
        player_id,
        position_group,
        stack(
            11,
            'offsides', offsides,
            'shots_total', shots_total,
            'shots_on', shots_on,
            'goals_total', goals_total,
            'goals_assists', goals_assists,
            'goals_saves', goals_saves,
            'passes_total', passes_total,
            'fouls_drawn', fouls_drawn,
            'fouls_committed', fouls_committed,
            'cards_yellow', cards_yellow,
            'cards_red', cards_red
        ) as (target_event, actual_count)
    from {{ ref('fct_football__player_event_features') }}
    where is_completed_fixture
),

active_predictions as (
    select *
    from {{ source('gold_predictions', 'pred_football__player_event_predictions') }}
    where is_active_prediction
)

select
    labels.fixture_id,
    labels.fixture_date_utc,
    labels.league_id,
    labels.league_name,
    labels.league_season,
    labels.team_id,
    labels.player_id,
    labels.position_group,
    labels.target_event,
    labels.actual_count,
    predictions.predicted_mean,
    predictions.predicted_p_ge_1,
    predictions.predicted_p_ge_2,
    predictions.predicted_p_ge_3,
    predictions.expected_minutes,
    predictions.lineup_source,
    predictions.model_name,
    predictions.model_version,
    predictions.prediction_set_id
from completed_labels as labels
inner join active_predictions as predictions
    on labels.fixture_id = predictions.fixture_id
   and labels.team_id = predictions.team_id
   and labels.player_id = predictions.player_id
   and labels.target_event = predictions.target_event
