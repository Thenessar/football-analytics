select
    fixture_id,
    team_id,
    player_id,
    target_event,
    model_name,
    model_version,
    count(*) as active_rows
from {{ source('gold_predictions', 'pred_football__player_event_predictions') }}
where is_active_prediction
group by fixture_id, team_id, player_id, target_event, model_name, model_version
having count(*) > 1
