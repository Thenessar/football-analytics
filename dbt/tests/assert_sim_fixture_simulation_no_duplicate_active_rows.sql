-- Exactly one active simulation row per fixture/entity/target/engine version
-- (ml_upgrade_backlog.md N4; mirrors the prediction-table uniqueness test).

select
    fixture_id,
    entity_type,
    entity_id,
    target_event,
    engine_version,
    count(*) as active_rows
from {{ source('gold_predictions', 'sim_football__fixture_simulation') }}
where is_active_simulation
group by fixture_id, entity_type, entity_id, target_event, engine_version
having count(*) > 1
