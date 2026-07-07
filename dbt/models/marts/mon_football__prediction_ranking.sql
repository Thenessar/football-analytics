{{ config(materialized='view') }}

-- Within-fixture ranking quality on live predictions (ml_upgrade_backlog.md
-- L5): did the player we predicted to lead his team in an event actually
-- lead it? Mirrors the top-1 leader hit rate the training harness logs (L3)
-- and the allocation-quality checks of the simulation backtest (N6).
-- Team-fixtures where nobody recorded the event cannot rank anyone and are
-- excluded from the hit rate (surfaced via team_fixtures_all_zero).

with vs_actual as (
    select *
    from {{ ref('mon_football__prediction_vs_actual') }}
),

ranked as (
    select
        fixture_id,
        team_id,
        target_event,
        model_version,
        league_name,
        actual_count,
        row_number() over (
            partition by fixture_id, team_id, target_event, model_version
            order by predicted_mean desc, player_id
        ) as predicted_rank,
        max(actual_count) over (
            partition by fixture_id, team_id, target_event, model_version
        ) as max_actual_count
    from vs_actual
),

team_fixture_outcomes as (
    select
        fixture_id,
        team_id,
        target_event,
        model_version,
        league_name,
        max(max_actual_count) as max_actual_count,
        max(
            case
                when predicted_rank = 1 and actual_count = max_actual_count then 1.0
                when predicted_rank = 1 then 0.0
            end
        ) as top1_hit
    from ranked
    group by fixture_id, team_id, target_event, model_version, league_name
)

select
    target_event,
    model_version,
    league_name,
    count(case when max_actual_count > 0 then 1 end) as team_fixtures_scored,
    count(case when max_actual_count = 0 then 1 end) as team_fixtures_all_zero,
    avg(case when max_actual_count > 0 then top1_hit end) as top1_hit_rate
from team_fixture_outcomes
group by target_event, model_version, league_name
