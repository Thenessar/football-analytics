{{ config(materialized='view') }}

-- Simulation-vs-actual monitoring (ml_upgrade_backlog.md O2): active
-- player-level simulation rows joined to realized labels after fixture
-- completion. Surfaces per target/position-group/engine-version: mean error,
-- 5-95% interval hit rate (should be >= ~0.90 for a calibrated simulator on
-- discrete counts), and P(>=1) calibration inputs. Empty-but-queryable until
-- simulations and completed fixtures overlap.

with completed_labels as (
    select
        fixture_id,
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

active_simulations as (
    select *
    from {{ source('gold_predictions', 'sim_football__fixture_simulation') }}
    where is_active_simulation
      and entity_type = 'player'
),

scored as (
    select
        labels.league_season,
        labels.target_event,
        labels.position_group,
        simulations.engine_version,
        labels.actual_count,
        simulations.sim_mean,
        simulations.p05,
        simulations.p95,
        simulations.p_ge_1,
        case when labels.actual_count between simulations.p05 and simulations.p95 then 1.0 else 0.0 end as interval_hit,
        case when labels.actual_count >= 1 then 1.0 else 0.0 end as event_ge_1
    from completed_labels as labels
    inner join active_simulations as simulations
        on labels.fixture_id = simulations.fixture_id
       and labels.team_id = simulations.team_id
       and labels.player_id = simulations.entity_id
       and labels.target_event = simulations.target_event
)

select
    league_season,
    target_event,
    position_group,
    engine_version,
    count(*) as row_count,
    avg(sim_mean - actual_count) as bias,
    avg(abs(sim_mean - actual_count)) as mae,
    avg(interval_hit) as interval_hit_rate,
    avg(p_ge_1) as avg_predicted_p_ge_1,
    avg(event_ge_1) as empirical_p_ge_1
from scored
group by league_season, target_event, position_group, engine_version
