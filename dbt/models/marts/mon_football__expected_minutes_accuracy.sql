{{ config(materialized='view') }}

-- Expected-minutes estimator accuracy (ml_upgrade_backlog.md K4): predicted
-- vs actual minutes over completed fixtures, segmented by confirmed role,
-- estimator branch, and position group. This view is the gate any future ML
-- minutes model must beat (ADR 0004 adoption trigger: >=10% relative MAE
-- improvement on a season holdout) and the evidence base for the v1-vs-v2
-- estimator comparison. Empty-but-queryable before completed fixtures land.
--
-- Actual minutes are coalesced to 0: a bench player who never came on
-- genuinely played 0 minutes, and the role-conditional estimator prices that
-- outcome through p_plays.

with completed_fixtures as (
    select
        fixture_id,
        fixture_date_utc,
        league_id,
        league_name,
        league_season
    from {{ ref('stg_football__fixtures') }}
    where status_short in ('FT', 'AET', 'PEN')
),

expected as (
    select
        expected_minutes.fixture_id,
        expected_minutes.team_id,
        expected_minutes.player_id,
        expected_minutes.is_starting,
        expected_minutes.position_group,
        expected_minutes.expected_minutes,
        expected_minutes.expected_minutes_method,
        expected_minutes.p_plays,
        completed_fixtures.league_season
    from {{ ref('fct_football__player_expected_minutes') }} as expected_minutes
    inner join completed_fixtures
        on expected_minutes.fixture_id = completed_fixtures.fixture_id
),

actuals as (
    select
        fixture_id,
        team_id,
        player_id,
        least(cast(coalesce(games_minutes, 0) as double), 130.0) as actual_minutes
    from {{ ref('stg_football__player_match_stats') }}
),

scored as (
    select
        expected.league_season,
        expected.is_starting,
        expected.expected_minutes_method,
        expected.position_group,
        expected.expected_minutes,
        expected.p_plays,
        coalesce(actuals.actual_minutes, 0.0) as actual_minutes,
        case when coalesce(actuals.actual_minutes, 0.0) >= 1 then 1.0 else 0.0 end as played_flag
    from expected
    left join actuals
        on expected.fixture_id = actuals.fixture_id
       and expected.team_id = actuals.team_id
       and expected.player_id = actuals.player_id
)

select
    league_season,
    is_starting,
    expected_minutes_method,
    position_group,
    count(*) as row_count,
    avg(abs(expected_minutes - actual_minutes)) as mae_minutes,
    avg(expected_minutes - actual_minutes) as bias_minutes,
    avg(expected_minutes) as avg_expected_minutes,
    avg(actual_minutes) as avg_actual_minutes,
    -- Participation calibration for bench players: p_plays should match the
    -- empirical played share within the segment.
    avg(p_plays) as avg_p_plays,
    avg(played_flag) as actual_played_share
from scored
group by league_season, is_starting, expected_minutes_method, position_group
