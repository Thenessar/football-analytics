{{ config(materialized='view') }}

-- Inference feature-health monitor (§16.3, prediction_quality_backlog FA-103).
-- Guards the E-2 class of serving skew: production once served every player of
-- a team the same player-Elo values, and validation never noticed because it
-- scored warehouse rows. One row per upcoming fixture/team/monitored feature:
-- null rate and within-team distinct-value count for the player-differentiating
-- families, plus played-history freshness. A full lineup (>= 11 rows) whose
-- feature collapses to <= 1 distinct value is flagged. Empty-but-queryable:
-- with no upcoming fixtures the view returns zero rows.

{% set monitored_features = [
    'player_offensive_modifier_pre',
    'player_defensive_modifier_pre',
    'player_offensive_elo_pre',
    'player_defensive_elo_pre',
    'player_offensive_rating_pre',
    'player_defensive_rating_pre',
    'appearances_l5_count',
    'offsides_l5_p90',
    'shots_total_l5_p90',
    'shots_on_l5_p90',
    'goals_total_l5_p90',
    'goals_assists_l5_p90',
    'goals_saves_l5_p90',
    'passes_total_l5_p90',
    'fouls_drawn_l5_p90',
    'fouls_committed_l5_p90',
    'cards_yellow_l5_p90',
    'cards_red_l5_p90',
] %}

with inference_rows as (
    select *
    from {{ ref('fct_football__player_event_features') }}
    where not is_completed_fixture
),

team_history_freshness as (
    select
        team_id,
        max(fixture_date_utc) as team_history_max_fixture_date_utc
    from {{ ref('fct_football__player_match_features') }}
    group by team_id
),

global_history_freshness as (
    select max(fixture_date_utc) as history_max_fixture_date_utc
    from {{ ref('fct_football__player_match_features') }}
),

feature_stats as (
    {% for feature in monitored_features %}
    select
        fixture_id,
        fixture_date_utc,
        team_id,
        team_name,
        status_short,
        '{{ feature }}' as feature_name,
        count(*) as lineup_row_count,
        count(*) - count({{ feature }}) as null_count,
        1.0 - count({{ feature }}) * 1.0 / count(*) as null_rate,
        count(distinct {{ feature }}) as distinct_value_count
    from inference_rows
    group by fixture_id, fixture_date_utc, team_id, team_name, status_short
    {% if not loop.last %}union all{% endif %}
    {% endfor %}
)

select
    feature_stats.fixture_id,
    feature_stats.fixture_date_utc,
    feature_stats.team_id,
    feature_stats.team_name,
    feature_stats.status_short,
    feature_stats.feature_name,
    feature_stats.lineup_row_count,
    feature_stats.null_count,
    feature_stats.null_rate,
    feature_stats.distinct_value_count,
    -- Distinct count <= 1 covers both all-null (0) and all-constant (1); the
    -- >= 11 gate skips partially ingested lineups where constancy can be
    -- legitimate. Sparse events (e.g. cards_red_l5_p90) may flag on quiet
    -- teams; the hard assertion lives in the FA-101 data test on modifiers.
    (feature_stats.lineup_row_count >= 11 and feature_stats.distinct_value_count <= 1) as is_flagged_constant,
    team_history_freshness.team_history_max_fixture_date_utc,
    global_history_freshness.history_max_fixture_date_utc,
    (unix_timestamp(current_timestamp()) - unix_timestamp(global_history_freshness.history_max_fixture_date_utc)) / 3600.0 as history_staleness_hours
from feature_stats
left join team_history_freshness
    on feature_stats.team_id = team_history_freshness.team_id
cross join global_history_freshness
