with fixtures as (
    select * from {{ ref('stg_football_fixtures') }}
),

home_rows as (
    select
        fixture_id,
        fixture_date_utc,
        home_team_id as team_id,
        home_team_name as team_name,
        away_team_id as opponent_team_id,
        away_team_name as opponent_team_name,
        'home' as home_away,
        home_goals as goals_for,
        away_goals as goals_against,
        league_id,
        league_name,
        league_season,
        status_short
    from fixtures
),

away_rows as (
    select
        fixture_id,
        fixture_date_utc,
        away_team_id as team_id,
        away_team_name as team_name,
        home_team_id as opponent_team_id,
        home_team_name as opponent_team_name,
        'away' as home_away,
        away_goals as goals_for,
        home_goals as goals_against,
        league_id,
        league_name,
        league_season,
        status_short
    from fixtures
)

select
    fixture_id,
    fixture_date_utc,
    team_id,
    team_name,
    opponent_team_id,
    opponent_team_name,
    home_away,
    goals_for,
    goals_against,
    league_id,
    league_name,
    league_season,
    status_short,
    {{ normalize_name('team_name') }} as team_name_normalized,
    {{ normalize_name('opponent_team_name') }} as opponent_team_name_normalized,
    cast(1.0 as double) as game_importance_scalar,
    cast(1.0 as double) as opponent_strength_adjustment,
    cast(1.0 as double) as defensive_containment_rating,
    cast(1500.0 as double) as defensive_elo,
    current_timestamp() as updated_at_utc
from (
    select * from home_rows
    union all
    select * from away_rows
)
