with players as (
    select * from {{ ref('stg_football_player_match_stats') }}
),

fixtures as (
    select
        fixture_id,
        fixture_date_utc,
        league_id,
        league_name,
        league_season,
        home_team_id,
        away_team_id
    from {{ ref('stg_football_fixtures') }}
)

select
    players.fixture_id,
    players.team_id,
    players.team_name,
    players.player_id,
    players.player_name,
    players.games_minutes,
    players.games_position,
    players.shots_total,
    players.shots_on,
    players.goals_total,
    players.response_hash,
    players.player_name_normalized,
    players.team_name_normalized,
    fixtures.fixture_date_utc,
    fixtures.league_id,
    fixtures.league_name,
    fixtures.league_season,
    case
        when players.team_id = fixtures.home_team_id then fixtures.away_team_id
        when players.team_id = fixtures.away_team_id then fixtures.home_team_id
    end as opponent_team_id,
    case
        when players.games_minutes > 0 then players.shots_total * 90.0 / players.games_minutes
        else 0.0
    end as shots_per_90,
    case
        when players.games_minutes > 0 then players.shots_on * 90.0 / players.games_minutes
        else 0.0
    end as shots_on_per_90,
    current_timestamp() as updated_at_utc
from players
left join fixtures using (fixture_id)
