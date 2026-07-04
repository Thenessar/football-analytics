with starters as (
    select
        fixture_id,
        team_id,
        team_name,
        player_id
    from {{ ref('stg_football__lineups') }}
    where is_starting = true
),

player_elo as (
    select
        fixture_id,
        team_id,
        player_id,
        player_offensive_elo_pre,
        player_defensive_elo_pre
    from {{ ref('fct_football__player_elo_history') }}
),

team_elo as (
    select
        fixture_id,
        team_id,
        team_elo_attack_pre,
        team_elo_defense_pre
    from {{ ref('fct_football__team_elo_history') }}
),

joined as (
    select
        starters.fixture_id,
        starters.team_id,
        starters.team_name,
        starters.player_id,
        coalesce(player_elo.player_offensive_elo_pre, team_elo.team_elo_attack_pre, 0.0) as player_offensive_elo_pre,
        coalesce(player_elo.player_defensive_elo_pre, team_elo.team_elo_defense_pre, 0.0) as player_defensive_elo_pre
    from starters
    left join player_elo
        on starters.fixture_id = player_elo.fixture_id
       and starters.team_id = player_elo.team_id
       and starters.player_id = player_elo.player_id
    left join team_elo
        on starters.fixture_id = team_elo.fixture_id
       and starters.team_id = team_elo.team_id
)

select
    fixture_id,
    team_id,
    team_name,
    avg(player_offensive_elo_pre) as starting_xi_attack_elo,
    avg(player_defensive_elo_pre) as starting_xi_defense_elo,
    count(distinct player_id) as starting_xi_player_count,
    current_timestamp() as updated_at_utc
from joined
group by
    fixture_id,
    team_id,
    team_name
having count(distinct player_id) = 11
