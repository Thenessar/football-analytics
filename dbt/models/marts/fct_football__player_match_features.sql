with player_stats as (
    select * from {{ ref('stg_football__player_match_stats') }}
),

lineups as (
    select * from {{ ref('stg_football__lineups') }}
),

team_context as (
    select * from {{ ref('fct_football__team_match_context') }}
),

joined as (
    select
        player_stats.fixture_id,
        player_stats.team_id,
        player_stats.player_id,
        player_stats.team_name,
        player_stats.team_name_normalized,
        player_stats.player_name,
        player_stats.player_name_normalized,
        player_stats.player_photo_url,
        player_stats.games_number,
        player_stats.games_position,
        lineups.position as lineup_position,
        coalesce(lineups.position, player_stats.games_position) as primary_position,
        player_stats.games_rating,
        player_stats.games_captain,
        player_stats.games_substitute,
        lineups.is_starting,
        lineups.formation,
        lineups.formation_grid,
        lineups.formation_row,
        lineups.formation_column,
        team_context.fixture_date_utc,
        team_context.league_id,
        team_context.league_name,
        team_context.league_season,
        team_context.status_short,
        team_context.opponent_team_id,
        team_context.opponent_team_name,
        team_context.opponent_team_name_normalized,
        team_context.home_away,
        team_context.goals_for,
        team_context.goals_against,
        coalesce(team_context.game_importance_scalar, 1.0) as game_importance_scalar,
        coalesce(team_context.opponent_strength_adjustment, 1.0) as opponent_strength_adjustment,
        coalesce(team_context.defensive_containment_rating, 1.0) as defensive_containment_rating,
        coalesce(team_context.defensive_elo, 1500.0) as defensive_elo,
        player_stats.games_minutes,
        player_stats.offsides,
        player_stats.shots_total,
        player_stats.shots_on,
        player_stats.goals_total,
        player_stats.goals_conceded,
        player_stats.goals_assists,
        player_stats.goals_saves,
        player_stats.passes_total,
        player_stats.passes_key,
        player_stats.passes_accuracy,
        player_stats.tackles_total,
        player_stats.tackles_blocks,
        player_stats.tackles_interceptions,
        player_stats.duels_total,
        player_stats.duels_won,
        player_stats.dribbles_attempts,
        player_stats.dribbles_success,
        player_stats.dribbles_past,
        player_stats.fouls_drawn,
        player_stats.fouls_committed,
        player_stats.cards_yellow,
        player_stats.cards_red,
        player_stats.penalty_won,
        player_stats.penalty_committed,
        player_stats.penalty_scored,
        player_stats.penalty_missed,
        player_stats.penalty_saved,
        player_stats.response_hash
    from player_stats
    left join lineups
        on player_stats.fixture_id = lineups.fixture_id
       and player_stats.team_id = lineups.team_id
       and player_stats.player_id = lineups.player_id
    left join team_context
        on player_stats.fixture_id = team_context.fixture_id
       and player_stats.team_id = team_context.team_id
),

featured as (
    select
        *,
        case
            when upper(primary_position) in ('G', 'GK', 'GOALKEEPER') then 'G'
            when upper(primary_position) in ('D', 'DEFENDER', 'DEFENCE', 'CB', 'LB', 'RB', 'LWB', 'RWB') then 'D'
            when upper(primary_position) in ('F', 'A', 'ATTACKER', 'FORWARD', 'STRIKER', 'CF', 'LW', 'RW') then 'F'
            else 'M'
        end as position_group,
        coalesce(is_starting, not coalesce(games_substitute, false)) as is_starter,
        coalesce(games_substitute, not coalesce(is_starting, true)) as was_substitute,
        coalesce(games_captain, false) as is_captain,
        case
            when upper(primary_position) in ('G', 'GK', 'GOALKEEPER') then true
            else false
        end as is_goalkeeper
    from joined
)

select
    fixture_id,
    team_id,
    player_id,

    fixture_date_utc,
    league_id,
    league_name,
    league_season,
    status_short,
    home_away,
    team_name,
    team_name_normalized,
    opponent_team_id,
    opponent_team_name,
    opponent_team_name_normalized,
    goals_for,
    goals_against,

    player_name,
    player_name_normalized,
    player_photo_url,
    games_number,
    games_position,
    lineup_position,
    primary_position,
    position_group,

    is_starting,
    is_starter,
    was_substitute,
    is_captain,
    is_goalkeeper,
    formation,
    formation_grid,
    formation_row,
    formation_column,

    coalesce(games_minutes, 0) as games_minutes,
    games_rating,

    coalesce(offsides, 0) as offsides,

    coalesce(shots_total, 0) as shots_total,
    coalesce(shots_on, 0) as shots_on,
    coalesce(goals_total, 0) as goals_total,
    coalesce(goals_assists, 0) as goals_assists,
    coalesce(goals_conceded, 0) as goals_conceded,
    coalesce(goals_saves, 0) as goals_saves,

    coalesce(passes_total, 0) as passes_total,
    coalesce(passes_key, 0) as passes_key,
    coalesce(passes_accuracy, 0) as passes_accuracy,

    coalesce(tackles_total, 0) as tackles_total,
    coalesce(tackles_blocks, 0) as tackles_blocks,
    coalesce(tackles_interceptions, 0) as tackles_interceptions,

    coalesce(duels_total, 0) as duels_total,
    coalesce(duels_won, 0) as duels_won,

    coalesce(dribbles_attempts, 0) as dribbles_attempts,
    coalesce(dribbles_success, 0) as dribbles_success,
    coalesce(dribbles_past, 0) as dribbles_past,

    coalesce(fouls_drawn, 0) as fouls_drawn,
    coalesce(fouls_committed, 0) as fouls_committed,

    coalesce(cards_yellow, 0) as cards_yellow,
    coalesce(cards_red, 0) as cards_red,

    coalesce(penalty_won, 0) as penalty_won,
    coalesce(penalty_committed, 0) as penalty_committed,
    coalesce(penalty_scored, 0) as penalty_scored,
    coalesce(penalty_missed, 0) as penalty_missed,
    coalesce(penalty_saved, 0) as penalty_saved,

    case when games_minutes > 0 then coalesce(shots_total, 0) * 90.0 / games_minutes else 0.0 end as shots_per_90,
    case when games_minutes > 0 then coalesce(shots_on, 0) * 90.0 / games_minutes else 0.0 end as shots_on_per_90,
    case when games_minutes > 0 then coalesce(goals_total, 0) * 90.0 / games_minutes else 0.0 end as goals_per_90,
    case when games_minutes > 0 then coalesce(goals_assists, 0) * 90.0 / games_minutes else 0.0 end as assists_per_90,
    case when games_minutes > 0 then coalesce(passes_total, 0) * 90.0 / games_minutes else 0.0 end as passes_per_90,
    case when games_minutes > 0 then coalesce(passes_key, 0) * 90.0 / games_minutes else 0.0 end as key_passes_per_90,
    case when games_minutes > 0 then coalesce(tackles_total, 0) * 90.0 / games_minutes else 0.0 end as tackles_per_90,
    case when games_minutes > 0 then coalesce(tackles_interceptions, 0) * 90.0 / games_minutes else 0.0 end as interceptions_per_90,
    case when games_minutes > 0 then coalesce(duels_total, 0) * 90.0 / games_minutes else 0.0 end as duels_per_90,
    case when games_minutes > 0 then coalesce(dribbles_attempts, 0) * 90.0 / games_minutes else 0.0 end as dribbles_attempted_per_90,
    case when games_minutes > 0 then coalesce(fouls_committed, 0) * 90.0 / games_minutes else 0.0 end as fouls_committed_per_90,
    case when games_minutes > 0 then coalesce(goals_saves, 0) * 90.0 / games_minutes else 0.0 end as saves_per_90,

    case when coalesce(shots_total, 0) > 0 then coalesce(shots_on, 0) * 1.0 / shots_total else 0.0 end as shots_on_target_rate,
    case when coalesce(passes_total, 0) > 0 then coalesce(passes_accuracy, 0) * 100.0 / passes_total else 0.0 end as pass_accuracy_pct,
    case when coalesce(duels_total, 0) > 0 then coalesce(duels_won, 0) * 1.0 / duels_total else 0.0 end as duels_won_rate,
    case when coalesce(dribbles_attempts, 0) > 0 then coalesce(dribbles_success, 0) * 1.0 / dribbles_attempts else 0.0 end as dribble_success_rate,

    game_importance_scalar,
    opponent_strength_adjustment,
    defensive_containment_rating,
    defensive_elo,
    response_hash,
    current_timestamp() as updated_at_utc
from featured
where coalesce(games_minutes, 0) between 1 and 130
