with base as (
    select * from {{ ref('fct_football__player_match_features') }}
),

zeroed as (
    select
        fixture_id,
        team_id,
        team_name,
        team_name_normalized,
        opponent_team_id,
        opponent_team_name,
        opponent_team_name_normalized,
        player_id,
        player_name,
        player_name_normalized,
        fixture_date_utc,
        league_id,
        league_name,
        league_season,
        status_short,
        home_away,

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

        cast(coalesce(games_minutes, 0) as double) as games_minutes,
        cast(coalesce(offsides, 0) as double) as offsides,
        cast(coalesce(shots_total, 0) as double) as shots_total,
        cast(coalesce(shots_on, 0) as double) as shots_on,
        cast(coalesce(goals_total, 0) as double) as goals_total,
        cast(coalesce(goals_assists, 0) as double) as goals_assists,
        cast(coalesce(tackles_total, 0) as double) as tackles_total,
        cast(coalesce(tackles_blocks, 0) as double) as tackles_blocks,
        cast(coalesce(tackles_interceptions, 0) as double) as tackles_interceptions,
        cast(coalesce(dribbles_attempts, 0) as double) as dribbles_attempts,
        cast(coalesce(dribbles_success, 0) as double) as dribbles_success,
        cast(coalesce(fouls_drawn, 0) as double) as fouls_drawn,
        cast(coalesce(fouls_committed, 0) as double) as fouls_committed,
        cast(coalesce(cards_yellow, 0) as double) as cards_yellow,
        cast(coalesce(cards_red, 0) as double) as cards_red,

        shots_per_90,
        shots_on_per_90,
        goals_per_90,
        assists_per_90,
        tackles_per_90,
        interceptions_per_90,
        dribbles_attempted_per_90,
        fouls_committed_per_90,
        shots_on_target_rate,
        dribble_success_rate,

        game_importance_scalar,
        opponent_strength_adjustment,
        defensive_containment_rating,
        defensive_elo,
        response_hash
    from base
),

lagged as (
    select
        *,

        sum(games_minutes) over (
            partition by player_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as minutes_l5,

        sum(shots_total) over (
            partition by player_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as shots_total_l5,

        sum(shots_on) over (
            partition by player_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as shots_on_l5,

        sum(dribbles_attempts) over (
            partition by player_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as dribbles_attempts_l5,

        sum(fouls_committed) over (
            partition by player_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as fouls_committed_l5,

        sum(tackles_interceptions) over (
            partition by player_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as tackles_interceptions_l5,

        avg(game_importance_scalar) over (
            partition by player_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as game_importance_l5,

        avg(defensive_elo) over (
            partition by opponent_team_id
            order by fixture_date_utc, fixture_id
            rows between 10 preceding and 1 preceding
        ) as opponent_defensive_elo_l10
    from zeroed
),

featured as (
    select
        *,
        case when coalesce(minutes_l5, 0) > 0 then shots_total_l5 * 90.0 / minutes_l5 else 0.0 end as shots_total_l5_p90,
        case when coalesce(minutes_l5, 0) > 0 then shots_on_l5 * 90.0 / minutes_l5 else 0.0 end as shots_on_l5_p90,
        case when coalesce(minutes_l5, 0) > 0 then dribbles_attempts_l5 * 90.0 / minutes_l5 else 0.0 end as dribbles_attempts_l5_p90,
        case when coalesce(minutes_l5, 0) > 0 then fouls_committed_l5 * 90.0 / minutes_l5 else 0.0 end as fouls_committed_l5_p90,
        case when coalesce(minutes_l5, 0) > 0 then tackles_interceptions_l5 * 90.0 / minutes_l5 else 0.0 end as tackles_interceptions_l5_p90,
        greatest(games_minutes / 90.0, 0.000001) as exposure
    from lagged
)

select
    fixture_id,
    fixture_date_utc,
    league_id,
    league_name,
    league_season,
    status_short,
    team_id,
    team_name,
    team_name_normalized,
    opponent_team_id,
    opponent_team_name,
    opponent_team_name_normalized,
    home_away,

    player_id,
    player_name,
    player_name_normalized,
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

    games_minutes,
    exposure,

    shots_total,
    shots_on,
    dribbles_attempts,
    fouls_committed,
    offsides,
    tackles_interceptions,
    goals_total,
    goals_assists,
    cards_yellow,
    cards_red,

    shots_per_90,
    shots_on_per_90,
    goals_per_90,
    assists_per_90,
    tackles_per_90,
    interceptions_per_90,
    dribbles_attempted_per_90,
    fouls_committed_per_90,
    shots_on_target_rate,
    dribble_success_rate,

    coalesce(minutes_l5, 0.0) as minutes_l5,
    coalesce(shots_total_l5, 0.0) as shots_total_l5,
    coalesce(shots_on_l5, 0.0) as shots_on_l5,
    coalesce(dribbles_attempts_l5, 0.0) as dribbles_attempts_l5,
    coalesce(fouls_committed_l5, 0.0) as fouls_committed_l5,
    coalesce(tackles_interceptions_l5, 0.0) as tackles_interceptions_l5,
    shots_total_l5_p90,
    shots_on_l5_p90,
    dribbles_attempts_l5_p90,
    fouls_committed_l5_p90,
    tackles_interceptions_l5_p90,
    coalesce(game_importance_l5, 1.0) as game_importance_l5,

    game_importance_scalar,
    opponent_strength_adjustment,
    defensive_containment_rating,
    defensive_elo,
    coalesce(opponent_defensive_elo_l10, 1500.0) as opponent_defensive_elo_l10,

    response_hash,
    current_timestamp() as updated_at_utc
from featured
