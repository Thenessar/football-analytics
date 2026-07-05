with base as (
    select * from {{ ref('fct_football__player_match_features') }}
),

team_elo as (
    select
        fixture_id,
        team_id,
        team_elo_general_pre,
        opponent_elo_general_pre,
        team_elo_attack_pre,
        team_elo_defense_pre,
        opponent_elo_attack_pre,
        opponent_elo_defense_pre,
        expected_goals_for_pre,
        expected_goals_against_pre
    from {{ ref('fct_football__team_elo_history') }}
),

player_elo as (
    select
        fixture_id,
        team_id,
        player_id,
        player_offensive_modifier_pre,
        player_defensive_modifier_pre,
        player_offensive_elo_pre,
        player_defensive_elo_pre,
        player_offensive_rating_pre,
        player_defensive_rating_pre,
        missed_fixture_count_pre
    from {{ ref('fct_football__player_elo_history') }}
),

lineup_strength as (
    select
        fixture_id,
        team_id,
        starting_xi_attack_elo,
        starting_xi_defense_elo
    from {{ ref('fct_football__lineup_elo_strength') }}
),

formation_history as (
    select
        fixture_id,
        team_id,
        formation_win_rate_pre,
        formation_count_pre,
        formation_matchup_win_rate_pre,
        formation_matchup_count_pre
    from {{ ref('fct_football__formation_matchup_history') }}
),

enriched as (
    select
        base.*,
        coalesce(team_elo.team_elo_general_pre, 1500.0) as team_elo_general_pre,
        coalesce(team_elo.opponent_elo_general_pre, 1500.0) as opponent_elo_general_pre,
        coalesce(team_elo.team_elo_attack_pre, 0.0) as team_elo_attack_pre,
        coalesce(team_elo.team_elo_defense_pre, 0.0) as team_elo_defense_pre,
        coalesce(team_elo.opponent_elo_attack_pre, 0.0) as opponent_elo_attack_pre,
        coalesce(team_elo.opponent_elo_defense_pre, 0.0) as opponent_elo_defense_pre,
        coalesce(team_elo.expected_goals_for_pre, 1.25) as expected_goals_for_pre,
        coalesce(team_elo.expected_goals_against_pre, 1.25) as expected_goals_against_pre,
        coalesce(player_elo.player_offensive_modifier_pre, 0.0) as player_offensive_modifier_pre,
        coalesce(player_elo.player_defensive_modifier_pre, 0.0) as player_defensive_modifier_pre,
        coalesce(player_elo.player_offensive_elo_pre, team_elo.team_elo_attack_pre, 0.0) as player_offensive_elo_pre,
        coalesce(player_elo.player_defensive_elo_pre, team_elo.team_elo_defense_pre, 0.0) as player_defensive_elo_pre,
        coalesce(player_elo.player_offensive_rating_pre, player_elo.player_offensive_elo_pre, team_elo.team_elo_attack_pre, 0.0) as player_offensive_rating_pre,
        coalesce(player_elo.player_defensive_rating_pre, player_elo.player_defensive_elo_pre, team_elo.team_elo_defense_pre, 0.0) as player_defensive_rating_pre,
        coalesce(player_elo.missed_fixture_count_pre, 0) as missed_fixture_count_pre,
        coalesce(lineup_strength.starting_xi_attack_elo, team_elo.team_elo_attack_pre, 0.0) as team_lineup_attack_strength,
        coalesce(lineup_strength.starting_xi_defense_elo, team_elo.team_elo_defense_pre, 0.0) as team_lineup_defense_strength,
        coalesce(formation_history.formation_win_rate_pre, 0.5) as formation_win_rate_pre,
        coalesce(formation_history.formation_count_pre, 0) as formation_count_pre,
        coalesce(formation_history.formation_matchup_win_rate_pre, 0.5) as formation_matchup_win_rate_pre,
        coalesce(formation_history.formation_matchup_count_pre, 0) as formation_matchup_count_pre
    from base
    left join team_elo
        on base.fixture_id = team_elo.fixture_id
       and base.team_id = team_elo.team_id
    left join player_elo
        on base.fixture_id = player_elo.fixture_id
       and base.team_id = player_elo.team_id
       and base.player_id = player_elo.player_id
    left join lineup_strength
        on base.fixture_id = lineup_strength.fixture_id
       and base.team_id = lineup_strength.team_id
    left join formation_history
        on base.fixture_id = formation_history.fixture_id
       and base.team_id = formation_history.team_id
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
        opponent_formation,

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
        least(greatest(exp((coalesce(opponent_elo_general_pre, 1500.0) - 1500.0) / 900.0) / coalesce(defensive_containment_rating, 1.0), 0.5), 1.8) as opponent_strength_adjustment,
        defensive_containment_rating,
        coalesce(opponent_elo_general_pre, 1500.0) as defensive_elo,
        team_elo_general_pre,
        opponent_elo_general_pre,
        team_elo_attack_pre,
        team_elo_defense_pre,
        opponent_elo_attack_pre,
        opponent_elo_defense_pre,
        expected_goals_for_pre,
        expected_goals_against_pre,
        player_offensive_modifier_pre,
        player_defensive_modifier_pre,
        player_offensive_elo_pre,
        player_defensive_elo_pre,
        player_offensive_rating_pre,
        player_defensive_rating_pre,
        missed_fixture_count_pre,
        team_lineup_attack_strength,
        team_lineup_defense_strength,
        formation_win_rate_pre,
        formation_count_pre,
        formation_matchup_win_rate_pre,
        formation_matchup_count_pre
    from enriched
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
    opponent_formation,

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
    team_elo_general_pre,
    opponent_elo_general_pre,
    team_elo_attack_pre,
    team_elo_defense_pre,
    opponent_elo_attack_pre,
    opponent_elo_defense_pre,
    expected_goals_for_pre,
    expected_goals_against_pre,
    player_offensive_modifier_pre,
    player_defensive_modifier_pre,
    player_offensive_elo_pre,
    player_defensive_elo_pre,
    player_offensive_rating_pre,
    player_defensive_rating_pre,
    missed_fixture_count_pre,
    team_lineup_attack_strength,
    team_lineup_defense_strength,
    formation_win_rate_pre,
    formation_count_pre,
    formation_matchup_win_rate_pre,
    formation_matchup_count_pre,

    current_timestamp() as updated_at_utc
from featured
