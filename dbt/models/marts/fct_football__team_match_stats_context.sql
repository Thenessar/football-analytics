with team_context as (
    select
        fixture_id,
        fixture_date_utc,
        team_id,
        team_name,
        team_name_normalized,
        opponent_team_id,
        opponent_team_name,
        opponent_team_name_normalized,
        home_away,
        league_id,
        league_name,
        league_season,
        status_short,
        team_formation,
        opponent_formation
    from {{ ref('fct_football__team_match_context') }}
),

team_stats as (
    select
        fixture_id,
        team_id,
        possession_pct,
        shots_total_team,
        shots_on_team,
        shots_off_team,
        shots_blocked_team,
        shots_inside_box_team,
        shots_outside_box_team,
        passes_total_team,
        passes_accurate_team,
        passes_pct_team,
        fouls_team,
        corners_team,
        offsides_team,
        yellow_cards_team,
        red_cards_team,
        goalkeeper_saves_team,
        expected_goals_team
    from {{ ref('stg_football__team_match_stats') }}
),

team_elo as (
    select
        fixture_id,
        team_id,
        team_elo_general_pre,
        opponent_elo_general_pre
    from {{ ref('fct_football__team_elo_history') }}
),

joined as (
    select
        team_context.*,
        own_stats.possession_pct,
        own_stats.shots_total_team,
        own_stats.shots_on_team,
        own_stats.shots_off_team,
        own_stats.shots_blocked_team,
        own_stats.shots_inside_box_team,
        own_stats.shots_outside_box_team,
        own_stats.passes_total_team,
        own_stats.passes_accurate_team,
        own_stats.passes_pct_team,
        own_stats.fouls_team,
        own_stats.corners_team,
        own_stats.offsides_team,
        own_stats.yellow_cards_team,
        own_stats.red_cards_team,
        own_stats.goalkeeper_saves_team,
        own_stats.expected_goals_team,
        opponent_stats.possession_pct as possession_pct_allowed,
        opponent_stats.shots_total_team as shots_total_allowed,
        opponent_stats.shots_on_team as shots_on_allowed,
        opponent_stats.passes_total_team as passes_total_allowed,
        opponent_stats.fouls_team as fouls_allowed
    from team_context
    left join team_stats as own_stats
        on team_context.fixture_id = own_stats.fixture_id
       and team_context.team_id = own_stats.team_id
    left join team_stats as opponent_stats
        on team_context.fixture_id = opponent_stats.fixture_id
       and team_context.opponent_team_id = opponent_stats.team_id
),

-- Rolling windows use only fixtures strictly prior to the current one
-- (rows between 5 preceding and 1 preceding), matching the leakage rule
-- shared with the player feature marts.
rolled as (
    select
        *,

        avg(possession_pct) over (
            partition by team_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as possession_l5_avg,

        avg(passes_total_team) over (
            partition by team_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as passes_l5_per_match,

        avg(shots_total_team) over (
            partition by team_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as shots_l5,

        avg(fouls_team) over (
            partition by team_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as fouls_l5,

        avg(passes_total_allowed) over (
            partition by team_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as passes_allowed_l5,

        avg(shots_total_allowed) over (
            partition by team_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as shots_allowed_l5,

        avg(possession_pct) over (
            partition by team_id, team_formation
            order by fixture_date_utc, fixture_id
            rows between 10 preceding and 1 preceding
        ) as formation_possession_profile
    from joined
),

featured as (
    select
        team_rows.*,
        team_rows.possession_l5_avg as team_possession_l5_avg,
        opponent_rows.possession_l5_avg as opponent_possession_l5_avg,
        team_rows.passes_l5_per_match as team_passes_l5_per_match,
        opponent_rows.passes_allowed_l5 as opponent_passes_allowed_l5,
        team_rows.shots_l5 as team_shots_l5,
        opponent_rows.shots_allowed_l5 as opponent_shots_allowed_l5,
        team_rows.fouls_l5 as team_fouls_l5,
        -- Fouls drawn by a team equal the fouls its opponent commits, so the
        -- fouls-drawn volume the opponent "allows" is its own committed-fouls
        -- rolling average.
        opponent_rows.fouls_l5 as opponent_fouls_drawn_allowed_l5,
        case
            when team_rows.possession_l5_avg is not null
                 and opponent_rows.possession_l5_avg is not null
                 and (team_rows.possession_l5_avg + opponent_rows.possession_l5_avg) > 0
            then team_rows.possession_l5_avg
                 / (team_rows.possession_l5_avg + opponent_rows.possession_l5_avg)
            else cast(0.5 as double)
        end as expected_possession_share
    from rolled as team_rows
    left join rolled as opponent_rows
        on team_rows.fixture_id = opponent_rows.fixture_id
       and team_rows.opponent_team_id = opponent_rows.team_id
),

elo_enriched as (
    select
        featured.*,
        coalesce(team_elo.team_elo_general_pre, 1500.0) as team_elo_general_pre,
        coalesce(team_elo.opponent_elo_general_pre, 1500.0) as opponent_elo_general_pre,
        1.0 / (
            1.0 + pow(
                10.0,
                (coalesce(team_elo.opponent_elo_general_pre, 1500.0)
                 - coalesce(team_elo.team_elo_general_pre, 1500.0)) / 400.0
            )
        ) as elo_expected_score
    from featured
    left join team_elo
        on featured.fixture_id = team_elo.fixture_id
       and featured.team_id = team_elo.team_id
)

select
    fixture_id,
    fixture_date_utc,
    team_id,
    team_name,
    team_name_normalized,
    opponent_team_id,
    opponent_team_name,
    opponent_team_name_normalized,
    home_away,
    league_id,
    league_name,
    league_season,
    status_short,
    team_formation,
    opponent_formation,

    possession_pct,
    shots_total_team,
    shots_on_team,
    shots_off_team,
    shots_blocked_team,
    shots_inside_box_team,
    shots_outside_box_team,
    passes_total_team,
    passes_accurate_team,
    passes_pct_team,
    fouls_team,
    corners_team,
    offsides_team,
    yellow_cards_team,
    red_cards_team,
    goalkeeper_saves_team,
    expected_goals_team,
    possession_pct_allowed,
    shots_total_allowed,
    shots_on_allowed,
    passes_total_allowed,
    fouls_allowed,

    team_possession_l5_avg,
    opponent_possession_l5_avg,
    expected_possession_share,
    team_passes_l5_per_match,
    opponent_passes_allowed_l5,
    team_shots_l5,
    opponent_shots_allowed_l5,
    team_fouls_l5,
    opponent_fouls_drawn_allowed_l5,

    team_elo_general_pre,
    opponent_elo_general_pre,
    elo_expected_score,
    elo_expected_score * expected_possession_share as elo_possession_interaction,
    formation_possession_profile,

    current_timestamp() as updated_at_utc
from elo_enriched
