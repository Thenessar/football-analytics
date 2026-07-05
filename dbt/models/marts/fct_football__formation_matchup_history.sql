with team_match_context as (
    select * from {{ ref('fct_football__team_match_context') }}
),

matchup_outcomes as (
    select
        fixture_id,
        fixture_date_utc,
        team_id,
        team_name,
        opponent_team_id,
        opponent_team_name,
        team_formation,
        opponent_formation,
        case
            when goals_for > goals_against then 1.0
            when goals_for = goals_against then 0.5
            else 0.0
        end as points
    from team_match_context
),

historical_matchups as (
    -- Non-equi join to compute cumulative win rate point-in-time without data leakage
    select
        cur.fixture_id,
        cur.team_id,
        
        -- Overall formation win rate prior to this match
        coalesce(
            avg(hist_all.points),
            0.5
        ) as formation_win_rate_pre,
        coalesce(
            count(hist_all.fixture_id),
            0
        ) as formation_count_pre,
        
        -- Specific matchup win rate prior to this match
        coalesce(
            avg(hist_match.points),
            0.5
        ) as formation_matchup_win_rate_pre,
        coalesce(
            count(hist_match.fixture_id),
            0
        ) as formation_matchup_count_pre
        
    from matchup_outcomes cur
    left join matchup_outcomes hist_all
        on (
            hist_all.fixture_date_utc < cur.fixture_date_utc
            or (hist_all.fixture_date_utc = cur.fixture_date_utc and hist_all.fixture_id < cur.fixture_id)
        )
       and hist_all.team_formation = cur.team_formation
       and cur.team_formation is not null
    left join matchup_outcomes hist_match
        on (
            hist_match.fixture_date_utc < cur.fixture_date_utc
            or (hist_match.fixture_date_utc = cur.fixture_date_utc and hist_match.fixture_id < cur.fixture_id)
        )
       and hist_match.team_formation = cur.team_formation
       and hist_match.opponent_formation = cur.opponent_formation
       and cur.team_formation is not null
       and cur.opponent_formation is not null
    group by
        cur.fixture_id,
        cur.team_id
)

select * from historical_matchups
