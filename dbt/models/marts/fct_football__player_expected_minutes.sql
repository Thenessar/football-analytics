with fixtures as (
    select
        fixture_id,
        fixture_date_utc
    from {{ ref('stg_football__fixtures') }}
),

lineup_players as (
    select
        lineups.fixture_id,
        fixtures.fixture_date_utc,
        lineups.team_id,
        lineups.team_name,
        lineups.player_id,
        lineups.player_name,
        lineups.position,
        lineups.is_starting,
        lineups.formation
    from {{ ref('stg_football__lineups') }} as lineups
    inner join fixtures
        on lineups.fixture_id = fixtures.fixture_id
),

-- Played appearances only (same 1..130 minute rule as the player feature
-- mart); these are the historical observations the estimator draws from.
appearances as (
    select
        player_stats.player_id,
        player_stats.fixture_id,
        fixtures.fixture_date_utc,
        cast(player_stats.games_minutes as double) as minutes_played
    from {{ ref('stg_football__player_match_stats') }} as player_stats
    inner join fixtures
        on player_stats.fixture_id = fixtures.fixture_id
    where coalesce(player_stats.games_minutes, 0) between 1 and 130
),

-- Last 5 appearances strictly prior to the target fixture. Ties on the same
-- date order by fixture_id so the current fixture can never join to itself.
prior_appearances as (
    select
        lineup_players.fixture_id,
        lineup_players.team_id,
        lineup_players.player_id,
        appearances.minutes_played,
        row_number() over (
            partition by lineup_players.fixture_id, lineup_players.team_id, lineup_players.player_id
            order by appearances.fixture_date_utc desc, appearances.fixture_id desc
        ) as recency_rank
    from lineup_players
    inner join appearances
        on lineup_players.player_id = appearances.player_id
       and (
            appearances.fixture_date_utc < lineup_players.fixture_date_utc
            or (
                appearances.fixture_date_utc = lineup_players.fixture_date_utc
                and appearances.fixture_id < lineup_players.fixture_id
            )
       )
),

history_aggregates as (
    select
        fixture_id,
        team_id,
        player_id,
        count(minutes_played) as prior_appearance_count,
        sum(minutes_played) as minutes_sum_l5,
        min(minutes_played) as minutes_min_l5,
        max(minutes_played) as minutes_max_l5,
        avg(minutes_played) as minutes_avg_l5
    from prior_appearances
    where recency_rank <= 5
    group by fixture_id, team_id, player_id
),

estimated as (
    select
        lineup_players.fixture_id,
        lineup_players.fixture_date_utc,
        lineup_players.team_id,
        lineup_players.team_name,
        lineup_players.player_id,
        lineup_players.player_name,
        lineup_players.position,
        case
            when upper(lineup_players.position) in ('G', 'GK', 'GOALKEEPER') then 'G'
            when upper(lineup_players.position) in ('D', 'DEFENDER', 'DEFENCE', 'CB', 'LB', 'RB', 'LWB', 'RWB') then 'D'
            when upper(lineup_players.position) in ('F', 'A', 'ATTACKER', 'FORWARD', 'STRIKER', 'CF', 'LW', 'RW') then 'F'
            else 'M'
        end as position_group,
        lineup_players.is_starting,
        lineup_players.formation,
        coalesce(history_aggregates.prior_appearance_count, 0) as prior_appearance_count,
        history_aggregates.minutes_avg_l5,
        -- Trimmed mean over the last 5 appearances: drop the single highest
        -- and lowest observation when at least 3 exist, so one red-card or
        -- late-cameo outlier cannot skew the estimate. With 1-2 observations
        -- fall back to the plain average; with none, to a documented
        -- role-based prior (starter 85, substitute 15).
        case
            when coalesce(history_aggregates.prior_appearance_count, 0) >= 3
            then (history_aggregates.minutes_sum_l5 - history_aggregates.minutes_min_l5 - history_aggregates.minutes_max_l5)
                 / (history_aggregates.prior_appearance_count - 2)
            when coalesce(history_aggregates.prior_appearance_count, 0) >= 1
            then history_aggregates.minutes_avg_l5
            when lineup_players.is_starting then cast(85.0 as double)
            else cast(15.0 as double)
        end as expected_minutes_raw
    from lineup_players
    left join history_aggregates
        on lineup_players.fixture_id = history_aggregates.fixture_id
       and lineup_players.team_id = history_aggregates.team_id
       and lineup_players.player_id = history_aggregates.player_id
)

select
    fixture_id,
    fixture_date_utc,
    team_id,
    team_name,
    player_id,
    player_name,
    position,
    position_group,
    is_starting,
    formation,
    prior_appearance_count,
    minutes_avg_l5,
    least(greatest(expected_minutes_raw, 0.0), 120.0) as expected_minutes,
    case
        when prior_appearance_count >= 3 then 'l5_trimmed_mean'
        when prior_appearance_count >= 1 then 'l5_plain_average'
        else 'role_prior'
    end as expected_minutes_method,
    current_timestamp() as updated_at_utc
from estimated
