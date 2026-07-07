-- Role-conditional expected minutes (ml_upgrade_backlog.md §2.1, ticket K1).
--
--   expected_minutes = p_plays * expected_minutes_if_plays
--
-- conditioned on the player's role in TODAY'S confirmed lineup:
--   * starters:   p_plays = 1; if_plays from the last 5 prior STARTS
--                 (trimmed mean, shrunk toward the starter prior).
--   * bench:      p_plays from the last 10 prior bench namings (unused-sub
--                 namings count as 0-minute observations), shrunk toward the
--                 global bench participation prior; if_plays from the nonzero
--                 bench cameos, shrunk toward the global cameo prior.
--
-- Role history comes from stg_football__player_match_stats.games_substitute:
-- stats rows exist only for completed fixtures, so future fixtures can never
-- contribute phantom zero-minute observations, and coverage does not depend
-- on historical lineup ingestion.

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

-- Role-tagged historical observations. Bench rows keep 0 minutes (unused
-- subs are real "named but did not play" observations); starter rows require
-- 1..130 played minutes to guard against provider data errors.
role_history as (
    select
        player_stats.player_id,
        player_stats.fixture_id,
        fixtures.fixture_date_utc,
        coalesce(player_stats.games_substitute, false) as was_substitute,
        least(cast(coalesce(player_stats.games_minutes, 0) as double), 130.0) as minutes_played
    from {{ ref('stg_football__player_match_stats') }} as player_stats
    inner join fixtures
        on player_stats.fixture_id = fixtures.fixture_id
),

-- Last 5 starts strictly prior to the target fixture. Ties on the same date
-- order by fixture_id so the current fixture can never join to itself.
prior_starts as (
    select
        lineup_players.fixture_id,
        lineup_players.team_id,
        lineup_players.player_id,
        role_history.minutes_played,
        row_number() over (
            partition by lineup_players.fixture_id, lineup_players.team_id, lineup_players.player_id
            order by role_history.fixture_date_utc desc, role_history.fixture_id desc
        ) as recency_rank
    from lineup_players
    inner join role_history
        on lineup_players.player_id = role_history.player_id
       and not role_history.was_substitute
       and role_history.minutes_played between 1 and 130
       and (
            role_history.fixture_date_utc < lineup_players.fixture_date_utc
            or (
                role_history.fixture_date_utc = lineup_players.fixture_date_utc
                and role_history.fixture_id < lineup_players.fixture_id
            )
       )
),

starts_aggregates as (
    select
        fixture_id,
        team_id,
        player_id,
        count(minutes_played) as starts_count,
        sum(minutes_played) as starts_minutes_sum,
        min(minutes_played) as starts_minutes_min,
        max(minutes_played) as starts_minutes_max,
        avg(minutes_played) as starts_minutes_avg
    from prior_starts
    where recency_rank <= 5
    group by fixture_id, team_id, player_id
),

-- Last 10 bench namings strictly prior to the target fixture, zeros included.
prior_bench as (
    select
        lineup_players.fixture_id,
        lineup_players.team_id,
        lineup_players.player_id,
        role_history.minutes_played,
        row_number() over (
            partition by lineup_players.fixture_id, lineup_players.team_id, lineup_players.player_id
            order by role_history.fixture_date_utc desc, role_history.fixture_id desc
        ) as recency_rank
    from lineup_players
    inner join role_history
        on lineup_players.player_id = role_history.player_id
       and role_history.was_substitute
       and (
            role_history.fixture_date_utc < lineup_players.fixture_date_utc
            or (
                role_history.fixture_date_utc = lineup_players.fixture_date_utc
                and role_history.fixture_id < lineup_players.fixture_id
            )
       )
),

bench_aggregates as (
    select
        fixture_id,
        team_id,
        player_id,
        count(*) as bench_namings_count,
        count(case when minutes_played >= 1 then 1 end) as cameo_count,
        sum(case when minutes_played >= 1 then minutes_played end) as cameo_minutes_sum,
        min(case when minutes_played >= 1 then minutes_played end) as cameo_minutes_min,
        max(case when minutes_played >= 1 then minutes_played end) as cameo_minutes_max,
        avg(case when minutes_played >= 1 then minutes_played end) as cameo_minutes_avg
    from prior_bench
    where recency_rank <= 10
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
        coalesce(starts_aggregates.starts_count, 0) as starts_in_history_count,
        coalesce(bench_aggregates.bench_namings_count, 0) as bench_namings_in_history_count,
        -- Participation probability. Starters are on the pitch at minute 0.
        -- Bench: shrunk empirical rate over the last 10 bench namings,
        -- p = (played + k*prior) / (namings + k) with k = 4.
        case
            when lineup_players.is_starting then cast(1.0 as double)
            else (
                coalesce(bench_aggregates.cameo_count, 0)
                + 4.0 * {{ var('bench_participation_prior') }}
            ) / (coalesce(bench_aggregates.bench_namings_count, 0) + 4.0)
        end as p_plays,
        -- Minutes conditional on playing, from role-matched history:
        -- trimmed mean (drop single min + max when >=3 observations, plain
        -- average with 1-2), shrunk toward the role prior with k = 2. With no
        -- history the shrinkage degenerates to the prior itself.
        case
            when lineup_players.is_starting then
                (
                    coalesce(starts_aggregates.starts_count, 0)
                    * case
                        when coalesce(starts_aggregates.starts_count, 0) >= 3
                        then (starts_aggregates.starts_minutes_sum - starts_aggregates.starts_minutes_min - starts_aggregates.starts_minutes_max)
                             / (starts_aggregates.starts_count - 2)
                        when coalesce(starts_aggregates.starts_count, 0) >= 1
                        then starts_aggregates.starts_minutes_avg
                        else cast(0.0 as double)
                    end
                    + 2.0 * {{ var('starter_minutes_prior') }}
                ) / (coalesce(starts_aggregates.starts_count, 0) + 2.0)
            else
                (
                    coalesce(bench_aggregates.cameo_count, 0)
                    * case
                        when coalesce(bench_aggregates.cameo_count, 0) >= 3
                        then (bench_aggregates.cameo_minutes_sum - bench_aggregates.cameo_minutes_min - bench_aggregates.cameo_minutes_max)
                             / (bench_aggregates.cameo_count - 2)
                        when coalesce(bench_aggregates.cameo_count, 0) >= 1
                        then bench_aggregates.cameo_minutes_avg
                        else cast(0.0 as double)
                    end
                    + 2.0 * {{ var('bench_cameo_minutes_prior') }}
                ) / (coalesce(bench_aggregates.cameo_count, 0) + 2.0)
        end as expected_minutes_if_plays_raw
    from lineup_players
    left join starts_aggregates
        on lineup_players.fixture_id = starts_aggregates.fixture_id
       and lineup_players.team_id = starts_aggregates.team_id
       and lineup_players.player_id = starts_aggregates.player_id
    left join bench_aggregates
        on lineup_players.fixture_id = bench_aggregates.fixture_id
       and lineup_players.team_id = bench_aggregates.team_id
       and lineup_players.player_id = bench_aggregates.player_id
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
    starts_in_history_count,
    bench_namings_in_history_count,
    -- History depth backing the active branch (starts for starters, bench
    -- namings for bench players).
    case
        when is_starting then starts_in_history_count
        else bench_namings_in_history_count
    end as prior_appearance_count,
    p_plays,
    least(greatest(expected_minutes_if_plays_raw, 0.0), 120.0) as expected_minutes_if_plays,
    least(greatest(p_plays * expected_minutes_if_plays_raw, 0.0), 120.0) as expected_minutes,
    case
        when is_starting and starts_in_history_count >= 3 then 'starter_l5_trimmed_shrunk'
        when is_starting and starts_in_history_count >= 1 then 'starter_low_history_shrunk'
        when is_starting then 'starter_prior'
        when bench_namings_in_history_count >= 3 then 'bench_l10_shrunk'
        when bench_namings_in_history_count >= 1 then 'bench_low_history_shrunk'
        else 'bench_prior'
    end as expected_minutes_method,
    current_timestamp() as updated_at_utc
from estimated
