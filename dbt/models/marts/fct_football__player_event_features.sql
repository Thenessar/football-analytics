-- General player-event feature mart (design: docs/design/player_event_features_schema.md).
-- One row per fixture/team/player: completed fixtures carry labels for
-- training, future fixtures with confirmed lineups carry null labels for
-- inference. All predictors are pre-match safe: Elo columns are _pre
-- variants and rolling history comes from strictly-prior played appearances.
-- Player-Elo columns join fixture-exact snapshots for completed rows and the
-- per-player current-state rows for inference rows (FA-101), so both paths
-- carry the same feature family the registered models were trained on.

with played as (
    select * from {{ ref('fct_football__player_match_features') }}
),

lineups as (
    select * from {{ ref('stg_football__lineups') }}
),

team_context as (
    select * from {{ ref('fct_football__team_match_context') }}
),

training_rows as (
    select
        fixture_id,
        fixture_date_utc,
        league_id,
        league_name,
        league_season,
        status_short,
        home_away,
        team_id,
        team_name,
        team_name_normalized,
        opponent_team_id,
        opponent_team_name,
        opponent_team_name_normalized,
        player_id,
        player_name,
        player_name_normalized,
        is_starter as is_starting,
        primary_position,
        position_group,
        is_goalkeeper,
        formation,
        formation_grid,
        formation_row,
        formation_column,
        opponent_formation,
        game_importance_scalar,
        true as is_completed_fixture,
        cast(games_minutes as double) as games_minutes,
        cast(offsides as double) as offsides,
        cast(shots_total as double) as shots_total,
        cast(shots_on as double) as shots_on,
        cast(goals_total as double) as goals_total,
        cast(goals_assists as double) as goals_assists,
        cast(goals_saves as double) as goals_saves,
        cast(passes_total as double) as passes_total,
        cast(fouls_drawn as double) as fouls_drawn,
        cast(fouls_committed as double) as fouls_committed,
        cast(cards_yellow as double) as cards_yellow,
        cast(cards_red as double) as cards_red
    from played
),

inference_rows as (
    select
        team_context.fixture_id,
        team_context.fixture_date_utc,
        team_context.league_id,
        team_context.league_name,
        team_context.league_season,
        team_context.status_short,
        team_context.home_away,
        team_context.team_id,
        team_context.team_name,
        team_context.team_name_normalized,
        team_context.opponent_team_id,
        team_context.opponent_team_name,
        team_context.opponent_team_name_normalized,
        lineups.player_id,
        lineups.player_name,
        {{ normalize_name('lineups.player_name') }} as player_name_normalized,
        lineups.is_starting,
        lineups.position as primary_position,
        case
            when upper(lineups.position) in ('G', 'GK', 'GOALKEEPER') then 'G'
            when upper(lineups.position) in ('D', 'DEFENDER', 'DEFENCE', 'CB', 'LB', 'RB', 'LWB', 'RWB') then 'D'
            when upper(lineups.position) in ('F', 'A', 'ATTACKER', 'FORWARD', 'STRIKER', 'CF', 'LW', 'RW') then 'F'
            else 'M'
        end as position_group,
        case
            when upper(lineups.position) in ('G', 'GK', 'GOALKEEPER') then true
            else false
        end as is_goalkeeper,
        lineups.formation,
        lineups.formation_grid,
        lineups.formation_row,
        lineups.formation_column,
        team_context.opponent_formation,
        coalesce(team_context.game_importance_scalar, 1.0) as game_importance_scalar,
        false as is_completed_fixture,
        cast(null as double) as games_minutes,
        cast(null as double) as offsides,
        cast(null as double) as shots_total,
        cast(null as double) as shots_on,
        cast(null as double) as goals_total,
        cast(null as double) as goals_assists,
        cast(null as double) as goals_saves,
        cast(null as double) as passes_total,
        cast(null as double) as fouls_drawn,
        cast(null as double) as fouls_committed,
        cast(null as double) as cards_yellow,
        cast(null as double) as cards_red
    from lineups
    inner join team_context
        on lineups.fixture_id = team_context.fixture_id
       and lineups.team_id = team_context.team_id
    where team_context.status_short not in ('FT', 'AET', 'PEN')
),

-- A stale fixture status could otherwise yield both a played (training) row
-- and a lineup-derived (inference) row for the same player; labels win.
inference_rows_without_labels as (
    select inference_rows.*
    from inference_rows
    left anti join training_rows
        on inference_rows.fixture_id = training_rows.fixture_id
       and inference_rows.team_id = training_rows.team_id
       and inference_rows.player_id = training_rows.player_id
),

base as (
    select * from training_rows
    union all
    select * from inference_rows_without_labels
),

-- Last 5 played appearances strictly prior to the row's fixture. A ranked
-- join (rather than a row-offset window over base) guarantees intervening
-- scheduled fixtures never consume window slots and the current fixture can
-- never feed its own features.
prior_appearances as (
    select
        base.fixture_id,
        base.team_id,
        base.player_id,
        played.games_minutes,
        played.offsides,
        played.shots_total,
        played.shots_on,
        played.goals_total,
        played.goals_assists,
        played.goals_saves,
        played.passes_total,
        played.fouls_drawn,
        played.fouls_committed,
        played.cards_yellow,
        played.cards_red,
        row_number() over (
            partition by base.fixture_id, base.team_id, base.player_id
            order by played.fixture_date_utc desc, played.fixture_id desc
        ) as recency_rank
    from base
    inner join played
        on base.player_id = played.player_id
       and (
            played.fixture_date_utc < base.fixture_date_utc
            or (
                played.fixture_date_utc = base.fixture_date_utc
                and played.fixture_id < base.fixture_id
            )
       )
),

history_aggregates as (
    select
        fixture_id,
        team_id,
        player_id,
        count(*) as appearances_l5_count,
        sum(games_minutes) as minutes_l5,
        sum(offsides) as offsides_l5_count,
        sum(shots_total) as shots_total_l5_count,
        sum(shots_on) as shots_on_l5_count,
        sum(goals_total) as goals_total_l5_count,
        sum(goals_assists) as goals_assists_l5_count,
        sum(goals_saves) as goals_saves_l5_count,
        sum(passes_total) as passes_total_l5_count,
        sum(fouls_drawn) as fouls_drawn_l5_count,
        sum(fouls_committed) as fouls_committed_l5_count,
        sum(cards_yellow) as cards_yellow_l5_count,
        sum(cards_red) as cards_red_l5_count
    from prior_appearances
    where recency_rank <= 5
    group by fixture_id, team_id, player_id
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
    where fixture_id is not null
),

-- Current per-player modifier state after all played fixtures (FA-101).
-- Player-Elo history only has rows for played appearances, so future
-- fixtures can never join fixture-exact; the current state is by
-- construction their pre-match state, and the family is recomputed below
-- against the future fixture's team Elo row exactly as training defines it.
player_elo_current as (
    select
        team_id,
        player_id,
        player_offensive_modifier_pre as current_offensive_modifier,
        player_defensive_modifier_pre as current_defensive_modifier,
        missed_fixture_count_pre as current_missed_fixture_count
    from {{ ref('fct_football__player_elo_history') }}
    where is_current_state
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

team_flow as (
    select
        fixture_id,
        team_id,
        team_possession_l5_avg,
        opponent_possession_l5_avg,
        expected_possession_share,
        team_passes_l5_per_match,
        opponent_passes_allowed_l5,
        team_shots_l5,
        opponent_shots_allowed_l5,
        team_fouls_l5,
        opponent_fouls_drawn_allowed_l5,
        elo_expected_score,
        elo_possession_interaction,
        formation_possession_profile
    from {{ ref('fct_football__team_match_stats_context') }}
),

expected_minutes as (
    select
        fixture_id,
        team_id,
        player_id,
        expected_minutes,
        expected_minutes_method,
        p_plays,
        expected_minutes_if_plays
    from {{ ref('fct_football__player_expected_minutes') }}
)

select
    base.fixture_id,
    base.fixture_date_utc,
    base.league_id,
    base.league_name,
    base.league_season,
    base.status_short,
    base.home_away,
    base.is_completed_fixture,

    base.team_id,
    base.team_name,
    base.team_name_normalized,
    base.opponent_team_id,
    base.opponent_team_name,
    base.opponent_team_name_normalized,
    coalesce(team_elo.team_elo_general_pre, 1500.0) as team_elo_general_pre,
    coalesce(team_elo.opponent_elo_general_pre, 1500.0) as opponent_elo_general_pre,
    coalesce(team_elo.team_elo_attack_pre, 0.0) as team_elo_attack_pre,
    coalesce(team_elo.team_elo_defense_pre, 0.0) as team_elo_defense_pre,
    coalesce(team_elo.opponent_elo_attack_pre, 0.0) as opponent_elo_attack_pre,
    coalesce(team_elo.opponent_elo_defense_pre, 0.0) as opponent_elo_defense_pre,
    coalesce(team_elo.expected_goals_for_pre, 1.25) as expected_goals_for_pre,
    coalesce(team_elo.expected_goals_against_pre, 1.25) as expected_goals_against_pre,
    base.game_importance_scalar,

    base.player_id,
    base.player_name,
    base.player_name_normalized,
    base.is_starting,
    base.primary_position,
    base.position_group,
    base.is_goalkeeper,
    base.formation,
    base.formation_grid,
    base.formation_row,
    base.formation_column,
    base.opponent_formation,
    coalesce(lineup_strength.starting_xi_attack_elo, team_elo.team_elo_attack_pre, 0.0) as team_lineup_attack_strength,
    coalesce(lineup_strength.starting_xi_defense_elo, team_elo.team_elo_defense_pre, 0.0) as team_lineup_defense_strength,
    coalesce(formation_history.formation_win_rate_pre, 0.5) as formation_win_rate_pre,
    coalesce(formation_history.formation_count_pre, 0) as formation_count_pre,
    coalesce(formation_history.formation_matchup_win_rate_pre, 0.5) as formation_matchup_win_rate_pre,
    coalesce(formation_history.formation_matchup_count_pre, 0) as formation_matchup_count_pre,

    -- Fixture-exact player Elo for completed rows; current-state recompute
    -- for inference rows (FA-101): elo/rating = team baseline + modifier,
    -- mirroring build_player_elo_history's snapshot formula.
    coalesce(player_elo.player_offensive_modifier_pre, player_elo_current.current_offensive_modifier, 0.0) as player_offensive_modifier_pre,
    coalesce(player_elo.player_defensive_modifier_pre, player_elo_current.current_defensive_modifier, 0.0) as player_defensive_modifier_pre,
    coalesce(
        player_elo.player_offensive_elo_pre,
        team_elo.team_elo_attack_pre + player_elo_current.current_offensive_modifier,
        team_elo.team_elo_attack_pre,
        0.0
    ) as player_offensive_elo_pre,
    coalesce(
        player_elo.player_defensive_elo_pre,
        team_elo.team_elo_defense_pre + player_elo_current.current_defensive_modifier,
        team_elo.team_elo_defense_pre,
        0.0
    ) as player_defensive_elo_pre,
    coalesce(
        player_elo.player_offensive_rating_pre,
        player_elo.player_offensive_elo_pre,
        team_elo.team_elo_attack_pre + player_elo_current.current_offensive_modifier,
        team_elo.team_elo_attack_pre,
        0.0
    ) as player_offensive_rating_pre,
    coalesce(
        player_elo.player_defensive_rating_pre,
        player_elo.player_defensive_elo_pre,
        team_elo.team_elo_defense_pre + player_elo_current.current_defensive_modifier,
        team_elo.team_elo_defense_pre,
        0.0
    ) as player_defensive_rating_pre,
    coalesce(player_elo.missed_fixture_count_pre, player_elo_current.current_missed_fixture_count, 0) as missed_fixture_count_pre,

    coalesce(history_aggregates.appearances_l5_count, 0) as appearances_l5_count,
    coalesce(history_aggregates.minutes_l5, 0.0) as minutes_l5,
    coalesce(history_aggregates.offsides_l5_count, 0.0) as offsides_l5_count,
    coalesce(history_aggregates.shots_total_l5_count, 0.0) as shots_total_l5_count,
    coalesce(history_aggregates.shots_on_l5_count, 0.0) as shots_on_l5_count,
    coalesce(history_aggregates.goals_total_l5_count, 0.0) as goals_total_l5_count,
    coalesce(history_aggregates.goals_assists_l5_count, 0.0) as goals_assists_l5_count,
    coalesce(history_aggregates.goals_saves_l5_count, 0.0) as goals_saves_l5_count,
    coalesce(history_aggregates.passes_total_l5_count, 0.0) as passes_total_l5_count,
    coalesce(history_aggregates.fouls_drawn_l5_count, 0.0) as fouls_drawn_l5_count,
    coalesce(history_aggregates.fouls_committed_l5_count, 0.0) as fouls_committed_l5_count,
    coalesce(history_aggregates.cards_yellow_l5_count, 0.0) as cards_yellow_l5_count,
    coalesce(history_aggregates.cards_red_l5_count, 0.0) as cards_red_l5_count,
    {% for event in [
        'offsides', 'shots_total', 'shots_on', 'goals_total', 'goals_assists',
        'goals_saves', 'passes_total', 'fouls_drawn', 'fouls_committed',
        'cards_yellow', 'cards_red'
    ] %}
    case
        when coalesce(history_aggregates.minutes_l5, 0) > 0
        then coalesce(history_aggregates.{{ event }}_l5_count, 0.0) * 90.0 / history_aggregates.minutes_l5
        else 0.0
    end as {{ event }}_l5_p90,
    {% endfor %}

    team_flow.team_possession_l5_avg,
    team_flow.opponent_possession_l5_avg,
    coalesce(team_flow.expected_possession_share, 0.5) as expected_possession_share,
    team_flow.team_passes_l5_per_match,
    team_flow.opponent_passes_allowed_l5,
    team_flow.team_shots_l5,
    team_flow.opponent_shots_allowed_l5,
    team_flow.team_fouls_l5,
    team_flow.opponent_fouls_drawn_allowed_l5,
    team_flow.elo_expected_score,
    team_flow.elo_possession_interaction,
    team_flow.formation_possession_profile,

    base.games_minutes,
    case when base.games_minutes is not null then greatest(base.games_minutes / 90.0, 0.000001) end as exposure,
    expected_minutes.expected_minutes,
    expected_minutes.expected_minutes_method,
    case when expected_minutes.expected_minutes is not null then greatest(expected_minutes.expected_minutes / 90.0, 0.000001) end as expected_exposure,
    -- Role-conditional decomposition (ADR 0004): the simulation layer samples
    -- bench participation from p_plays and scales rates by minutes-if-plays.
    expected_minutes.p_plays,
    expected_minutes.expected_minutes_if_plays,

    base.offsides,
    base.shots_total,
    base.shots_on,
    base.goals_total,
    base.goals_assists,
    base.goals_saves,
    base.passes_total,
    base.fouls_drawn,
    base.fouls_committed,
    base.cards_yellow,
    base.cards_red,

    current_timestamp() as updated_at_utc
from base
left join history_aggregates
    on base.fixture_id = history_aggregates.fixture_id
   and base.team_id = history_aggregates.team_id
   and base.player_id = history_aggregates.player_id
left join team_elo
    on base.fixture_id = team_elo.fixture_id
   and base.team_id = team_elo.team_id
left join player_elo
    on base.fixture_id = player_elo.fixture_id
   and base.team_id = player_elo.team_id
   and base.player_id = player_elo.player_id
-- Current-state fallback is restricted to inference rows: a completed row
-- falling back to post-history state would leak its own fixture's update.
left join player_elo_current
    on base.team_id = player_elo_current.team_id
   and base.player_id = player_elo_current.player_id
   and not base.is_completed_fixture
left join lineup_strength
    on base.fixture_id = lineup_strength.fixture_id
   and base.team_id = lineup_strength.team_id
left join formation_history
    on base.fixture_id = formation_history.fixture_id
   and base.team_id = formation_history.team_id
left join team_flow
    on base.fixture_id = team_flow.fixture_id
   and base.team_id = team_flow.team_id
left join expected_minutes
    on base.fixture_id = expected_minutes.fixture_id
   and base.team_id = expected_minutes.team_id
   and base.player_id = expected_minutes.player_id
