with players as (
    select
        *,
        case
            when upper(games_position) in ('F', 'A', 'ATTACKER', 'FORWARD', 'STRIKER') then 'F'
            when upper(games_position) in ('D', 'DEFENDER', 'DEFENCE') then 'D'
            when upper(games_position) in ('G', 'GK', 'GOALKEEPER') then 'G'
            else 'M'
        end as position_group
    from {{ ref('stg_football_player_match_stats') }}
),

with_priors as (
    select
        *,
        case
            when position_group = 'F' then 2.6 * 3.0
            when position_group = 'M' then 1.2 * 3.0
            when position_group = 'D' then 0.5 * 3.0
            else 0.0
        end as prior_alpha,
        270.0 as prior_beta
    from players
),

with_context as (
    select
        with_priors.*,
        coalesce(context.game_importance_scalar, 1.0) as game_importance_scalar,
        coalesce(context.opponent_strength_adjustment, 1.0) as opponent_strength_adjustment,
        coalesce(context.defensive_containment_rating, 1.0) as defensive_containment_rating,
        coalesce(context.defensive_elo, 1500.0) as defensive_elo
    from with_priors
    left join {{ ref('football_team_match_context') }} as context
        on with_priors.fixture_id = context.fixture_id
       and with_priors.team_id = context.team_id
)

select
    *,
    case
        when prior_beta + games_minutes > 0
            then (prior_alpha + shots_total) / (prior_beta + games_minutes)
        else 0.0
    end as shot_rate_smoothed_per_minute,
    game_importance_scalar * opponent_strength_adjustment as sapm_interaction_weight,
    game_importance_scalar * opponent_strength_adjustment * shots_total as weighted_shots,
    game_importance_scalar * opponent_strength_adjustment * games_minutes as weighted_minutes
from with_context
