with players as (
    select * from {{ ref('fct_football__player_match_features') }}
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
from with_priors
