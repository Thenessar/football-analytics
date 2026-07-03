select *
from {{ ref('fct_football__player_match_features') }}
where games_minutes < 1
   or games_minutes > 130
