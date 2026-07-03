select *
from {{ ref('stg_football__player_match_stats') }}
where games_minutes <= 0
