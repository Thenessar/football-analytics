select *
from {{ ref('stg_football_player_match_stats') }}
where games_minutes <= 0
