select *
from {{ ref('stg_football__team_match_stats') }}
where (possession_pct is not null and (possession_pct < 0 or possession_pct > 100))
   or (passes_pct_team is not null and (passes_pct_team < 0 or passes_pct_team > 100))
