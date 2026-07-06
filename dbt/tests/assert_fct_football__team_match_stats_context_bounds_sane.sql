select *
from {{ ref('fct_football__team_match_stats_context') }}
where (possession_pct is not null and (possession_pct < 0 or possession_pct > 100))
   or (possession_pct_allowed is not null and (possession_pct_allowed < 0 or possession_pct_allowed > 100))
   or (passes_pct_team is not null and (passes_pct_team < 0 or passes_pct_team > 100))
   or (team_possession_l5_avg is not null and (team_possession_l5_avg < 0 or team_possession_l5_avg > 100))
   or (opponent_possession_l5_avg is not null and (opponent_possession_l5_avg < 0 or opponent_possession_l5_avg > 100))
   or (expected_possession_share is not null and (expected_possession_share < 0 or expected_possession_share > 1))
   or (elo_expected_score is not null and (elo_expected_score < 0 or elo_expected_score > 1))
   or (elo_possession_interaction is not null and (elo_possession_interaction < 0 or elo_possession_interaction > 1))
   or (formation_possession_profile is not null and (formation_possession_profile < 0 or formation_possession_profile > 100))
