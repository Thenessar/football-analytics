select *
from {{ ref('fct_football__player_shot_features') }}
where shots_total is null
   or shots_on is null
   or dribbles_attempts is null
   or fouls_committed is null
   or offsides is null
   or tackles_interceptions is null
   or shots_total_l5_p90 is null
   or dribbles_attempts_l5_p90 is null
   or fouls_committed_l5_p90 is null
   or team_elo_attack_pre is null
   or opponent_elo_defense_pre is null
   or player_offensive_elo_pre is null
   or player_defensive_elo_pre is null
   or team_lineup_attack_strength is null
   or team_lineup_defense_strength is null
