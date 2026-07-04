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
