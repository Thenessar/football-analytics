select *
from {{ ref('fct_football__player_expected_minutes') }}
where expected_minutes is null
   or expected_minutes < 0
   or expected_minutes > 120
   or expected_minutes_if_plays is null
   or expected_minutes_if_plays < 0
   or expected_minutes_if_plays > 120
   or p_plays is null
   or p_plays < 0
   or p_plays > 1
   -- Starters are on the pitch at kickoff by definition.
   or (is_starting and p_plays != 1.0)
