select *
from {{ ref('fct_football__player_expected_minutes') }}
where expected_minutes is null
   or expected_minutes < 0
   or expected_minutes > 120
