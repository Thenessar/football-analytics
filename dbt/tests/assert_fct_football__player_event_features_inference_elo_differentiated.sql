-- Serving-skew guard (prediction_quality_backlog FA-101 / E-2): any
-- non-completed fixture/team with a full confirmed lineup (>= 11 rows) must
-- carry more than one distinct player_offensive_modifier_pre. A single
-- distinct value means the current-state join degenerated and every player
-- collapsed back to the team average — the exact bug behind the flat
-- France vs Morocco predictions. Passes trivially with no upcoming fixtures.
select
    fixture_id,
    team_id,
    count(*) as lineup_row_count,
    count(distinct player_offensive_modifier_pre) as distinct_modifier_count
from {{ ref('fct_football__player_event_features') }}
where not is_completed_fixture
group by fixture_id, team_id
having count(*) >= 11
   and count(distinct player_offensive_modifier_pre) <= 1
