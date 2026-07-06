-- Rolling L5 features may be null only when the team has zero prior fixtures
-- with statistics coverage inside the window. Any row with prior covered
-- fixtures but a null rolling average signals a broken window definition.
with windowed as (
    select
        fixture_id,
        team_id,
        team_possession_l5_avg,
        count(possession_pct) over (
            partition by team_id
            order by fixture_date_utc, fixture_id
            rows between 5 preceding and 1 preceding
        ) as prior_covered_fixtures
    from {{ ref('fct_football__team_match_stats_context') }}
)

select *
from windowed
where prior_covered_fixtures > 0
  and team_possession_l5_avg is null
