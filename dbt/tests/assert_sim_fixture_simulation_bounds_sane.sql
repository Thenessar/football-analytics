-- Simulation summary sanity (ml_upgrade_backlog.md N4): non-negative means,
-- probabilities in [0,1] and non-increasing across thresholds, percentiles
-- monotone. The fixture-level scoreline row carries nulls by design and is
-- excluded.

select *
from {{ source('gold_predictions', 'sim_football__fixture_simulation') }}
where entity_type != 'fixture'
  and (
    sim_mean is null or sim_mean < 0
    or sim_std is null or sim_std < 0
    or p_ge_1 is null or p_ge_1 < 0 or p_ge_1 > 1
    or p_ge_2 is null or p_ge_2 < 0 or p_ge_2 > 1
    or p_ge_3 is null or p_ge_3 < 0 or p_ge_3 > 1
    or p_ge_1 < p_ge_2
    or p_ge_2 < p_ge_3
    or p05 > p25
    or p25 > p50
    or p50 > p75
    or p75 > p95
  )
