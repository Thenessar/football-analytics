{{ config(materialized='view') }}

-- Aggregated monitoring surface for §16.3: model metrics by target and
-- position group, calibration for sparse events (avg predicted P(>=1) vs the
-- empirical rate), prediction drift by competition, and the active model
-- version producing each segment. Query with coarser GROUP BYs as needed.

select
    target_event,
    position_group,
    league_id,
    league_name,
    model_name,
    model_version,
    count(*) as prediction_count,
    avg(actual_count) as mean_actual,
    avg(predicted_mean) as mean_predicted,
    avg(predicted_mean) - avg(actual_count) as mean_bias,
    avg(abs(predicted_mean - actual_count)) as mae,
    avg(predicted_p_ge_1) as avg_predicted_p_ge_1,
    avg(case when actual_count >= 1 then 1.0 else 0.0 end) as empirical_p_ge_1,
    avg(predicted_p_ge_2) as avg_predicted_p_ge_2,
    avg(case when actual_count >= 2 then 1.0 else 0.0 end) as empirical_p_ge_2,
    min(fixture_date_utc) as first_fixture_date_utc,
    max(fixture_date_utc) as last_fixture_date_utc
from {{ ref('mon_football__prediction_vs_actual') }}
group by
    target_event,
    position_group,
    league_id,
    league_name,
    model_name,
    model_version
