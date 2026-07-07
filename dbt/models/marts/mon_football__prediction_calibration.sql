{{ config(materialized='view') }}

-- Production calibration curve data (ml_upgrade_backlog.md L5): predicted
-- P(>=k) versus empirical frequency in 0.1-wide probability bins, per
-- target/position-group/model-version segment. The offline analogue is the
-- reliability-table artifact logged by training (L3); this view measures the
-- same thing on live predictions after fixtures complete.

with vs_actual as (
    select *
    from {{ ref('mon_football__prediction_vs_actual') }}
),

thresholds as (
    select
        target_event,
        position_group,
        model_version,
        threshold,
        predicted_p,
        case when actual_count >= threshold then 1.0 else 0.0 end as event_occurred
    from vs_actual
    lateral view stack(
        2,
        1, predicted_p_ge_1,
        2, predicted_p_ge_2
    ) as threshold, predicted_p
    where predicted_p_ge_1 is not null
)

select
    target_event,
    position_group,
    model_version,
    threshold,
    least(floor(predicted_p * 10) / 10, 0.9) as bin_lower,
    least(floor(predicted_p * 10) / 10, 0.9) + 0.1 as bin_upper,
    count(*) as row_count,
    avg(predicted_p) as mean_predicted_p,
    avg(event_occurred) as empirical_frequency,
    avg(event_occurred) - avg(predicted_p) as calibration_gap
from thresholds
group by
    target_event,
    position_group,
    model_version,
    threshold,
    least(floor(predicted_p * 10) / 10, 0.9)
