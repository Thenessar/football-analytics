# ADR 0003: Sparse / Special Target Handling (v1)

- **Status:** Accepted
- **Date:** 2026-07-06
- **Relates to:** business_logic.md §2, §13.3; implementation_backlog.md ticket F2

## Scope

How the v1 training pipeline handles targets that are position-gated
(`goals_saves`) or sparse (`cards_yellow`, `cards_red`, `goals_total`,
`goals_assists`).

## Decisions

### `goals_saves` — goalkeeper-gated (implemented)

- Training rows are restricted to goalkeepers
  (`filter_rows_for_target` in `football_analytics/ml_training.py`; uses
  `is_goalkeeper`, falling back to `position_group = 'G'`).
- The registered model wrapper carries `goalkeeper_only=True`; the inference
  job must emit a **structural zero** (mean 0, p_ge_k 0) for non-goalkeepers
  instead of calling the model.

### Sparse count targets — Poisson baseline accepted for v1

`cards_yellow`, `cards_red`, `goals_total`, `goals_assists` train with the
same exposure-offset Poisson LightGBM as the volume targets. Rationale:

- Poisson with a log-exposure offset is well-defined for rare counts; the
  predicted means are simply small.
- The known failure mode is *calibration* of threshold probabilities
  (p_ge_1 for red cards especially), not ranking. That is measurable from
  stored predictions (Epic H) before it is worth extra model machinery.
- Follow-up (post-v1, once Epic H/I provide observed-vs-predicted data):
  evaluate binary classification + isotonic/Platt calibration for
  `cards_red` and `cards_yellow`, and a zero-inflated or hurdle variant for
  `goals_total`/`goals_assists` if calibration plots show systematic bias.

#### M4/F-1 decision (2026-07-12): Poisson/NB kept — no hurdle for any target

The pre-committed rule (ml_upgrade_backlog M4): keep NB when `p_ge_1` ECE
≤ 0.02. Measured on the L2 rolling-origin backtest (3 folds, FA-104 feature
set, serving-parity-verified rows) and corroborated by the FA-107 retrain
validation split:

| target | ece_ge_1, backtest cross-fold mean | ece_ge_1, retrain split |
| --- | --- | --- |
| cards_red | 0.0005 | 0.0003 |
| goals_assists | 0.0044 | 0.0039 |
| goals_total | 0.0066 | 0.0028 |
| cards_yellow | 0.0098 | 0.0106 |

All four pass with at least 2× headroom, so the calibration concern this
ADR flagged is answered by measurement: the exposure-offset Poisson mean +
NB dispersion stack stays. Revisit only if the live calibration monitor
(`mon_football__prediction_calibration`) shows systematic direction.

### Position-group-specific models — deferred for v1

One model per target across all positions, with position signal carried by
features (`position_group` gating for saves, per-90 history, formation grid).
Splitting every target by position group would multiply the model count by
up to 4 while shrinking already-small international-football training sets.
Revisit when F4's per-position-group validation metrics show a target where
one position group is consistently mis-calibrated.
