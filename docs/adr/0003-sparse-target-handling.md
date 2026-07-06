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

### Position-group-specific models — deferred for v1

One model per target across all positions, with position signal carried by
features (`position_group` gating for saves, per-90 history, formation grid).
Splitting every target by position group would multiply the model count by
up to 4 while shrinking already-small international-football training sets.
Revisit when F4's per-position-group validation metrics show a target where
one position group is consistently mis-calibrated.
