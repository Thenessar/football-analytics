# Design: `fct_football__player_event_features` (backlog D1)

- **Date:** 2026-07-06
- **Relates to:** business_logic.md §2, §11.3, §11.4, §13.4; implementation_backlog.md Epic D

## Purpose

One general, model-ready mart with one row per fixture/team/player serving both
training (completed fixtures, labels populated) and inference (future fixtures
with confirmed lineups, labels null). Supersedes the shot-specific mart as the
canonical training source.

## Decision: fold in `fct_football__player_shot_features`

**Fold in.** The event mart is a strict superset (same joins plus team flow,
expected minutes, and all 11 labels), so keeping the shot mart standalone would
duplicate the join graph and rolling-window logic and invite drift. Removal is a
fast-follow: `football_analytics/ml_training.py` and
`scripts/train_poisson_lgbm.py` still read the shot mart until F1 repoints
them, so the model is deprecated now and deleted in J1 after Epic F lands.

## Row population

| Row type | Source | Labels | Filter |
| --- | --- | --- | --- |
| Training | `fct_football__player_match_features` (played rows, minutes 1–130) | populated | fixture status in FT/AET/PEN |
| Inference | `stg_football__lineups` × `fct_football__team_match_context` | null | fixture status **not** in FT/AET/PEN |

Unused substitutes in completed fixtures appear in neither set (no minutes, no
labels) — consistent with the existing training convention. Inference rows
exist only once lineups are confirmed, matching §14.4.

`is_completed_fixture` (boolean) marks the split for consumers.

## Column inventory

**Keys / fixture context** — `fixture_id`, `fixture_date_utc`, `league_id`,
`league_name`, `league_season`, `status_short`, `home_away`,
`is_completed_fixture`.

**Team / opponent context** — `team_id`, `team_name`, `team_name_normalized`,
`opponent_team_id`, `opponent_team_name`, `opponent_team_name_normalized`,
`team_elo_general_pre`, `opponent_elo_general_pre`, `team_elo_attack_pre`,
`team_elo_defense_pre`, `opponent_elo_attack_pre`, `opponent_elo_defense_pre`,
`expected_goals_for_pre`, `expected_goals_against_pre`,
`game_importance_scalar` (all Elo values are `_pre` variants per §13.4).

**Lineup context** — `is_starting`, `primary_position`, `position_group`
(G/D/M/F), `is_goalkeeper`, `formation`, `formation_grid`, `formation_row`,
`formation_column`, `opponent_formation`, `team_lineup_attack_strength`,
`team_lineup_defense_strength`, `formation_win_rate_pre`,
`formation_count_pre`, `formation_matchup_win_rate_pre`,
`formation_matchup_count_pre`.

**Player identity / history** — `player_id`, `player_name`,
`player_name_normalized`, `player_offensive_modifier_pre`,
`player_defensive_modifier_pre`, `player_offensive_elo_pre`,
`player_defensive_elo_pre`, `player_offensive_rating_pre`,
`player_defensive_rating_pre`, `missed_fixture_count_pre`,
`appearances_l5_count`, `minutes_l5`, and per event E in the 11-target list:
`E_l5_count` and `E_l5_p90` (rates use 90 × count / minutes_l5, 0 when no
history). Windows draw from the **last 5 played appearances strictly prior**
to the row's fixture via a ranked join (not a row-offset window), so
intervening scheduled fixtures can never consume window slots or leak.

**Team flow context** (from `fct_football__team_match_stats_context`) —
`team_possession_l5_avg`, `opponent_possession_l5_avg`,
`expected_possession_share`, `team_passes_l5_per_match`,
`opponent_passes_allowed_l5`, `team_shots_l5`, `opponent_shots_allowed_l5`,
`team_fouls_l5`, `opponent_fouls_drawn_allowed_l5`, `elo_expected_score`,
`elo_possession_interaction`, `formation_possession_profile`.

**Exposure** — `games_minutes` (actual; null on inference rows), `exposure`
(games_minutes/90; null on inference rows), `expected_minutes`,
`expected_minutes_method` (from `fct_football__player_expected_minutes`),
`expected_exposure` (expected_minutes/90). Training uses actual exposure,
inference uses expected, per §13.3.

**Target labels (§2, null on inference rows)** — `offsides`, `shots_total`,
`shots_on`, `goals_total`, `goals_assists`, `goals_saves`, `passes_total`,
`fouls_drawn`, `fouls_committed`, `cards_yellow`, `cards_red`.

## Leakage rules (§13.4)

- Only `_pre` Elo/rating columns; no post-match Elo anywhere.
- All rolling player-history features exclude the current fixture by
  construction (strictly-prior ranked join, date then fixture_id tiebreak).
- Team flow L5 columns come from C1, which enforces the same rule.
- Current-match team stats (possession etc.) are **not** joined onto this mart
  at all — only their rolling derivatives.
- D3 adds an executable regression: a mocked player whose current fixture has
  extreme stats must show rolling features computed only from prior fixtures.
