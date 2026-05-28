from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Named OT era string constants
# Display eras (5-era scheme used in API response / determine_ot_era):
OT_ERA_SUDDEN_DEATH = "SUDDEN_DEATH"
OT_ERA_MODIFIED = "MODIFIED"          # Display-only; not in training feature scheme
OT_ERA_MODIFIED_SHORT = "MODIFIED_SHORT"
OT_ERA_GUARANTEED = "GUARANTEED"      # Display-only; not in training feature scheme
OT_ERA_CURRENT = "CURRENT"
# Training eras (3-era scheme that matches the model's feature contract):
_FEATURE_ERA_SUDDEN_DEATH = "SUDDEN_DEATH"
_FEATURE_ERA_MODIFIED_SHORT = "MODIFIED_SHORT"
_FEATURE_ERA_CURRENT = "CURRENT"

# ---------------------------------------------------------------------------
# Named numeric constants
REGULATION_SECONDS = 3600
OT_QUARTER_SECONDS = 900
MAX_YARDLINE = 100


def determine_ot_era(season: int, week: int) -> str:
    """Return the display-facing OT era string for the given season and week.

    The NFL expanded to a 17-game regular season in 2021, which shifted the
    playoff week boundary:
      - Through 2020: regular season ended at week 17 → playoffs start week 18+
      - 2021+:        regular season ends  at week 18 → playoffs start week 19+
    """
    # PDX-106: use the correct playoff-week threshold per schedule era.
    if season >= 2021:
        is_playoffs = week >= 19
    else:
        is_playoffs = week >= 18

    if season <= 2009:
        return OT_ERA_SUDDEN_DEATH
    elif season <= 2011:
        return OT_ERA_MODIFIED if is_playoffs else OT_ERA_SUDDEN_DEATH
    elif season <= 2016:
        return OT_ERA_MODIFIED
    elif season <= 2021:
        return OT_ERA_MODIFIED_SHORT
    else:
        return OT_ERA_CURRENT if is_playoffs else OT_ERA_MODIFIED_SHORT


def scenario_to_features(
    down: int,
    distance: int,
    yardline_100: int,
    quarter: int,
    seconds_remaining_quarter: int,
    score_differential: int,
    is_home_possession: bool,
    era_season: int,
    era_week: int,
) -> tuple[np.ndarray, str]:
    # UI convention: 1=own endzone, 99=opp endzone.
    # nflfastR model convention: 1=opp endzone, 99=own endzone. Invert.
    corrected_yardline = MAX_YARDLINE - yardline_100

    # Display era — returned in the API response (unchanged 5-era scheme).
    ot_era = determine_ot_era(era_season, era_week)

    # Elapsed seconds since kickoff.
    if quarter <= 4:
        quarter_start = (quarter - 1) * OT_QUARTER_SECONDS
        elapsed_in_quarter = OT_QUARTER_SECONDS - seconds_remaining_quarter
        game_seconds_elapsed = quarter_start + elapsed_in_quarter
    else:
        ot_period = quarter - 4
        ot_base_time = REGULATION_SECONDS + ((ot_period - 1) * OT_QUARTER_SECONDS)
        elapsed_in_ot = OT_QUARTER_SECONDS - seconds_remaining_quarter
        game_seconds_elapsed = ot_base_time + elapsed_in_ot

    seconds_remaining = REGULATION_SECONDS - game_seconds_elapsed

    # ---------------------------------------------------------------------------
    # Feature era — 3-era scheme that matches the model's training contract.
    # This is SEPARATE from the display era above and must not be changed without
    # re-training the model.
    #   SUDDEN_DEATH  : season < 2012
    #   MODIFIED_SHORT: 2012 ≤ season ≤ 2021
    #   CURRENT       : season ≥ 2022
    if era_season >= 2022:
        feature_era = _FEATURE_ERA_CURRENT
    elif era_season >= 2012:
        feature_era = _FEATURE_ERA_MODIFIED_SHORT
    else:
        feature_era = _FEATURE_ERA_SUDDEN_DEATH

    in_ot = quarter >= 5

    # Slots 6-10: OT-era indicator variables derived from feature_era + quarter.
    is_sudden_death_active = 1.0 if (feature_era == _FEATURE_ERA_SUDDEN_DEATH and in_ot) else 0.0
    is_modified_ot_active  = 1.0 if (feature_era == _FEATURE_ERA_MODIFIED_SHORT and in_ot) else 0.0
    is_sudden_death_era    = 1.0 if feature_era == _FEATURE_ERA_SUDDEN_DEATH else 0.0
    is_modified_era        = 1.0 if feature_era == _FEATURE_ERA_MODIFIED_SHORT else 0.0
    is_current_era         = 1.0 if feature_era == _FEATURE_ERA_CURRENT else 0.0

    # ---------------------------------------------------------------------------
    # Feature vector — 11 slots, ordered to match training contract exactly.
    #   [0]  score_differential
    #   [1]  seconds_remaining
    #   [2]  yardline_100          (corrected: 100 - UI value)
    #   [3]  down
    #   [4]  ydstogo               (distance)
    #   [5]  is_home_possession
    #   [6]  is_sudden_death_active  = 1 if (era==SUDDEN_DEATH AND quarter>=5)
    #   [7]  is_modified_ot_active   = 1 if (era==MODIFIED_SHORT AND quarter>=5)
    #   [8]  is_sudden_death_era     = 1 if era==SUDDEN_DEATH (any quarter)
    #   [9]  is_modified_era         = 1 if era==MODIFIED_SHORT (any quarter)
    #   [10] is_current_era          = 1 if era==CURRENT (any quarter)
    features = np.array([[
        float(score_differential),    # [0]
        float(seconds_remaining),     # [1]
        float(corrected_yardline),    # [2]
        float(down),                  # [3]
        float(distance),              # [4]
        1.0 if is_home_possession else 0.0,  # [5]
        is_sudden_death_active,       # [6]
        is_modified_ot_active,        # [7]
        is_sudden_death_era,          # [8]
        is_modified_era,              # [9]
        is_current_era,               # [10]
    ]], dtype=np.float32)

    return features, ot_era
