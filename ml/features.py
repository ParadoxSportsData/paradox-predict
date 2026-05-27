from __future__ import annotations

import numpy as np


def determine_ot_era(season: int, week: int) -> str:
    is_playoffs = week >= 19
    if season <= 2009:
        return "SUDDEN_DEATH"
    elif season <= 2011:
        return "MODIFIED" if is_playoffs else "SUDDEN_DEATH"
    elif season <= 2016:
        return "MODIFIED"
    elif season <= 2021:
        return "MODIFIED_SHORT"
    else:
        return "GUARANTEED" if is_playoffs else "MODIFIED_SHORT"


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
    corrected_yardline = 100 - yardline_100

    ot_era = determine_ot_era(era_season, era_week)

    if quarter <= 4:
        quarter_start = (quarter - 1) * 900
        elapsed_in_quarter = 900 - seconds_remaining_quarter
        game_seconds_elapsed = quarter_start + elapsed_in_quarter
    else:
        ot_period = quarter - 4
        ot_base_time = 3600 + ((ot_period - 1) * 900)
        elapsed_in_ot = 900 - seconds_remaining_quarter
        game_seconds_elapsed = ot_base_time + elapsed_in_ot

    seconds_remaining = 3600 - game_seconds_elapsed

    is_sudden_death = 1.0 if ot_era == "SUDDEN_DEATH" else 0.0
    is_modified = 1.0 if ot_era == "MODIFIED" else 0.0
    is_modified_short = 1.0 if ot_era == "MODIFIED_SHORT" else 0.0
    is_guaranteed = 1.0 if ot_era == "GUARANTEED" else 0.0

    is_deep_ot = quarter >= 6
    is_sudden_death_active = 1.0 if (ot_era == "SUDDEN_DEATH" or is_deep_ot) else 0.0

    features = np.array([[
        float(score_differential),
        float(seconds_remaining),
        float(corrected_yardline),
        float(down),
        float(distance),
        1.0 if is_home_possession else 0.0,
        is_sudden_death,
        is_modified,
        is_modified_short,
        is_guaranteed,
        is_sudden_death_active,
    ]], dtype=np.float32)

    return features, ot_era
