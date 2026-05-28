import pytest
import numpy as np
from ml.features import scenario_to_features


def make_scenario(**overrides):
    """Return a valid scenario dict with sensible defaults."""
    defaults = {
        "down": 3,
        "distance": 7,
        "yardline_100": 65,
        "quarter": 2,
        "seconds_remaining_quarter": 300,
        "score_differential": 0,
        "is_home_possession": True,
        "era_season": 2024,
        "era_week": 8,
    }
    return {**defaults, **overrides}


def features(**overrides):
    """Call scenario_to_features() and return the numpy array."""
    s = make_scenario(**overrides)
    arr, _ = scenario_to_features(**s)
    return arr


# ---------------------------------------------------------------------------
# 1. Output shape
def test_output_shape():
    arr, ot_era = scenario_to_features(**make_scenario())
    assert arr.shape == (1, 11), f"Expected (1, 11), got {arr.shape}"
    assert isinstance(ot_era, str)


# ---------------------------------------------------------------------------
# 2. CURRENT era (season 2024, regulation quarter)
def test_current_era_slot10():
    arr = features(era_season=2024, quarter=2)
    assert arr[0, 10] == 1.0, "Slot 10 (is_current_era) must be 1 for 2024"
    assert arr[0, 6] == 0.0, "Slot 6 (is_sudden_death_active) must be 0 for CURRENT era"
    assert arr[0, 7] == 0.0, "Slot 7 (is_modified_ot_active) must be 0 for CURRENT era"
    assert arr[0, 8] == 0.0, "Slot 8 (is_sudden_death_era) must be 0 for CURRENT era"
    assert arr[0, 9] == 0.0, "Slot 9 (is_modified_era) must be 0 for CURRENT era"


# ---------------------------------------------------------------------------
# 3. MODIFIED_SHORT era (2012–2021), regulation
def test_modified_short_era_slot9():
    arr = features(era_season=2015, quarter=2)
    assert arr[0, 9] == 1.0, "Slot 9 (is_modified_era) must be 1 for 2015"
    assert arr[0, 10] == 0.0, "Slot 10 (is_current_era) must be 0 for 2015"
    assert arr[0, 7] == 0.0, "Slot 7 (is_modified_ot_active) must be 0 in regulation"


# ---------------------------------------------------------------------------
# 4. SUDDEN_DEATH era (season < 2012), regulation
def test_sudden_death_era_slot8():
    arr = features(era_season=2010, quarter=2)
    assert arr[0, 8] == 1.0, "Slot 8 (is_sudden_death_era) must be 1 for 2010"
    assert arr[0, 9] == 0.0, "Slot 9 (is_modified_era) must be 0 for 2010"
    assert arr[0, 10] == 0.0, "Slot 10 (is_current_era) must be 0 for 2010"


# ---------------------------------------------------------------------------
# 5. CURRENT era in OT — no OT-active flags
def test_current_era_no_ot_active():
    arr = features(era_season=2024, quarter=5)
    assert arr[0, 6] == 0.0, "Slot 6 (is_sudden_death_active) must be 0 for CURRENT era in OT"
    assert arr[0, 7] == 0.0, "Slot 7 (is_modified_ot_active) must be 0 for CURRENT era in OT"
    assert arr[0, 10] == 1.0, "Slot 10 (is_current_era) must still be 1 in OT"


# ---------------------------------------------------------------------------
# 6. MODIFIED_SHORT era in OT → is_modified_ot_active + is_modified_era both on
def test_modified_short_ot_active():
    arr = features(era_season=2015, quarter=5)
    assert arr[0, 7] == 1.0, "Slot 7 (is_modified_ot_active) must be 1 for 2015 Q5"
    assert arr[0, 9] == 1.0, "Slot 9 (is_modified_era) must be 1 for 2015"
    assert arr[0, 6] == 0.0, "Slot 6 (is_sudden_death_active) must be 0 for 2015"
    assert arr[0, 8] == 0.0, "Slot 8 (is_sudden_death_era) must be 0 for 2015"


# ---------------------------------------------------------------------------
# 7. SUDDEN_DEATH era in OT → is_sudden_death_active + is_sudden_death_era both on
def test_sudden_death_ot_active():
    arr = features(era_season=2010, quarter=5)
    assert arr[0, 6] == 1.0, "Slot 6 (is_sudden_death_active) must be 1 for 2010 Q5"
    assert arr[0, 8] == 1.0, "Slot 8 (is_sudden_death_era) must be 1 for 2010"
    assert arr[0, 7] == 0.0, "Slot 7 (is_modified_ot_active) must be 0 for 2010"
    assert arr[0, 9] == 0.0, "Slot 9 (is_modified_era) must be 0 for 2010"


# ---------------------------------------------------------------------------
# 8. No OT-active flags in regulation for any era
@pytest.mark.parametrize("era_season", [2010, 2015, 2024])
def test_no_ot_active_in_regulation(era_season):
    arr = features(era_season=era_season, quarter=4)
    assert arr[0, 6] == 0.0, f"Slot 6 must be 0 in Q4 for season {era_season}"
    assert arr[0, 7] == 0.0, f"Slot 7 must be 0 in Q4 for season {era_season}"


# ---------------------------------------------------------------------------
# 9. Yardline inversion: UI value 20 → corrected = 80
def test_yardline_inversion():
    arr = features(yardline_100=20)
    assert arr[0, 2] == 80.0, f"Slot 2 must be 80.0, got {arr[0, 2]}"


# ---------------------------------------------------------------------------
# 10. Slot 0 carries score_differential directly
def test_slot0_score_differential():
    arr = features(score_differential=7)
    assert arr[0, 0] == 7.0, f"Slot 0 must be 7.0, got {arr[0, 0]}"


# ---------------------------------------------------------------------------
# 11. Slot 5 carries is_home_possession
def test_slot5_home_possession():
    arr_home = features(is_home_possession=True)
    arr_away = features(is_home_possession=False)
    assert arr_home[0, 5] == 1.0, "Slot 5 must be 1.0 when home has possession"
    assert arr_away[0, 5] == 0.0, "Slot 5 must be 0.0 when away has possession"


# ---------------------------------------------------------------------------
# 12. Era boundary: season 2012 → MODIFIED_SHORT (slot 9)
def test_era_boundary_2012():
    arr = features(era_season=2012, quarter=2)
    assert arr[0, 9] == 1.0, "Slot 9 must be 1.0 for era_season=2012 (MODIFIED_SHORT)"
    assert arr[0, 10] == 0.0
    assert arr[0, 8] == 0.0


# ---------------------------------------------------------------------------
# 13. Era boundary: season 2022 → CURRENT (slot 10)
def test_era_boundary_2022():
    arr = features(era_season=2022, quarter=2)
    assert arr[0, 10] == 1.0, "Slot 10 must be 1.0 for era_season=2022 (CURRENT)"
    assert arr[0, 9] == 0.0
    assert arr[0, 8] == 0.0
