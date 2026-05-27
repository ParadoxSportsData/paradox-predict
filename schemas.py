from pydantic import BaseModel, Field


class ScenarioRequest(BaseModel):
    down: int = Field(..., ge=1, le=4)
    distance: int = Field(..., ge=1, le=99)
    # UI convention: 1 = own endzone, 99 = opponent endzone.
    # Inverted before model input: corrected = 100 - yardline_100
    yardline_100: int = Field(..., ge=1, le=99)
    quarter: int = Field(..., ge=1, le=5)
    seconds_remaining_quarter: int = Field(..., ge=0, le=900)
    score_differential: int = Field(..., ge=-50, le=50)
    is_home_possession: bool
    era_season: int = Field(..., ge=2010, le=2030)
    era_week: int = Field(..., ge=1, le=22)


class ScenarioPrediction(BaseModel):
    win_probability: float
    ot_era: str
    scenario: ScenarioRequest
