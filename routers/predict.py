from fastapi import APIRouter, Request
from schemas import ScenarioRequest, ScenarioPrediction
from ml.features import scenario_to_features

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("/scenario", response_model=ScenarioPrediction)
async def predict_scenario(scenario: ScenarioRequest, request: Request) -> ScenarioPrediction:
    model = request.app.state.model

    features, ot_era = scenario_to_features(
        down=scenario.down,
        distance=scenario.distance,
        yardline_100=scenario.yardline_100,
        quarter=scenario.quarter,
        seconds_remaining_quarter=scenario.seconds_remaining_quarter,
        score_differential=scenario.score_differential,
        is_home_possession=scenario.is_home_possession,
        era_season=scenario.era_season,
        era_week=scenario.era_week,
    )

    win_probability = float(model.predict_proba(features)[0][1])

    # Model always returns home-team WP. When away team has possession,
    # the caller's perspective (posteam WP) is the complement.
    if not scenario.is_home_possession:
        win_probability = 1.0 - win_probability

    return ScenarioPrediction(
        win_probability=win_probability,
        ot_era=ot_era,
        scenario=scenario,
    )
