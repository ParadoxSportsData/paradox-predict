from contextlib import asynccontextmanager
import os
import pickle
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.predict import router as predict_router

MODEL_PATH = Path(__file__).parent / "ml" / "model.pkl"

# Number of input features expected by the trained model.
# Must match the feature vector produced by scenario_to_features().
EXPECTED_N_FEATURES = 11


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"model.pkl not found at {MODEL_PATH}")
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)

    # PDX-118: Validate that the loaded model's feature count matches our contract.
    actual = getattr(model, "n_features_in_", None)
    if actual is not None and actual != EXPECTED_N_FEATURES:
        raise RuntimeError(
            f"Model feature count mismatch: expected {EXPECTED_N_FEATURES}, got {actual}. "
            "Re-train the model or fix scenario_to_features()."
        )

    app.state.model = model
    yield
    del app.state.model


app = FastAPI(title="paradox-predict", version="0.1.0", lifespan=lifespan)

# PDX-118: CORS origins from environment variable with fallback to local dev defaults.
cors_origins_env = os.getenv("CORS_ORIGINS", "")
cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] or [
    "http://localhost:5173",
    "http://localhost:4173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(predict_router)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "paradox-predict",
        "model_loaded": hasattr(app.state, "model"),
    }
