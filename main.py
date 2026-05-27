from contextlib import asynccontextmanager
import pickle
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.predict import router as predict_router

MODEL_PATH = Path(__file__).parent / "ml" / "model.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"model.pkl not found at {MODEL_PATH}")
    with MODEL_PATH.open("rb") as f:
        app.state.model = pickle.load(f)
    yield
    del app.state.model


app = FastAPI(title="paradox-predict", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
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
