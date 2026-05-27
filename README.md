# paradox-predict

Win-probability prediction service for the ParadoxSportsData platform. Wraps a trained XGBoost model (binary classifier) and exposes a single HTTP endpoint for real-time scenario simulation.

**Port:** 8002  
**Python:** 3.14  
**Part of:** clock-gate (8080) · paradox-stats (8001) · paradox-predict (8002) · paradox-ui (5173)

---

## System dependency

XGBoost on macOS requires OpenMP:

```bash
brew install libomp
```

---

## Install

```bash
python3.14 -m venv venv
venv/bin/pip install -e .
```

---

## Run

```bash
venv/bin/uvicorn main:app --port 8002
```

The server loads `ml/model.pkl` at startup and fails fast if the file is missing.

---

## Endpoints

### `GET /health`

```json
{"status": "ok", "service": "paradox-predict", "model_loaded": true}
```

### `POST /predict/scenario`

**Request body:**

| Field | Type | Range | Notes |
|-------|------|-------|-------|
| `down` | int | 1–4 | |
| `distance` | int | 1–99 | Yards to go |
| `yardline_100` | int | 1–99 | **UI convention:** 1=own endzone, 99=opp endzone |
| `quarter` | int | 1–5 | 5=OT |
| `seconds_remaining_quarter` | int | 0–900 | |
| `score_differential` | int | -50–50 | Possession team score minus opponent |
| `is_home_possession` | bool | | |
| `era_season` | int | 2010–2030 | For OT rule era detection |
| `era_week` | int | 1–22 | ≥19 = playoffs |

**Response:**

```json
{
  "win_probability": 0.712,
  "ot_era": "MODIFIED_SHORT",
  "scenario": { ... }
}
```

`win_probability` is always expressed from the **possession team's perspective** (not always home team). The model internally computes home-team WP; when the away team has possession, `1 - home_wp` is returned.

**yardline convention:** The UI slider uses 1=own endzone, 99=opponent endzone. This is inverted before model input (`corrected = 100 - yardline_100`) to match the nflfastR training convention.

---

## Example curl

```bash
# 4th & Goal from the 1, down 4, 10s left, Q4
curl -s -X POST http://localhost:8002/predict/scenario \
  -H "Content-Type: application/json" \
  -d '{
    "down": 4,
    "distance": 1,
    "yardline_100": 1,
    "quarter": 4,
    "seconds_remaining_quarter": 10,
    "score_differential": -4,
    "is_home_possession": true,
    "era_season": 2024,
    "era_week": 1
  }'
# → {"win_probability": ~0.15, "ot_era": "MODIFIED_SHORT", "scenario": {...}}
```

---

## OT Era Rules

| Era | Seasons | Playoffs |
|-----|---------|---------|
| `SUDDEN_DEATH` | ≤2009, 2010–2011 reg | — |
| `MODIFIED` | 2010–2011 playoffs, 2012–2016 | Both teams get possession |
| `MODIFIED_SHORT` | 2017–2021, 2022+ reg | 10-min OT |
| `GUARANTEED` | 2022+ playoffs | Both teams guaranteed possession |

---

## Model

- **Algorithm:** XGBoost binary classifier
- **Source:** Trained in `paradox-platform/src/ml/`; serialized as `ml/model.pkl`
- **Target:** Home-team win probability
- **Features (11, in training order):**
  1. `score_differential`
  2. `seconds_remaining` (computed from quarter + clock)
  3. `yardline_100` (inverted to nflfastR convention)
  4. `down`
  5. `ydstogo` (distance)
  6. `is_home_possession`
  7. `is_sudden_death_era`
  8. `is_modified_era`
  9. `is_modified_short_era`
  10. `is_guaranteed_era`
  11. `is_sudden_death_active` (true in Q6+, or if era is SUDDEN_DEATH)
