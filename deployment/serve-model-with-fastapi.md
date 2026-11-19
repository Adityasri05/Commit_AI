# Serve a Machine Learning Model with FastAPI

A trained model becomes useful to other applications when exposed through a stable interface. FastAPI can provide a small HTTP service that validates input and returns predictions.

## Minimal example

Assume a scikit-learn pipeline has already been trained and saved as `model.joblib`. Only load artifacts from trusted sources; pickle-based formats can execute code during deserialization.

```python
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(Path("model.joblib"))
    yield

app = FastAPI(lifespan=lifespan)

class PredictionInput(BaseModel):
    age: float = Field(ge=0, le=120)
    income: float = Field(ge=0)

@app.post("/predict")
def predict(payload: PredictionInput):
    prediction = app.state.model.predict([[payload.age, payload.income]])[0]
    return {"prediction": str(prediction)}
```

Run locally with `uvicorn app:app --reload` and inspect the interactive API docs at `/docs`.

## Production considerations

- Pin dependencies and record the model version.
- Validate input ranges and schema; reject malformed requests.
- Add authentication and authorization where needed.
- Avoid exposing stack traces or sensitive training data.
- Add health/readiness checks, structured logs, timeouts, and rate limits.
- Monitor latency, errors, input drift, and model performance.
- Load-test before choosing worker counts and hardware.

This is a teaching skeleton, not a production security template. Input fields and feature order must match the trained pipeline.

## Exercise

Extend the endpoint with a request ID, model version in the response, and a test using FastAPI's TestClient. Verify that negative income receives a validation error.

**Takeaway:** A prediction API needs versioning, validation, security, and monitoring—not just a call to `predict()`.