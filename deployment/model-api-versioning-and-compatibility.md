# Model API Versioning and Compatibility

A deployed model is consumed through an interface: request fields, response fields, authentication, error behavior, and latency expectations. Changing model weights is not the same as changing that interface.

Separating API versioning from model versioning helps teams update models without unexpectedly breaking applications.

## Two different version concepts

- **API version:** The contract clients use to send requests and interpret responses.
- **Model version:** The specific trained artifact and associated preprocessing or postprocessing configuration.

For example, API v1 might serve model 12 today and model 13 after a rollout without changing the request schema. A breaking schema change may require API v2 even if the same model artifact is used.

## Contract design

Document:
- Required and optional request fields.
- Types, units, allowed ranges, and missing-value behavior.
- Response schema and confidence semantics.
- Error codes and retry guidance.
- Authentication and authorization.
- Model and feature versions where useful.
- Latency and availability expectations.

A probability field should state which class it refers to and whether it is calibrated. A timestamp should state its timezone and format. Ambiguous contracts create subtle downstream bugs.

## Backward-compatible and breaking changes

Usually backward-compatible:
- Adding an optional response field when clients tolerate unknown fields.
- Adding an optional request field with a safe default.
- Improving internal model quality without changing output semantics.

Potentially breaking:
- Renaming or removing fields.
- Changing a field's type or units.
- Reordering positional outputs.
- Changing class-label meanings.
- Changing null handling or error behavior.
- Returning a different prediction target under the same field name.

Even additive changes can break strict clients that reject unknown response fields. Test actual consumers rather than relying only on theoretical compatibility.

## Example with FastAPI and explicit schemas

~~~python
from pydantic import BaseModel, Field
from fastapi import FastAPI

app = FastAPI()

class PredictRequestV1(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)

class PredictResponseV1(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    model_version: str

@app.post("/v1/predict", response_model=PredictResponseV1)
def predict_v1(request: PredictRequestV1):
    label, score = run_current_model(request.text)
    return PredictResponseV1(
        label=label,
        score=float(score),
        model_version="classifier-2026-11",
    )
~~~

The route and schema are illustrative. In a real service, load the model at application startup, validate input size and content, enforce authentication, and avoid hard-coding a version that can drift from the actual artifact.

## Safe migration process

1. Write down the current contract and identify all consumers.
2. Classify the proposed change as compatible or breaking.
3. Add contract tests for valid, invalid, and boundary requests.
4. If needed, publish a new versioned endpoint while keeping the old one available.
5. Migrate clients gradually and observe usage.
6. Set a deprecation date and communicate it clearly.
7. Remove the old version only after usage and ownership checks are complete.

## Compatibility testing

Useful tests include:
- Schema validation and serialization tests.
- Golden request/response fixtures for stable semantics.
- Consumer-driven contract tests.
- Load and latency tests.
- Tests for missing, extra, malformed, and out-of-range fields.
- Compatibility tests for preprocessing and label mappings.

Golden outputs should allow justified numerical tolerances. Floating-point inference may vary slightly across hardware and library versions.

## Common pitfalls

- Reusing an API version for a changed prediction meaning.
- Returning a model version string that does not match the loaded artifact.
- Breaking clients by adding fields to strict schemas.
- Removing old endpoints before checking real traffic.
- Forgetting feature transformations and label mappings during model swaps.
- Exposing internal stack traces or sensitive details in API errors.

## Mini-project

Build /v1/predict and a deliberately incompatible /v2/predict endpoint. Write tests showing which changes are backward-compatible. Route both versions to the same model, then switch one to a candidate model without changing the API contract.

## References

- Google API Design Guide: https://cloud.google.com/apis/design
- FastAPI request and response models: https://fastapi.tiangolo.com/tutorial/response-model/
- Microsoft REST API Guidelines: https://github.com/microsoft/api-guidelines
