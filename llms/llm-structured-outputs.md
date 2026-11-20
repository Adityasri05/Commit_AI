# Reliable Structured Outputs from LLMs

Many applications need machine-readable output such as JSON matching a schema. Asking an LLM to "return valid JSON" is not, by itself, a reliability strategy.

## Recommended approach
1. Define a schema with required fields, types, ranges, and allowed values.
2. Use provider-supported constrained or structured output when available.
3. Parse the response with a real JSON parser.
4. Validate it against the schema.
5. Handle refusals, truncation, timeouts, and validation failures explicitly.
6. Keep a safe fallback path rather than trusting malformed output.

## Example with Pydantic
```python
from pydantic import BaseModel, Field, ValidationError

class StudyPlan(BaseModel):
    topic: str
    hours: float = Field(gt=0, le=12)
    tasks: list[str]

raw = '{"topic":"linear algebra","hours":2,"tasks":["vectors","matrices"]}'
try:
    plan = StudyPlan.model_validate_json(raw)
except ValidationError as exc:
    print("Invalid model output:", exc)
```

## Design principles
- Treat model output as untrusted input.
- Validate semantic constraints, not only syntax.
- Avoid executing generated code or SQL directly.
- Log validation failures without storing unnecessary sensitive data.
- Test empty, oversized, adversarial, and partially generated outputs.

## Exercise
Create a schema for a quiz question and test valid output, missing fields, wrong types, and invalid answer options.

**Key point:** Structured generation helps; application-side validation remains mandatory.