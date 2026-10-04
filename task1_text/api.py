from typing import Annotated, Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

MODEL_NAME = "rogue-security/prompt-injection-jailbreak-sentinel-v2"

app = FastAPI(
    title="Prompt Injection Detector",
    description="Детекция prompt-injection / jailbreak атак в запросах к LLM",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, description="текст запроса для проверки")


class PredictResponse(BaseModel):
    label: str
    score: float
    is_attack: bool


_classifier = None


def get_classifier() -> Any:
    global _classifier
    if _classifier is None:
        from task1_text.prompt_injection_detector import load_pipe

        _classifier = load_pipe()
    return _classifier


@app.get("/")
def root() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest, pipe: Annotated[Any, Depends(get_classifier)]
) -> PredictResponse:
    result = pipe(request.text)[0]
    label = result["label"]
    return PredictResponse(
        label=label,
        score=result["score"],
        is_attack=label.lower() != "benign",
    )
