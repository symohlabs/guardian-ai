import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ml.nlp.predict import predict_message
from schemas import TextAnalysisRequest, TextAnalysisResponse



PROJECT_ROOT = Path(__file__).resolve().parent#tells the code where the file is located
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")

if not GUARDIAN_API_KEY:
    raise RuntimeError(
        "GUARDIAN_API_KEY is missing. Check your .env file."
    )

app = FastAPI(
    title="GUARDIAN AI API",
    description="AI-powered text risk analysis for child online safety.",
    version="1.0.0"
)



api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(
    api_key: str = Security(api_key_header)
):
    if not api_key or api_key != GUARDIAN_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    return api_key

limiter = Limiter(
    key_func=get_remote_address
)

app.state.limiter = limiter

app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler
)


"""
@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "GUARDIAN AI",
        "message": "Text analysis API is running"
    }

"""

@app.post(
    "/api/v1/analyze/text",
    response_model=TextAnalysisResponse,
    dependencies=[Depends(verify_api_key)]
)
@limiter.limit("30/minute")
def analyze_text(
    request: Request,
    body: TextAnalysisRequest
):
    prediction = predict_message(body.text)
    risk_mapping = {
        "safe": {
            "risk_level": "low",
            "risk_score": 0.1,
            "recommended_action": "allow"
        },
        "cyberbullying": {
            "risk_level": "medium",
            "risk_score": 0.6,
            "recommended_action": "notify_guardian"
        },
        "grooming": {
            "risk_level": "high",
            "risk_score": 0.9,
            "recommended_action": "guardian_review"
        },
        "threat": {
            "risk_level": "high",
            "risk_score": 0.9,
            "recommended_action": "guardian_review"
        },
        "sexual_solicitation": {
            "risk_level": "high",
            "risk_score": 0.95,
            "recommended_action": "block_and_notify_guardian"
        }
    }

    result = risk_mapping.get(
        prediction,
        {
            "risk_level": "unknown",
            "risk_score": 0.0,
            "recommended_action": "manual_review"
        }
    )


    return TextAnalysisResponse(
        status="success",
        risk_level=result["risk_level"],
        risk_score=result["risk_score"],
        threat_type=prediction,
        recommended_action=result["recommended_action"]
    )
