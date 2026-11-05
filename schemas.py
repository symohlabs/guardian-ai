from pydantic import BaseModel


class TextAnalysisRequest(BaseModel):
    user_id: str
    text: str
    context: str | None = None


class TextAnalysisResponse(BaseModel):
    status: str
    risk_level: str
    risk_score: float
    threat_type: str
    recommended_action: str