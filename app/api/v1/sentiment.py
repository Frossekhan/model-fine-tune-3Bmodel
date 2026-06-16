from fastapi import APIRouter
from pydantic import BaseModel
from app.services.sentiment_service import SentimentService
from app.api.v1.schemas import SentimentResponse

router = APIRouter()

class SentimentRequest(BaseModel):
    text: str

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    return SentimentService.analyze(request.text)