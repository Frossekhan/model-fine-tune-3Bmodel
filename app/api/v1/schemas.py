from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    name: str
    description: str
    inputs: Dict[str, str]


class Message(BaseModel):
    role: str = Field(..., description="user or assistant")
    content: str


class ToolCall(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


class InferenceRequest(BaseModel):
    session_id: str
    prompt: str
    history: Optional[List[Message]] = None
    tools: Optional[List[ToolDefinition]] = None
    use_rag: bool = True
    stream: Optional[bool] = True


class InferenceResponse(BaseModel):
    session_id: str
    response: str
    metadata: Optional[Dict[str, Any]] = None


class ToolInvocation(BaseModel):
    tool_name: str
    args: Dict[str, Any]
    reason: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify")


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    model: str

