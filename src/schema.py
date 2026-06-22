from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Represents the grounded source used to construct an answer."""
    source_doc: str = Field(
        ..., description="The exact document identifier, e.g., 'Doc 1: Returns & Refunds'"
    )
    excerpt: str = Field(
        ..., description="The raw chunk text pulled from the vector store"
    )


class EscalationDecision(BaseModel):
    """The explicit routing state of the transaction."""
    should_escalate: bool = Field(
        ..., description="True if the assistant must refuse to answer and hand off"
    )
    reason: Optional[str] = Field(
        None, description="Concrete justification, e.g., 'Topic distance 0.71 exceeds threshold 0.62'"
    )


class AssistantResponse(BaseModel):
    """The final, unified return object passed to the CLI layer."""
    user_query: str
    answer: str = Field(
        ..., description="The natural language output, or verbatim 'I don't know.'"
    )
    citation: Optional[Citation] = None
    escalation: EscalationDecision