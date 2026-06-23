from typing import Literal, Optional
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
        None,
        description="Concrete justification, e.g., 'SAFETY_EMERGENCY: Query contains a physical hazard signal'",
    )
    handoff_summary: Optional[str] = Field(
        None,
        description="Short human-readable summary for the agent handoff queue",
    )


class RoutingDecision(BaseModel):
    """AI-authored routing choice for the next step in the pipeline."""

    action: Literal["respond", "escalate"] = Field(
        ..., description="Whether the assistant should answer or hand off"
    )
    reason: str = Field(
        ..., description="AI-generated rationale for the chosen action"
    )
    handoff_summary: Optional[str] = Field(
        None,
        description="Short human-readable summary for the agent handoff queue",
    )
    specialist: Optional[str] = Field(
        None,
        description="Optional specialist label when escalation is needed",
    )


class AssistantResponse(BaseModel):
    """The final, unified return object passed to the CLI layer."""
    user_query: str
    answer: str = Field(
        ..., description="The natural language output, or verbatim 'I don't know.'"
    )
    citation: Optional[Citation] = None
    escalation: EscalationDecision


class SessionCreateResponse(BaseModel):
    """Response returned when a new support session is created."""

    session_id: str
    created_at: str
    message_count: int = 0


class SessionMessageRequest(BaseModel):
    """Request payload for sending a message into a support session."""

    message: str = Field(..., min_length=1)


class SessionMessageResponse(BaseModel):
    """Response returned after processing a session message."""

    session_id: str
    response: AssistantResponse
    history: list[dict[str, str]]