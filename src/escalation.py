import json
import logging
import re
from typing import Any, Optional

from litellm import completion

from src.config import ESCALATION_DISTANCE_THRESHOLD, LLM_MODEL, ROUTING_SYSTEM_PROMPT
from src.schema import EscalationDecision, RoutingDecision

logger = logging.getLogger(__name__)

# Gate 1a: Physical safety hazards. Burning/smoke/electrical/overheating signals.
SAFETY_REGEX = re.compile(
    r"\b("
    r"burning|on fire|catch(?:es|ing)? fire|"
    r"smoke|smoking|spark(?:s|ing)?|"
    r"shock(?:ed|ing)?|electrocut\w*|"
    r"overheat\w*|melt(?:s|ed|ing)?|"
    r"smell\w* like burning|"
    r"very hot|too hot|getting (?:very |really )?hot"
    r")\b",
    re.IGNORECASE,
)

# Gate 1b: explicit human-handoff demands and legal threats.
DEMAND_REGEX = re.compile(
    r"("
    r"\b(?:speak|talk)(?: to)?(?: a| the)? (?:human|person|manager|supervisor|someone|representative|agent)\b|"
    r"\breal (?:person|human)\b|"
    r"\bhuman (?:agent|representative|being)\b|"
    r"\b(?:a |the )?manager\b|\bsupervisor\b|\brepresentative\b|"
    r"\bfile a complaint\b|\blawsuit\b|\bsue you\b|\blawyer\b|\battorney\b|\blegal action\b"
    r")",
    re.IGNORECASE,
)

# Meta questions about the assistant's own capabilities should be answered
# directly instead of going through document retrieval.
CAPABILITY_REGEX = re.compile(
    r"(what(?:\s+can)?\s+you\s+(?:do|help\s+with|assist\s+with)|"
    r"what\s+do\s+you\s+do|"
    r"what\s+are\s+you\s+able\s+to\s+do|"
    r"how\s+can\s+you\s+help)",
    re.IGNORECASE,
)

# Short greetings and acknowledgements should get a friendly reply instead of
# being treated like a support issue.
GREETING_REGEX = re.compile(
    r"^(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|thanks|thank you|ok|okay|cool|great)\b",
    re.IGNORECASE,
)


def _build_handoff_summary(
    user_query: str,
    reason: str,
    evidence: Optional[str] = None,
    recent_questions: Optional[list[str]] = None,
) -> str:
    """Create a short handoff note a human agent can scan quickly."""
    summary_parts = [f"Asked: {user_query.strip()}"]
    if recent_questions:
        summary_parts.append(
            f"Relevant question(s): {' | '.join(question.strip() for question in recent_questions if question.strip())}"
        )
    if evidence:
        summary_parts.append(f"Known: {evidence.strip()}")
    summary_parts.append(f"Why escalated: {reason.strip()}")
    return " | ".join(summary_parts)


def _build_history_context(chat_history: Optional[list[dict[str, Any]]]) -> str:
    if not chat_history:
        return "No prior conversation history."

    recent_turns = []
    for turn in chat_history[-6:]:
        role = str(turn.get("role", "unknown")).strip().capitalize()
        content = str(turn.get("content", "")).strip()
        if content:
            recent_turns.append(f"{role}: {content}")

    return "\n".join(recent_turns) if recent_turns else "No prior conversation history."


def _extract_recent_user_questions(chat_history: Optional[list[dict[str, Any]]]) -> list[str]:
    if not chat_history:
        return []

    questions = [
        str(turn.get("content", "")).strip()
        for turn in chat_history
        if str(turn.get("role", "")).lower() == "user" and str(turn.get("content", "")).strip()
    ]
    return questions[-2:]


def _build_retrieval_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return "No documents were retrieved."

    lines = []
    for chunk in retrieved_chunks:
        distance = chunk.get("distance")
        distance_text = (
            f" (distance={distance:.2f})"
            if isinstance(distance, (int, float))
            else ""
        )
        lines.append(
            f"- {chunk.get('source_doc', 'unknown source')}{distance_text}: {chunk.get('excerpt', '')}"
        )

    return "\n".join(lines)


def _parse_routing_payload(raw_content: str) -> RoutingDecision:
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    parsed_payload: Optional[dict[str, Any]] = None
    try:
        parsed_payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            parsed_payload = json.loads(match.group(0))

    if parsed_payload is None:
        raise ValueError("AI router did not return valid JSON.")

    return RoutingDecision.model_validate(parsed_payload)


def decide_routing_action(
    user_query: str,
    retrieved_chunks: list[dict[str, Any]],
    draft_answer: str,
    top_distance: Optional[float] = None,
    triage_hint: Optional[str] = None,
    chat_history: Optional[list[dict[str, Any]]] = None,
) -> RoutingDecision:
    """Ask the LLM whether to respond or escalate."""
    retrieval_context = _build_retrieval_context(retrieved_chunks)
    history_context = _build_history_context(chat_history)
    distance_note = (
        f"Top semantic distance: {top_distance:.2f}."
        if top_distance is not None
        else "Top semantic distance: unavailable."
    )
    messages = [
        {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User query: {user_query.strip()}\n\n"
                f"Conversation history:\n{history_context}\n\n"
                f"Retrieved context:\n{retrieval_context}\n\n"
                f"Draft answer:\n{draft_answer.strip()}\n\n"
                f"Triage hint:\n{triage_hint or 'None'}\n\n"
                f"{distance_note}"
            ),
        },
    ]

    try:
        response = completion(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.0,
        )
        raw_content = response.choices[0].message.content or ""
        decision = _parse_routing_payload(raw_content)
    except Exception as exc:
        logger.warning(f"AI routing failed, falling back to deterministic routing: {exc}")
        should_escalate = triage_hint is not None
        if should_escalate:
            # The fallback escalates because the lexical pre-gate flagged this
            # request (safety or explicit human demand). Report that real cause
            # rather than a distance comparison, which may not actually hold.
            reason = f"Fallback routing after AI failure: escalated on pre-gate flag ({triage_hint})."
        else:
            reason = "Fallback routing after AI failure: no human action was required, so the assistant responded."
        return RoutingDecision(
            action="escalate" if should_escalate else "respond",
            reason=reason,
            handoff_summary=(
                _build_handoff_summary(
                    user_query,
                    reason,
                    evidence="The router could not produce a valid AI decision.",
                    recent_questions=_extract_recent_user_questions(chat_history),
                )
                if should_escalate
                else None
            ),
            specialist="human support specialist" if should_escalate else None,
        )

    if decision.action == "escalate" and not decision.handoff_summary:
        top_source = retrieved_chunks[0]["source_doc"] if retrieved_chunks else "no retrieved documents"
        decision.handoff_summary = _build_handoff_summary(
            user_query,
            decision.reason,
            evidence=f"Closest available context: {top_source}.",
            recent_questions=_extract_recent_user_questions(chat_history),
        )

    return decision


def check_pre_retrieval(user_query: str) -> Optional[EscalationDecision]:
    """Gate 1: Lexical scan before touching the vector database.

    Safety is checked first so a query that mentions both a hazard and a demand
    is routed as the higher-severity safety case.
    """
    safety_match = SAFETY_REGEX.search(user_query)
    if safety_match:
        reason = (
            "SAFETY_EMERGENCY: Query contains a physical hazard signal "
            f"('{safety_match.group(0).strip().lower()}'). Routed to a human immediately."
        )
        return EscalationDecision(
            should_escalate=True,
            reason=reason,
            handoff_summary=_build_handoff_summary(
                user_query,
                reason,
                evidence="Safety keyword detected in the request; the assistant should not continue.",
            ),
        )

    demand_match = DEMAND_REGEX.search(user_query)
    if demand_match:
        reason = (
            "EXPLICIT_ESCALATION: Customer explicitly requested a human or raised a "
            f"legal demand ('{demand_match.group(0).strip().lower()}')."
        )
        return EscalationDecision(
            should_escalate=True,
            reason=reason,
            handoff_summary=_build_handoff_summary(
                user_query,
                reason,
                evidence=f"Customer asked for escalation via '{demand_match.group(0).strip().lower()}'.",
            ),
        )

    capability_match = CAPABILITY_REGEX.search(user_query)
    if capability_match:
        reason = (
            "CAPABILITY_QUERY: Customer asked what the assistant can do, so "
            f"it should be answered directly ('{capability_match.group(0).strip().lower()}')."
        )
        return EscalationDecision(
            should_escalate=False,
            reason=reason,
            handoff_summary=_build_handoff_summary(
                user_query,
                reason,
                evidence="This is a capability question; no escalation needed.",
            ),
        )

    greeting_match = GREETING_REGEX.search(user_query.strip())
    if greeting_match:
        reason = (
            "GREETING_QUERY: Customer sent a greeting or acknowledgement, so "
            f"it should be answered directly ('{greeting_match.group(0).strip().lower()}')."
        )
        return EscalationDecision(
            should_escalate=False,
            reason=reason,
            handoff_summary=_build_handoff_summary(
                user_query,
                reason,
                evidence="This is a greeting; no escalation needed.",
            ),
        )

    return None


def check_post_retrieval(
    top_distance: float, threshold: float = ESCALATION_DISTANCE_THRESHOLD
) -> Optional[EscalationDecision]:
    """Gate 2: Semantic distance check after vector retrieval."""
    if top_distance > threshold:
        reason = f"OUT_OF_DOMAIN: Top document cosine distance ({top_distance:.2f}) exceeds confidence threshold ({threshold:.2f})."
        return EscalationDecision(should_escalate=True, reason=reason)

    return None