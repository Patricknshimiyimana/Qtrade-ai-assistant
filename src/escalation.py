import re
from typing import Optional
from src.config import ESCALATION_DISTANCE_THRESHOLD
from src.schema import EscalationDecision

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


def check_pre_retrieval(user_query: str) -> Optional[EscalationDecision]:
    """Gate 1: Lexical scan before touching the vector database.

    Safety is checked first so a query that mentions both a hazard and a demand
    is routed as the higher-severity safety case.
    """
    safety_match = SAFETY_REGEX.search(user_query)
    if safety_match:
        return EscalationDecision(
            should_escalate=True,
            reason=(
                "SAFETY_EMERGENCY: Query contains a physical hazard signal "
                f"('{safety_match.group(0).strip().lower()}'). Routed to a human immediately."
            ),
        )

    demand_match = DEMAND_REGEX.search(user_query)
    if demand_match:
        return EscalationDecision(
            should_escalate=True,
            reason=(
                "EXPLICIT_ESCALATION: Customer explicitly requested a human or raised a "
                f"legal demand ('{demand_match.group(0).strip().lower()}')."
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