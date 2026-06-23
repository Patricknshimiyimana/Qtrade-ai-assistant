import logging
from typing import List, Optional
from litellm import completion
from src.config import LLM_MODEL, RETRIEVAL_RETRY_COUNT, SYSTEM_PROMPT
from src.escalation import check_pre_retrieval, decide_routing_action
from src.ingest import build_ephemeral_store
from src.schema import AssistantResponse, Citation, EscalationDecision

# Initialize logger for monitoring and debugging
logger = logging.getLogger(__name__)

CAPABILITY_RESPONSE = (
    "I can help with QTrade returns and refunds, shipping questions, "
    "SmartHub setup and troubleshooting, and warranty support. "
    "If you want, ask me a specific question about one of those topics."
)

GREETING_RESPONSE = (
    "Hello. I can help with QTrade returns and refunds, shipping questions, "
    "SmartHub setup and troubleshooting, and warranty support. "
    "Ask me a question whenever you're ready."
)


class SupportPipeline:
    """Orchestrates the QTrade customer support transaction lifecycle."""

    def __init__(self):
        logger.info(
            "Initializing ephemeral vector store and ingesting Appendix A..."
        )
        self.vector_store = build_ephemeral_store()

    def process_query(
        self, user_query: str, chat_history: Optional[List[dict]] = None
    ) -> AssistantResponse:
        """Executes the full RAG loop: Gate 1 -> Retrieval -> Gate 2 -> Generation."""
        history = chat_history or []

        # GATE 1: Pre-retrieval lexical hazard & demand scan
        pre_decision = check_pre_retrieval(user_query)

        if pre_decision and not pre_decision.should_escalate:
            logger.info(f"Direct response detected: {pre_decision.reason}")

            if pre_decision.reason and pre_decision.reason.startswith("GREETING_QUERY"):
                direct_response = GREETING_RESPONSE
                direct_reason = "Answered directly as a greeting query."
            else:
                direct_response = CAPABILITY_RESPONSE
                direct_reason = "Answered directly as a capability query."

            return AssistantResponse(
                user_query=user_query,
                answer=direct_response,
                citation=None,
                escalation=EscalationDecision(
                    should_escalate=False,
                    reason=direct_reason,
                ),
            )

        # RETRIEVAL: Fetch top semantic matches from RAM store, retrying when the
        # match quality is weak or the store returns nothing.
        retrieved_chunks = []
        top_hit = None
        top_distance = None

        for attempt in range(1, RETRIEVAL_RETRY_COUNT + 1):
            logger.info(
                f"Querying vector space for: '{user_query}' (attempt {attempt}/{RETRIEVAL_RETRY_COUNT})"
            )
            retrieved_chunks = self.vector_store.query(user_query)

            if not retrieved_chunks:
                logger.warning(
                    f"Retrieval attempt {attempt} returned zero matching documents."
                )
                continue

            top_hit = retrieved_chunks[0]
            top_distance = top_hit["distance"]
            break


        # GENERATION: Assemble grounded prompt and call Groq
        context_block = "\n\n".join(
            [
                f"Document: {chunk['source_doc']}\nContent: {chunk['excerpt']}"
                for chunk in retrieved_chunks
            ]
        ) if retrieved_chunks else ""

        # Grounding decision: We do not replay prior turns into the grounded prompt, because earlier assistant
        # answers become "context" the model can lean on, which dilutes the strict
        # "answer only from the retrieved docs / else say I don't know" rule across a
        # multi-turn session. History is still kept by the CLI for display, and could
        # later be used for query rewriting (resolving "it"/"that") rather than as
        # answer context. `chat_history` is intentionally unused here for now.
        _ = history
        sys_prompt = SYSTEM_PROMPT.format(context=context_block)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_query},
        ]

        try:
            logger.info(f"Invoking LiteLLM inference endpoint ({LLM_MODEL})...")
            if retrieved_chunks:
                response = completion(
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=0.0,  # Zero temperature for strict factual grounding
                )
                raw_answer = response.choices[0].message.content.strip()
            else:
                raw_answer = "I don't know."

        except Exception as e:
            logger.error(f"Inference API Failure: {str(e)}")
            return AssistantResponse(
                user_query=user_query,
                answer="I am currently experiencing technical difficulties reaching the knowledge base.",
                citation=None,
                escalation=EscalationDecision(
                    should_escalate=True,
                    reason=f"UPSTREAM_API_ERROR: {str(e)}",
                    handoff_summary=(
                        f"Asked: {user_query.strip()} | Known: The system could not reach the language model. | "
                        f"Why escalated: UPSTREAM_API_ERROR: {str(e)}"
                    ),
                ),
            )

        route_decision = decide_routing_action(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
            draft_answer=raw_answer,
            top_distance=top_distance,
            triage_hint=pre_decision.reason if pre_decision and pre_decision.should_escalate else None,
            chat_history=history,
        )

        if route_decision.action == "escalate":
            specialist = route_decision.specialist or "human support specialist"
            logger.warning(
                f"AI router escalated query to {specialist}: {route_decision.reason}"
            )
            return AssistantResponse(
                user_query=user_query,
                answer=f"I am routing your inquiry to a {specialist}.",
                citation=None,
                escalation=EscalationDecision(
                    should_escalate=True,
                    reason=route_decision.reason,
                    handoff_summary=route_decision.handoff_summary,
                ),
            )

        # PARSING: Grounding Enforcement & Citation Mapping
        if "i don't know" in raw_answer.lower():
            return AssistantResponse(
                user_query=user_query,
                answer="I don't know.",
                citation=None,
                escalation=EscalationDecision(
                    should_escalate=False,
                    reason=route_decision.reason,
                ),
            )

        # Map the answer back to the exact supporting source document
        citation_obj = None
        for chunk in retrieved_chunks:
            doc_title = chunk["source_doc"]
            if doc_title in raw_answer or f"[{doc_title}]" in raw_answer:
                citation_obj = Citation(
                    source_doc=doc_title, excerpt=chunk["excerpt"]
                )
                break

        # Fallback mapping if LLM omitted bracket but semantic match was near-perfect
        if not citation_obj and top_distance < 0.35:
            citation_obj = Citation(
                source_doc=top_hit["source_doc"], excerpt=top_hit["excerpt"]
            )

        return AssistantResponse(
            user_query=user_query,
            answer=raw_answer,
            citation=citation_obj,
            escalation=EscalationDecision(
                should_escalate=False,
                reason=route_decision.reason,
            ),
        )