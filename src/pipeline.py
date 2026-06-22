import logging
from typing import List, Optional
from litellm import completion
from src.config import LLM_MODEL, SYSTEM_PROMPT
from src.escalation import check_post_retrieval, check_pre_retrieval
from src.ingest import build_ephemeral_store
from src.schema import AssistantResponse, Citation, EscalationDecision

# Initialize logger for monitoring and debugging
logger = logging.getLogger(__name__)


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
        if pre_decision and pre_decision.should_escalate:
            logger.warning(
                f"Gate 1 Escalation triggered: {pre_decision.reason}"
            )
            return AssistantResponse(
                user_query=user_query,
                answer="I am routing your request to a live human support representative.",
                citation=None,
                escalation=pre_decision,
            )

        # RETRIEVAL: Fetch top semantic matches from RAM store
        logger.info(f"Querying vector space for: '{user_query}'")
        retrieved_chunks = self.vector_store.query(user_query)

        if not retrieved_chunks:
            return AssistantResponse(
                user_query=user_query,
                answer="I don't know.",
                citation=None,
                escalation=EscalationDecision(
                    should_escalate=True,
                    reason="Knowledge base returned zero matching documents.",
                ),
            )

        top_hit = retrieved_chunks[0]
        top_distance = top_hit["distance"]

        # GATE 2: Post-retrieval semantic distance check
        post_decision = check_post_retrieval(top_distance)
        if post_decision and post_decision.should_escalate:
            logger.warning(
                f"Gate 2 Escalation triggered (distance={top_distance:.2f}): {post_decision.reason}"
            )
            return AssistantResponse(
                user_query=user_query,
                answer="I am routing your inquiry to a senior support specialist.",
                citation=None,
                escalation=post_decision,
            )

        # GENERATION: Assemble grounded prompt and call Groq
        context_block = "\n\n".join(
            [
                f"Document: {chunk['source_doc']}\nContent: {chunk['excerpt']}"
                for chunk in retrieved_chunks
            ]
        )

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
            response = completion(
                model=LLM_MODEL,
                messages=messages,
                temperature=0.0,  # Zero temperature for strict factual grounding
            )
            raw_answer = response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Inference API Failure: {str(e)}")
            return AssistantResponse(
                user_query=user_query,
                answer="I am currently experiencing technical difficulties reaching the knowledge base.",
                citation=None,
                escalation=EscalationDecision(
                    should_escalate=True, reason=f"UPSTREAM_API_ERROR: {str(e)}"
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
                    reason="Query unanswerable from context. Handled gracefully.",
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
                reason="Successfully generated cited, grounded response.",
            ),
        )