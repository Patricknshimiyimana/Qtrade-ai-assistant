# Design Write-Up — QTrade AI Support Assistant

The system uses a fast word check followed by an LLM router, over an in-memory
RAG pipeline. The word check flags safety words and explicit human requests, and
answers greetings directly. The router then looks at the query, retrieved
context, draft answer, and history to decide respond vs. escalate, and writes a
short handoff summary when it escalates. These notes answer the four design
questions.

## 1. A different escalation policy

The escalation policy implemented is LLM-led: after a lexical pre-check, an LLM router judges
each message in context and hands off when a human is actually needed. It catches
more real cases than fixed rules and produces a readable handoff summary.

A different policy would be deterministic-first: pre-defined rules escalate safety and
explicit human requests _before_ any model call, and the LLM is only consulted
for the ambiguous middle. This is more predictable, cheaper, and fail-safe. A
safety hand-off never depends on a model. But it catches fewer subtle cases and
needs the rules maintained by hand.

The trade-off is recall vs. predictability. The LLM router helps more customers
without a human but costs an extra call per query and can, in principle, decline
to escalate. The deterministic-first policy is safer and faster but less flexible.
The right answer is a mix: deterministic rules for safety and clear demands, and
the router for everything genuinely uncertain, which is the direction I'd take
this next.

## 2. Scaling to 10x and 100x

At 10x (about 40 docs), the "one doc = one chunk" choice breaks first: bigger
docs hurt search accuracy and waste tokens, so I'd add real chunking with
overlap. Re-embedding everything at startup also gets slow, so I'd save the index
to disk and only re-embed changed docs.

At 100x (thousands of docs, many users at once), brute-force search and a single
process break. I'd move to a real vector database with an ANN index, add
retrieve-then-rerank and keyword+vector search for accuracy, cache embeddings and
common answers to control cost, and split the app into a stateless API plus a
separate indexing job that runs when docs change. LLM cost is the other limit:
the router adds a second model call per query (which already triggers free-tier
rate limits), so at scale I'd cache and consolidate it toward a single call.

## 3. Measuring quality and catching regressions

I'd build a labelled test set (question → which doc, hand off or not, "I don't
know" or not) and score retrieval (is the right doc in the top results?),
grounding (is every claim backed by a doc?), citation accuracy, and hand-off
accuracy, tracking missed safety cases closely, since those are the costly ones.

To catch regressions early, I'd run this set in CI on every change to a prompt,
model, threshold, or doc, and fail the build if a score drops. In production I'd
watch hand-off rate, "I don't know" rate, and thumbs-down rate, and alert on
sudden changes, since these usually show a problem before customers feel it.

## 4. Deployment and monitoring

I'd package the API in a Docker container, keep secrets out of the image, and run
it behind an autoscaler with a health check. Indexing runs as a separate job, so
serving pods load a prebuilt index instead of embedding at startup.

For each request I'd log one JSON line: the question, the docs found, the
decision and why, latency, and token cost (no personal data). I'd alert on high
error rate or latency, hand-off or "I don't know" rates moving out of their
normal range, cost over budget, and — most important — a dedicated alert if
safety hand-offs ever drop to zero, since a missed safety case is the worst
failure.
