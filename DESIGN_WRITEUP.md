# Design Write-Up — QTrade AI Support Assistant

The system uses a word-based hand-off check (safety + human requests) plus a
distance check after search, over an in-memory RAG pipeline. These notes answer
the four design questions.

## 1. A different hand-off policy

I built a fixed, per-message check: each question is matched against safety and
human-request phrases before search, and against a distance cutoff after search.
It hands off the moment either one fires.

A different policy would use confidence bands with a "clarify" step: answer when
the match is strong, ask one follow-up question when it's borderline, and hand
off only when it's weak or the model says it's unsure. It could also hand off
based on the whole chat — repeated weak answers, the same question asked again,
or signs of frustration.

This catches more cases and gives fewer dead-end "I don't know"s, so more
customers get helped without a human. But it costs more LLM calls and latency,
self-rated confidence needs tuning, and chat-level rules risk handing off too
late, after the customer is already upset. The fixed rule I shipped is simpler to
predict, test, and audit, which makes it the right default for v1.

## 2. Scaling to 10x and 100x

At 10x (about 40 docs), the "one doc = one chunk" choice breaks first: bigger
docs hurt search accuracy and waste tokens, so I'd add real chunking with
overlap. Re-embedding everything at startup also gets slow, so I'd save the index
to disk and only re-embed changed docs.

At 100x (thousands of docs, many users at once), brute-force search and a single
process break. I'd move to a real vector database with an ANN index, add
retrieve-then-rerank and keyword+vector search for accuracy, cache embeddings and
common answers to control cost, and split the app into a stateless API plus a
separate indexing job that runs when docs change.

## 3. Measuring quality and catching regressions

I'd build a labelled test set (question → which doc, hand off or not, "I don't
know" or not) and score retrieval (is the right doc in the top results?),
grounding (is every claim backed by a doc?), citation accuracy, and hand-off
accuracy — tracking missed safety cases closely, since those are the costly ones.

To catch regressions early, I'd run this set in CI on every change to a prompt,
model, threshold, or doc, and fail the build if a score drops. In production I'd
watch hand-off rate, "I don't know" rate, and thumbs-down rate, and alert on
sudden changes — these usually show a problem before customers feel it.

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
