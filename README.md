# QTrade AI Support Assistant

A small RAG assistant for QTrade's help docs. It answers questions from four
help documents, cites the doc it used, says **"I don't know"** when the answer
isn't in the docs, and hands off to a human when it shouldn't answer.

It runs for free: local embeddings plus a free-tier or local LLM.

## What it does

- Reads the four help docs and embeds them locally with `sentence-transformers`
  (`all-MiniLM-L6-v2`), stored in an in-memory ChromaDB collection.
- Finds the top-k closest docs for a question.
- Answers with a citation (`[Cited: Doc N: …]`) using a strict, temperature-0
  prompt, or replies "I don't know." when the docs don't cover it.
- Sends the question to a human using a two-gate trigger (below).
- Runs as a command-line app.

## How it works

```
question
  │
  ├─▶ Gate 1  Word check before search
  │           safety words ("burning", "smoke", "overheating", …)
  │           human requests ("a manager", "speak to a person", "lawsuit")
  │           → hand off now, no LLM call
  │
  ├─▶ Search  embed question → top-k from the vector store
  │
  ├─▶ Gate 2  Distance check
  │           top distance > 0.62 → not in our docs → hand off
  │
  └─▶ Answer  grounded, temperature-0 LLM call
              cites the doc, or says "I don't know."
```

Files:

| Path | Job |
|------|-----|
| `src/ingest.py` | Read the 4 docs, tag each with its title, index it |
| `src/vector_store.py` | In-memory ChromaDB wrapper (cosine distance) |
| `src/escalation.py` | Gate 1 (words) and Gate 2 (distance) |
| `src/pipeline.py` | Runs Gate 1 → search → Gate 2 → answer |
| `src/schema.py` | Pydantic response models |
| `src/cli.py` | Terminal UI |
| `src/config.py` | Models, top-k, threshold, prompt |

## How to run

Needs Python 3.10+.

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set up the LLM
cp .env.example .env   # then edit .env
```

- **Groq free tier (default):** get a free key at
  <https://console.groq.com/keys> and set `GROQ_API_KEY` in `.env`.
- **Fully local, no API:** install [Ollama](https://ollama.com), run
  `ollama pull llama3.2`, and set `LLM_MODEL=ollama/llama3.2` in `.env`.

```bash
# 3. Run
python main.py
```

Ask things like *"Can I return an opened item?"* or *"How do I reset my
SmartHub?"*. Type `exit` to quit.

## Key decisions

**One doc = one chunk.** Each doc is only 3–4 sentences, so splitting it would
break apart facts that belong together. At this size the whole doc is the right
chunk. This is the first thing I'd change as the docs grow (see the write-up).

**In-memory store.** Four docs embed in well under a second at startup, so a
database on disk would add complexity for no speed gain. `EphemeralClient()`
keeps the repo clean.

**Two-gate hand-off.** The brief asks for one rule; I use a word check before
search (safety + human requests) as the main rule, plus a distance check after
search to catch off-topic questions the words miss. The word check uses phrases,
not single words, because an earlier version escalated on common words like "now"
and "hot" and wrongly bounced normal questions.

**Grounding in three places:** a strict temperature-0 prompt, a fixed "I don't
know." fallback, and a step that links the answer back to a retrieved doc's
title. (That link matches the title; it doesn't prove the claim — see next
steps.)

**Stateless answers.** Past turns are not fed back into the prompt, so earlier
answers can't sneak in as context and weaken the "only use the docs" rule.

### How the 6 sample queries route

| Query | Result |
|-------|--------|
| "I opened the box, can I still return it, and is there a fee?" | Answer (Doc 1) |
| "How do I reset my SmartHub?" | Answer (Doc 3) |
| "My order hasn't shipped in 4 days, where is it?" | Answer (Doc 2) |
| "My SmartHub is getting very hot and smells like burning." | Hand off — safety |
| "…I want a refund and a manager NOW." | Hand off — human request |
| "Do you offer bulk discounts for commercial installs?" | Off-topic → "I don't know" |

## What I'd do next

- **Hand-off summary:** one short LLM call summarizing the question, what's
  known, and why it escalated, for the human agent.
- **Check the citation:** confirm the answer is actually backed by the cited doc,
  not just that the title appears.
- **Evaluation harness:** score answer quality across the sample queries
  (grounding, citation accuracy, hand-off correctness), run in CI.
- **Follow-up questions:** use chat history to resolve "is *it* refundable?"
  without feeding it into the grounding context.

See [`DESIGN_WRITEUP.md`](./DESIGN_WRITEUP.md) for the design questions:
other hand-off policies, scaling to 10×/100×, measuring quality, and
deployment.
