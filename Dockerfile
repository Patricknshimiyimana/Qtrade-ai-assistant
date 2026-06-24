# syntax=docker/dockerfile:1
# QTrade AI Support Assistant image
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# chroma-hnswlib (a C++ extension pulled in by chromadb) has no prebuilt wheel
# for Python 3.12, so it compiles from source — which needs a C++ compiler.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer is cached unless requirements change.
COPY requirements.txt .

# Install the CPU-only build of torch BEFORE the rest, so sentence-transformers
# reuses it instead of pulling the much larger CUDA build (~2GB). The pip cache
# mount keeps downloads across rebuilds without bloating the final image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Pre-download the embedding model at build time so the container starts fast
# and does not need network access for embeddings at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the application code.
COPY . .

# Bind to all interfaces inside the container; PORT is overridable by the host/PaaS.
ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

# GROQ_API_KEY must be supplied at runtime.
# Defaults to the CLI, matching `python main.py`. For the HTTP API, override the
# command with `python main.py --api` (and publish port 8000).
CMD ["python", "main.py"]
