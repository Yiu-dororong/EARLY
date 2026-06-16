FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install -r requirements.txt

# Copy shared resources required by the API
COPY utils/ utils/
COPY agents/ agents/
COPY models/ models/
COPY core/ core/
COPY training/ training/

COPY api/ api/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]