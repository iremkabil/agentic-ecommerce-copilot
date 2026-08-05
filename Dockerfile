FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached unless deps change.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[retrieval,dev]"

# Copy the rest of the project.
COPY . .

# api -> 8000, streamlit dashboard -> 8501
EXPOSE 8000 8501

# Overridden per-service in docker-compose.yml; this is the sensible default.
CMD ["uvicorn", "copilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
