FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first to avoid pulling the 2.5GB CUDA variant
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
COPY pokermon/ pokermon/
COPY server/ server/
COPY checkpoints/ checkpoints/

# Install remaining deps (torch already satisfied, so this skips it)
RUN pip install --no-cache-dir -e ".[web]"

EXPOSE ${PORT:-8000}

CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}
