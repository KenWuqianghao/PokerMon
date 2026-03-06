FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY pokermon/ pokermon/
COPY server/ server/
COPY checkpoints/ checkpoints/

RUN pip install --no-cache-dir -e ".[web]"

EXPOSE ${PORT:-8000}

CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}
