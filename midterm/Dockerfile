FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --only-binary :all: -r requirements.txt && \
    find /usr/local -type d -name '__pycache__' -exec rm -r {} + 2>/dev/null || true && \
    find /usr/local -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true

COPY . .

EXPOSE $PORT
CMD ["sh", "-c", "gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app"]