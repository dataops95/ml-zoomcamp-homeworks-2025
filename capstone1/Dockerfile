FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY models/ models/
COPY templates/ templates/
COPY predict.py .
COPY serve.py .

EXPOSE 9696

CMD ["python", "serve.py"]
