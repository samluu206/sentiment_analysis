FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY setup.py .
COPY src/ src/

RUN pip install -e .

# Create models directory (model will be downloaded at runtime or mounted as volume)
RUN mkdir -p models/roberta_sentiment

ENV MODEL_PATH=/app/models/roberta_sentiment
ENV PYTHONUNBUFFERED=1

EXPOSE 8000 7860

CMD ["python", "-m", "uvicorn", "src.sentiment_analyzer.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
