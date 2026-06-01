FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    jq \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /data/events /data/clips
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# Pre-load POS transactions so the API has data on first start
RUN DATABASE_URL="sqlite+aiosqlite:////data/store_intelligence.db" \
    python scripts/load_pos.py --csv data/pos_transactions.csv 2>/dev/null || true
RUN DATABASE_URL="sqlite+aiosqlite:////data/store_intelligence.db" \
    python scripts/load_pos.py --csv data/Brigade_Bangalore_10_April_26.csv --raw 2>/dev/null || true

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
