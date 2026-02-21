# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim as runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "radiology_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
