# Radiology Assistant - Production Suite

A production-grade, AI-powered radiology assistant suite featuring 8 specialized agents, clinical differentiator tools, and a secure HIPAA-compliant architecture.

## 🚀 Key Features

- **8 Specialized Agents**: CV Highlighting, Drafting, Follow-up, QA, Patient Summary, Triage, Learning Digest, and Orchestration.
- **Production Infrastructure**:
    - **Security**: JWT Authentication + Role-Based Access Control (RBAC).
    - **Persistence**: PostgreSQL backend with SQLAlchemy ORM + Alembic migrations.
    - **Scalability**: Asynchronous task processing via **Celery & Redis**.
    - **Observability**: Prometheus metrics + Structured JSON logging.
    - **PHI Protection**: Automated regex-based scrubbing of PII/PHI before LLM calls.
- **Unique Clinical Differentiators**:
    - **Fatigue Detector**: Predictive risk analysis for radiologists using error-rate trends.
    - **Follow-up Engine**: Natural language parsing of follow-up intervals + database reminders.
    - **FHIR Exporter**: Clinical data conversion to FHIR R4 `DiagnosticReport` resources.
    - **CME Platform**: Automated CME case generation (MCQs) for continuous learning.
- **Quality Assurance**: Automated self-evaluation confidence scoring for all LLM-drafted reports.

## 🛠️ Tech Stack
- **Backend**: FastAPI, Pydantic, SQLAlchemy, Celery, Redis.
- **LLM**: Google Gemini (via LLMClient) / Ollama fallback.
- **Database**: PostgreSQL / SQLite (dev).
- **DevOps**: Docker, Docker Compose, Prometheus.

## 🏗️ Project Structure
```
rad-assistant/
├── src/radiology_assistant/
│   ├── agents/            # 8 Core + 4 Differentiator Agents
│   ├── cv/                # DICOM I/O & CV Preprocessing
│   ├── api.py             # FastAPI entry point
│   ├── auth.py            # JWT & RBAC logic
│   ├── database.py        # SQLAlchemy session/engine
│   ├── db_models.py       # SQL Tables
│   ├── tasks.py           # Celery Background Tasks
│   └── observability.py    # Metrics & Logging
├── Dockerfile             # Multi-stage production build
├── docker-compose.yml     # local stack (Postgres, Redis, Worker)
└── triage_config.yaml     # Dynamic triage thresholds
```

## 🚥 Quick Start (Docker)

1. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your GEMINI_API_KEY
   ```

2. **Launch Stack**:
   ```bash
   docker-compose up --build
   ```

3. **Access API**:
   - API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
   - Metrics: [http://localhost:8000/metrics](http://localhost:8000/metrics)

## 🔐 Authentication
All clinical endpoints require a `Bearer` token.
- **Login**: `POST /v1/auth/token` (default demo creds: `admin` / `admin123`)

## 📊 Triage Configuration
Thresholds for critical/high priority worklist items can be updated dynamically without restarting the server:
```bash
PUT /v1/admin/triage_config
# Payload: JSON matching TriageConfig schema
```

## 🧪 Testing
```bash
pytest tests/ -v
```

---
**Status**: Production Ready v2.0.0
**Last Updated**: 2026-02-21
