# StratOps Intel

**Competitive Intelligence Platform** — A production-grade platform for real-time competitive intelligence gathering, analysis, and automated briefing generation.

## Architecture Overview

```
stratops-intel/
├── backend/          # FastAPI gateway, DB, streams, intelligence engines
├── bentoml/          # BentoML inference services (vLLM backend)
├── frontend/         # Next.js web application
├── docker/           # Docker Compose infrastructure definitions
├── docs/             # ADRs, benchmarks, deployment guides
└── .github/          # CI/CD workflows
```

### Core Stack

- **API Gateway**: FastAPI + vLLM via BentoML
- **Database**: PostgreSQL 16 with pgvector + Row-Level Security (RLS)
- **Graph Database**: Neo4j (5-community) with Redis-buffered batch writes
- **Messaging**: Redis Streams (core), Celery (long-running exports only)
- **AI/ML**: LangGraph agentic workflows, LangChain, Instructor for structured extraction
- **Object Storage**: MinIO (S3-compatible) for raw payloads
- **Observability**: OpenTelemetry, Prometheus, structlog

### Multi-Tenant Security

- PostgreSQL RLS enforced on every connection via `app.current_tenant` session variable
- Tenant isolation at the database level — no application-level filtering
- API key + JWT authentication with role-based access control (RBAC)

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Node.js 20+ (for frontend)

### Infrastructure

Start the core infrastructure services:

```bash
docker compose -f docker/docker-compose.infra.yml up -d
```

This starts:
- PostgreSQL 16 with pgvector (port 5432)
- Neo4j 5 (ports 7474, 7687)
- Redis 7 (port 6379)
- MinIO with console (ports 9000, 9001)

### Development

Install backend dependencies:

```bash
cd backend
pip install -e ".[dev]"
```

Run the API server:

```bash
uvicorn api.main:app --reload
```

## License

MIT