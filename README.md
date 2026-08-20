# StratOps-Intel

**Production-Grade Competitive Intelligence Platform** — Multi-modal signal ingestion, AI-powered entity extraction, temporal knowledge graphs, and automated executive briefings. Built with BentoML + vLLM model mesh, LangGraph agentic workflows, and hardened multi-tenant security.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-300%2B%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![BentoML](https://img.shields.io/badge/BentoML-1.4%2B-orange)
![vLLM](https://img.shields.io/badge/vLLM-0.6%2B-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-teal)
![Neo4j](https://img.shields.io/badge/Neo4j-5%2B-yellow)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%2B-blue)

---

## What Makes This Different

| # | Architectural Decision | Why It Matters |
|---|------------------------|----------------|
| **1** | **BentoML + vLLM Model Mesh** | PagedAttention, continuous batching, guided JSON decoding. No raw transformers — production-grade serving with 10× throughput vs naive implementations. |
| **2** | **Pointer-Only LangGraph State** | S3/MinIO URIs only in state. Checkpoints < 5KB vs ~1.5MB raw payloads. Enables horizontal scaling of LangGraph workers. |
| **3** | **PostgreSQL RLS + Partitioning** | Tenant isolation at DB level. Declarative partitioning by `tenant_id`. Bounded HNSW per partition — no cross-tenant leakage. |
| **4** | **Micro-Batching Graph Writer** | Redis-buffered, `UNWIND ... MERGE`, deduplication by `(entity_id, rel_type)`. Zero direct Neo4j writes from stream consumers. |
| **5** | **Event-Driven (Redis Streams)** | Core messaging via Streams. Consumer groups, exactly-once semantics. Celery only for long-running exports (PDF/PPTX). |

---

## Architecture

```
┌─────────────────┐
│  Next.js Dash   │
└────────┬────────┘
         │ HTTPS
         ▼
┌─────────────────┐     ┌──────────────────┐
│ FastAPI Gateway │────▶│  Redis Streams   │◀──┐
│  JWT + API Key  │     │  Event Bus       │   │
└─────────────────┘     └────────┬─────────┘   │
         │                       │             │
         │         ┌─────────────┼─────────────┤
         │       ▼              ▼             ▼
         │ ┌──────────┐ ┌──────────────┐ ┌────────────┐
         │ │Ingestion │ │Intelligence  │ │  BentoML   │
         │ │ Workers  │ │   Core       │ │  Mesh      │
         │ │(Web/SEC/ │ │(LangGraph)   │ │(vLLM)      │
         │ │Earnings) │ │              │ │            │
         │ └──────────┘ └──────┬───────┘ └─────┬──────┘
         │                    │                 │
         │         ┌──────────┴──────────┐     │
         │         │  BentoML Services   │     │
         │         │  ┌───────────────┐  │     │
         │         │  │ extraction-svc│  │     │  (vLLM)
         │         │  │ embedding-svc │  │     │
         │         │  │ summarization │  │     │
         │         │  │ narrative-svc │  │     │
         │         │  │ fallback-svc  │  │     │
         │         │  └───────────────┘  │     │
         │         └─────────┬───────────┘     │
         │                   │                 │
    ┌────┴─────┐    ┌────────┴───────┐ ┌──────┴───────┐
    │PostgreSQL│    │     Neo4j      │ │    MinIO     │
    │+ pgvector│    │  Temporal Graph│ │  (S3-compat) │
    │  RLS +   │    │ valid_from/    │ │  Pointers    │
    │Partition │    │ valid_to       │ │  Only        │
    └──────────┘    └────────────────┘ └──────────────┘
```

---

## Component Specifications

| Service | Model | GPU | Purpose | Batch Size |
|---------|-------|-----|---------|------------|
| **extraction-svc** | Qwen2.5-7B-AWQ | 1× GPU | Structured entity extraction | 32 |
| **embedding-svc** | bge-large-en-v1.5 | 0.5× GPU | 1024-dim semantic embeddings | 128 |
| **summarization-svc** | Llama-3.1-8B | 1× GPU | Executive/technical summaries | 16 |
| **narrative-svc** | Qwen2.5-14B-AWQ | 1× GPU | Strategic narrative synthesis | 1 |
| **fallback-svc** | Qwen2.5-3B-GGUF | 0.5× GPU | Lightweight fallback processing | 64 |

---

## Week-by-Week Implementation Progress

| Week | Focus | Key Deliverables | Tests |
|------|-------|-----------------|-------|
| **W1** | Multi-Tenant Foundation | PostgreSQL RLS, Redis Streams, FastAPI Gateway, JWT+API key auth | 102 unit, 8 integration |
| **W2** | Ingestion Engine | SourceAdapter protocol, WebMonitor (Playwright+SimHash), SEC EDGAR, BentoML scaffold | 139 unit, 9 integration |
| **W3** | Intelligence Core | vLLM guided decoding, pointer-only LangGraph, micro-batching graph writer, embedding service | 180+ unit, 10 integration |
| **W4** | Correlation & Trend | Temporal Cypher queries, earnings pipeline, Z-score/STL, anomaly detection (Isolation Forest), narrative synthesis | 250+ unit, 12 integration |
| **W5** | Briefing & Alerts | Briefing composer with versioning, delta regeneration, alert rule engine, Slack/email/webhook router, fallback service | 300+ unit, 15 integration |

**Total: ~300+ unit tests, 15+ integration tests — all passing.**

---

## Data Flow

1. **Ingest**: WebMonitor/SECFiling/Earnings adapters fetch raw signals from public sources
2. **Normalize**: Raw content → MinIO pointer. SimHash fingerprint dedup. RLS-scoped PostgreSQL insert
3. **Extract**: LangGraph `EntityExtractorNode` calls vLLM extraction service. Entities written to MinIO as JSON pointers
4. **Correlate**: Temporal Cypher queries find pricing/talent/co-mention/patent patterns via `CorrelationEngineNode`
6. **Analyze**: Z-score/STL trends via `TrendAnalyzerNode`, Isolation Forest anomalies via `AnomalyDetectorNode`
7. **Synthesize**: `NarrativeService` generates strategic narratives from multi-section intelligence
8. **Compose**: `BriefingComposerNode` assembles versioned executive briefings with MinIO storage
9. **Alert**: `AlertRuleEngine` evaluates thresholds/anomalies/trends, routes to Slack/email/webhook
10. **Delta**: `BriefingDeltaGenerator` computes incremental updates (append/replace/full-regeneration) for living briefings

---

## Multi-Tenant Security

- **Row-Level Security**: `SET app.current_tenant = :uuid` on every connection — enforced at DB level
- **Declarative Partitioning**: All tables partitioned by `tenant_id` — physical isolation per tenant
- **API Key + JWT Dual Auth**: JWT for users, API keys for services — RBAC scopes
- **Rate Limiting**: Redis sliding window per tenant (configurable per tier)
- **Audit Logging**: Structured logging on all mutations with tenant context

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/aragit/stratops-intel.git
cd stratops-intel

# 2. Start infrastructure (Postgres, Neo4j, Redis, MinIO)
docker compose -f docker/docker-compose.infra.yml up -d

# 3. Run migrations
cd backend && alembic upgrade head

# 4. Start API Gateway
uvicorn api.gateway:app --reload

# 5. Health check
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```

### BentoML Services (separate processes)

```bash
cd bentoml && pip install -e .
bentoml serve services.extraction:ExtractionService
bentoml serve services.embedding:EmbeddingService
bentoml serve services.summarization:SummarizationService
bentoml serve services.narrative:NarrativeService
bentoml serve services.fallback:FallbackService
```

---

## API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/auth/login` | Public | Email + password → JWT |
| `POST` | `/auth/api-keys` | JWT | Create API key |
| `GET` | `/health` | Public | Liveness probe |
| `GET` | `/health/ready` | Public | Readiness (DB, Redis, Neo4j) |
| `GET` | `/health/tenant` | API key | Tenant-scoped health |
| *(future)* | `/api/v1/signals` | API key | List signals |
| *(future)* | `/api/v1/briefings` | API key | List briefings |
| *(future)* | `/api/v1/briefings/current` | API key | Get current briefing |
| *(future)* | `/api/v1/alerts` | API key | List alerts |

---

## Testing

```bash
# Unit tests (fast, mocked)
cd backend && pytest tests/unit/ -v --cov=.

# Integration tests (testcontainers: Postgres, Redis, Neo4j, MinIO)
pytest tests/integration/ -v

# Security tests (RLS, tenant isolation)
pytest tests/security/ -v

# Load tests (Week 8)
cd benchmarks && python load_test.py
```

---

## Deployment Topology (docker-compose.prod.yml)

```yaml
services:
  postgres:       pgvector/pgvector:pg16      # RLS + partitioning
  neo4j:          neo4j:5-community           # Temporal graph
  redis:          redis:7-alpine              # Streams + rate limiting
  minio:          minio/minio:latest          # S3-compatible object storage
  api-gateway:    stratops-api:latest         # FastAPI + JWT + rate limiting
  extraction-svc: stratops-extraction:latest  # BentoML + vLLM (Qwen2.5-7B-AWQ)
  embedding-svc:  stratops-embedding:latest   # bge-large-en-v1.5
  summarization:  stratops-summarization:latest # Llama-3.1-8B
  narrative-svc:  stratops-narrative:latest   # Qwen2.5-14B-AWQ
  fallback-svc:   stratops-fallback:latest    # Qwen2.5-3B-GGUF
  workers:
    - ingestion-worker      # Source adapters
    - graph-writer          # Micro-batching Neo4j
    - intelligence-worker   # LangGraph pipeline
    - graph-writer          # Delta updates
    - alert-router          # Multi-channel routing
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **API Gateway** | FastAPI, Pydantic v2, JWT + API key, RBAC |
| **AI/ML** | BentoML + vLLM (Qwen2.5-7B, Llama-3.1-8B, Qwen2.5-14B, bge-large), LangGraph, Instructor |
| **Embeddings** | bge-large-en-v1.5 (1024-dim), adaptive batching |
| **Database** | PostgreSQL 16 + pgvector, RLS, declarative partitioning |
| **Graph DB** | Neo4j 5 Community, temporal relationships (`valid_from`/`valid_to`) |
| **Messaging** | Redis Streams (core), Celery (exports only) |
| **Object Storage** | MinIO (S3-compatible) |
| **Frontend** | Next.js 14, TypeScript, TailwindCSS, shadcn/ui |
| **Observability** | OpenTelemetry, Prometheus, structlog, Grafana |

---

## Project Structure

```
stratops-intel/
├── backend/
│   ├── api/              # FastAPI gateway, auth, middleware
│   ├── db/               # Postgres RLS, Alembic, Neo4j client
│   ├── streams/          # Redis Streams producer/consumer
│   ├── ingestion/        # Source adapters (web, SEC, earnings, jobs, patents)
│   ├── intelligence/     # LangGraph agents (extractor, correlation, trend, anomaly, composer, delta)
│   ├── workers/          # Ingestion, intelligence, graph writer
│   ├── alerts/           # Rule engine, router (Slack/email/webhook)
│   ├── briefings/        # Repository, delta generator
│   └── tests/
├── bentoml/
│   ├── services/         # extraction, embedding, summarization, narrative, fallback
│   └── tests/
├── frontend/             # Next.js 14 App Router
├── docker/               # Docker Compose definitions
├── docs/                 # ADRs, benchmarks, deployment
├── benchmarks/           # Performance benchmarks
└── docker/               # Docker Compose definitions
```

---

## Portfolio Narrative

> **"For regulated domains, I build deterministic state machines with symbolic kernels. For intelligence synthesis, I build event-driven platforms with BentoML + vLLM model meshes and temporal knowledge graphs. I choose the architecture to fit the problem's non-functional requirements — not the other way around."**

---

## License

MIT — See [LICENSE](LICENSE) for details.