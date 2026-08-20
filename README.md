# StratOps-Intel

**Production-Grade Competitive Intelligence Platform** — Multi-modal signal ingestion, AI-powered entity extraction, temporal knowledge graphs, and automated executive briefings. Built with BentoML + vLLM model mesh, LangGraph agentic workflows, and hardened multi-tenant security.

## Architecture

```mermaid
graph LR
    subgraph "Client Layer"
        N[Next.js Dashboard]
    end

    subgraph "API Gateway"
        G[FastAPI Gateway]
    end

    subgraph "Inference Layer"
        B1[BentoML + vLLM Mesh]
        E[Embedding Service]
        N1[Narrative Service]
        F[Fallback Service]
    end

    subgraph "Data Layer"
        PS[PostgreSQL + pgvector + RLS]
        NG[Neo4j Temporal Graph]
        MS[MinIO (S3-compat)]
        RS[Redis Streams]
    end

    subgraph "Workers"
        IW[Ingestion Worker]
        GW[Graph Writer]
        EW[Embedding Worker]
    end
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API Gateway** | FastAPI, Pydantic v2, JWT + API key auth, RBAC |
| **AI/ML** | BentoML + vLLM (Qwen2.5-7B-AWQ, Llama-3.1-8B), LangGraph, Instructor |
| **Embeddings** | bge-large-en-v1.5 (1024-dim), adaptive batching |
| **Database** | PostgreSQL 16 + pgvector, Row-Level Security, declarative partitioning |
| **Graph DB** | Neo4j 5 Community, temporal relationships, Redis-buffered batch writes |
| **Messaging** | Redis Streams (core), Celery (exports only) |
| **Object Storage** | MinIO (S3-compatible) for raw payloads |
| **Frontend** | Next.js 14, TypeScript, TailwindCSS, shadcn/ui |
| **Observability** | OpenTelemetry, Prometheus, structlog, Grafana |

## Core Features (Implemented)

### Weeks 1-3: Foundation + Ingestion + Intelligence Core

- **Multi-tenant PostgreSQL** with RLS — every connection enforces `app.current_tenant`
- **Redis Streams** infrastructure — producer/consumer base, tenant-aware key builder
- **FastAPI Gateway** — JWT + API key dual auth, rate limiting, request logging
- **SourceAdapter Protocol** — plugin registry with `@register_adapter` decorator
- **WebMonitorAdapter** — Playwright + SimHash near-deduplication
- **SECFilingAdapter** — EDGAR RSS + XBRL parsing with rate limiting
- **BentoML + vLLM Extraction Service** — Qwen2.5-7B-AWQ, guided JSON decoding, adaptive batching
- **Pointer-Only LangGraph State** — S3/MinIO URIs only, checkpoint target < 5KB
- **Entity Extractor Node** — downloads from MinIO, calls vLLM, writes entities back as pointers
- **Micro-Batching Graph Writer** — Redis-buffered, `UNWIND ... MERGE`, deduplication by `(entity_id, rel_type)`
- **Embedding Service** — bge-large-en-v1.5, 1024-dim, normalized, batchable
- **Neo4j Temporal Schema** — `valid_from`/`valid_to` on all relationships, tenant isolation

## Quick Start

### Prerequisites

- Docker + Docker Compose
- Python 3.11+
- Node.js 20+ (for frontend)

### 1. Start Infrastructure

```bash
docker compose -f docker/docker-compose.infra.yml up -d
```

Starts: PostgreSQL 16 (pgvector), Neo4j 5, Redis 7, MinIO.

### 2. Run Migrations

```bash
cd backend
alembic upgrade head
```

### 3. Start API Gateway

```bash
uvicorn api.gateway:app --reload
```

### 4. Health Check

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```

### Development

```bash
cd backend
pip install -e ".[dev]"
pytest tests/unit/ -v
pytest tests/integration/ -v
```

### BentoML Services

```bash
cd bentoml
pip install -e .
bentoml serve services.extraction:ExtractionService
```

## Project Structure

```text
stratops-intel/
├── backend/
│   ├── api/              # FastAPI gateway, auth, middleware
│   ├── db/               # Postgres RLS, Alembic, Neo4j client
│   ├── streams/          # Redis Streams producer/consumer
│   ├── ingestion/        # Source adapters (web, SEC, jobs, patents, earnings)
│   ├── intelligence/     # LangGraph agents (extractor, correlation, trend, narrative)
│   ├── workers/          # Ingestion, intelligence, graph writer
│   ├── alerts/           # Alert rules, router
│   ├── briefings/        # Composer, delta updates
│   └── tests/
├── bentoml/
│   ├── services/         # extraction, embedding, summarization, narrative, fallback
│   └── composed/         # Multi-model pipelines
├── frontend/             # Next.js 14 App Router
├── docker/               # Docker Compose definitions
├── docs/                 # ADRs, benchmarks, deployment
└── benchmarks/           # Performance benchmarks
```

## Architecture Decisions

See `docs/adr/` for full Architecture Decision Records:

- **BentoML + vLLM over raw transformers** — PagedAttention, continuous batching, guided decoding
- **Pointer-only LangGraph state** — S3/MinIO URIs, checkpoints < 5KB vs ~1.5MB raw
- **PostgreSQL RLS + partitioning** — tenant isolation at DB level, bounded HNSW per partition
- **Redis-buffered Neo4j writes** — `UNWIND ... MERGE` micro-batching, no direct stream→graph writes
- **Redis Streams over Celery** — core messaging via Streams, Celery only for long-running exports

## License

MIT