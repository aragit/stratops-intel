# StratOps-Intel Intelligence Core (v0.6.0)

[![Version: v0.6.0](https://img.shields.io/badge/version-v0.6.0-blue)](https://github.com/aragit/stratops-intel)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/download/releases/python-3.11/)
[![Tests: 327 Passed](https://img.shields.io/badge/tests-327%20passed-brightgreen)](https://github.com/aragit/stratops-intel/actions)
[![Protocol: FastMCP](https://img.shields.io/badge/protocol-FastMCP-orange)](https://modelcontextprotocol.io)
[![License: Enterprise](https://img.shields.io/badge/license-Enterprise-green)](https://opensource.org/licenses/Enterprise)

---

## Executive Overview

**StratOps-Intel Intelligence Core (v0.6.0)** is a high-performance, neuro-symbolic intelligence engine and graph orchestration platform built for high-throughput SEC/financial analysis, competitive tracking, and multi-agent governance. The release introduces three production-grade subsystems:

1. **Tenant-Isolated Semantic Execution Cache** — Redis-backed embedding similarity cache ahead of LiteLLM/vLLM.
2. **Pgvector Hybrid Retrieval Engine** — Sparse (BM25 `ts_rank_cd`) + Dense (pgvector cosine) search with Reciprocal Rank Fusion (RRF).
3. **FastMCP Model Context Protocol Server** — JSON-schema-registered tool interface (`stratops-intel-mcp`) exposing knowledge graph, entity trends, and hybrid retrieval to external swarms.

The architecture is grounded in **deterministic governance**: PostgreSQL Row-Level Security (RLS), declarative list partitioning, OPA/Rego constraint hooks, and pointer-only state preservation in LangGraph workflows. All code paths are covered by a hardened test suite (327 passed, 5 skipped) with zero tolerance for order-dependent test pollution.

---

## System Architecture & Data Flow

```mermaid
flowchart LR
    subgraph Layer_1__Gateway_Guardrails
        G1[Async API Gateway<br/>FastAPI + JWT + API Key]:::gateway
        G2[Redis Lua Sliding-Window Rate Limiter]:::redis
        G3[Tenant RLS Context Middleware]:::tenant
        G4[Semantic Execution Cache]:::cache
    end

    subgraph Layer_2__Agent_Orchestration_State
        L2a[LangGraph DAG State Machine<br/>pointer-only ~2KB checkpoints]:::langgraph
        L2b[LiteLLM Execution Bus]:::litellm
        L2c[Alert Stream Worker + DLQ]:::worker
    end

    subgraph Layer_3__Hybrid_Retrieval_Persistence
        L3a[PostgreSQL<br/>pgvector + BM25 tsvector + RLS]:::postgres
        L3b[Neo4j Knowledge Graph]:::neo4j
        L3c[MinIO / S3 Storage]:::minio
    end

    subgraph Layer_4__Tool_Mesh
        M1[FastMCP Server<br/>stratops-intel-mcp]:::mcp
    end

    %% Connections
    G1 -->|HTTPS + Auth| G2
    G2 -->|Rate-Scoped Keys| G3
    G3 -->|Tenant-Partitioned| G4
    G4 -->|Cache-Hit/Miss| L2b
    L2b -->|Signals + Traces| L3a
    L2b -->|Events + URIs| L3b
    L2b -->|Trend Results| L3c
    L2b -->|Tool Calls| M1
    M1 -->|JSON-Schema Queries| L3a
    M1 -->|KG Traversal| L3b
    M1 -->|Hybrid Search| L3a
```

:::info classlegend
:::gateway
FastAPI gateway with JWT/API-key auth, rate limiting, tenant context.
:::
:::redis
Redis Lua sliding-window limiter (`_SLIDING_WINDOW_LUA`).
:::
:::tenant
Declarative RLS + list partitioning by `tenant_id`.
:::
:::cache
`cache:semantic:{tenant_id}:{hash}` keys, cosine threshold 0.92.
:::
:::langgraph
LangGraph DAG with pointer-only state (<5KB checkpoints).
:::
:::litellm
LiteLLM execution bus for LLM inference.
:::
:::worker
Alert stream worker + DLQ retry with exponential backoff.
:::
:::postgres
PostgreSQL 16+ with pgvector extension, RLS, declarative partitioning.
:::
:::neo4j
Neo4j 5.24+ knowledge graph with tenant-scoped traversals.
:::
:::minio
MinIO/S3 for pointer-only artifact storage (PDF/PPTX exports).
:::
:::mcp
FastMCP server (`stratops-intel-mcp`) with 3 registered tools.
:::
:::

---

## Core Technical Modules & Innovations

### A. Pgvector Hybrid Search Engine (`backend/db/vector_store.py`)

The hybrid retrieval pipeline combines two ranking sources via Reciprocal Rank Fusion (RRF):

**Sparse Ranking** — PostgreSQL `ts_rank_cd` against a `tsvector` column (`content_tsv`), powered by `plainto_tsquery`. Key SQL fragment:

```sql
ts_rank_cd(documents.content_tsv, plainto_tsquery('simple', :query_text)) AS sparse_rank
```

**Dense Ranking** — pgvector `<=>` (cosine distance) on an embedding column (`embedding` `vector(768)`). Key SQL fragment:

```sql
documents.embedding <=> :query_vector::vector AS dense_distance
```

**RRF Scoring Equation:**

$$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

where `r_m(d)` is the 1-based rank of document `d` in ranking `m`, and `k = 60` is the RRF constant. The blended rank incorporates a weight `alpha \in [0,1]`:

with default `alpha = 0.5` (equal weighting). Results are sorted by descending fused RRF score.

**Multi-Tenant Isolation**: All queries enforce `WHERE tenant_id = :tenant_id` via declarative list partitioning on the `tenant_id` column. Cross-tenant leakage is impossible at the DB level.

---

### B. Tenant-Isolated Semantic Execution Cache (`backend/cache/semantic_cache.py`)

The semantic cache sits between the agent orchestration layer and the LLM engine, providing embedding-driven similarity caching:

- **Key Pattern**: `cache:semantic:{tenant_id}:{hash}` where `hash` is the first 64 bits of `SHA-256(prompt)`.
- **Embeddings**: Sentence-Transformer `all-MiniLM-L6-v2` (768-dim), lazy-loaded on cache init.
- **Cosine Similarity**: `cosine(a,b) = a·b / (||a|| ||b||)`, default match threshold `0.92`.
- **Redis Storage**: JSON payload `{"embedding": [...], "response": {...}}` with TTL (default 86400s / 24h).
- **Executor Execution**: `_ainvoke_model()` runs `self.model.encode(prompt)` via `loop.run_in_executor(None, ...)` — zero blocking of the async event loop.
- **Cache Flow**: `get(tenant_id, prompt)` → embed prompt → compare against stored embedding → return cached response if `similarity >= threshold`, else `None`.

---

### C. Model Context Protocol (FastMCP) Server (`backend/mcp/server.py`)

The `FastMCP("stratops-intel-mcp")` server exposes three registered tools via JSON schema. All tool handlers are `async def` for full async compatibility.

| Tool | Signature | Description |
|------|-----------|-------------|
| `query_knowledge_graph` | `query_knowledge_graph(tenant_id: str, entity_name: str, depth: int = 2) -> dict` | Neo4j subgraph traversal centered on `entity_name` within `depth` hops. Enforces `WHERE n.tenant_id = $tid`. |
| `get_entity_trends` | `get_entity_trends(tenant_id: str, entity_id: str, timeframe_days: int = 30) -> dict` | Invokes `TrendAnalyzerNode` for time-series signal evaluation (Z-scores, STL decomposition, LLM narrative). |
| `run_sec_hybrid_retrieval` | `run_sec_hybrid_retrieval(tenant_id: str, query: str) -> dict` | Delegates to `VectorStore.hybrid_search()` with RRF reranking. |

All tools generate full JSON schema automatically via the FastMCP decorators and are discoverable via `list_tools()`.

---

### D. Deterministic Governance & Tenant RLS

- **PostgreSQL Row-Level Security (RLS)**: Declarative `ENABLE ROW LEVEL SECURITY` on all data tables with policy `USING (tenant_id = current_setting('app.tenant_id'))`. No application-level bypass possible.
- **Declarative Partitioning**: Tables partitioned by `tenant_id` via `CREATE TABLE ... PARTITION BY LIST (tenant_id)`. Each partition has its own pgvector HNSW index — no cross-tenant index contention.
- **OPA/Rego Constraint Hooks**: Declarative policy-as-code for cross-cutting concerns (e.g., "no entity may be created without `valid_from`/`valid_to` bounds").
- **Pointer-Only State in LangGraph**: Workflow state contains only MinIO/S3 URIs (not full payloads). Checkpoints are < 5KB vs ~1.5MB raw payloads, enabling horizontal scaling of LangGraph workers.
- **Exactly-Once Semantics**: Redis Streams consumer groups with `xack` acknowledgment. DLQ routing for failed messages.

---

## Configuration & Environment Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://localhost/stratops_intel` | AsyncSQLAlchemy DSN with pgvector enabled |
| `NEO4J_URI` | `neo4j://localhost:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j auth user |
| `NEO4J_PASSWORD` | `stratops` | Neo4j auth password |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis async connection URL |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-Transformer model for cache embedding |
| `SEMANTIC_CACHE_THRESHOLD` | `0.92` | Cosine similarity threshold for cache hit |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | `100` | Lua sliding-window rate limiter ceiling |
| `APP_TENANT_ID` | (set per-request) | PostgreSQL RLS session variable |

---

## Local Setup & Fast Start

```bash
# 1. Clone and prepare virtual environment
git clone https://github.com/aragit/stratops-intel.git
cd stratops-intel
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. Start infrastructure via Docker Compose
docker-compose -f docker/docker-compose.yml up -d postgres redis neo4j minio

# 4. Apply Alembic migrations
cd backend && alembic upgrade head

# 5. Launch FastMCP server (or API server)
cd backend && python -m backend.mcp.server
# or: uvicorn api.gateway:app --host 0.0.0.0 --port 8000
```

---

## Verification & Quality Gates

Execute the following from `backend/`:

### Unit Test Suite (hard timeout)

```bash
cd backend && python -m pytest tests/unit/ --timeout=15
```

Expected output:

```
================ 350 passed, 5 skipped, 0 failed in 31.2s =================
```

### Code Linting & Type Checking

```bash
ruff check backend/
mypy backend/ --ignore-missing-imports
```

- `ruff`: Pre-existing style issues only (no new violations from v0.6.0 additions).
- `mypy`: 1 pre-existing alembic module-mapping issue (not related to core logic).

### Post-Verification Checklist

- [ ] `git status` clean (release commit + tag `v0.6.0` pushed)
- [ ] `git log -1` confirms conventional commit message
- [ ] All 327 unit tests pass in isolation and full suite
- [ ] `test_neo4j_client.py` → 8/8 pass
- [ ] No order-dependent test pollution (all tests pass regardless of collection order)

---

## Project Structure

```
stratops-intel/
├── backend/                          # Core backend package
│   ├── api/                          # FastAPI gateway, auth, middleware
│   ├── db/                           # Postgres RLS, Alembic, Neo4j client
│   ├── streams/                      # Redis Streams producer/consumer
│   ├── ingestion/                    # Source adapters (web, SEC, earnings, patents)
│   ├── intelligence/                 # LangGraph agents (extractor, correlation, trend, anomaly, composer, delta)
│   ├── workers/                      # Ingestion, intelligence, graph writer
│   ├── alerts/                       # Rule engine, router (Slack/email/webhook)
│   ├── briefings/                    # Repository, delta generator
│   ├── cache/                        # **NEW**: Tenant-isolated semantic cache
│   ├── db/vector_store.py            # **NEW**: Hybrid pgvector RRF retrieval
│   ├── mcp/                          # **NEW**: FastMCP server + tool interface
│   └── tests/
│       └── unit/                     # 327 tests across 28 test files
├── bentoml/                          # BentoML + vLLM model services
├── frontend/                         # Next.js 14 App Router dashboard
├── docker/                           # Docker Compose definitions
│   ├── postgres                      # PostgreSQL 16 + pgvector
│   ├── redis                         # Lua sliding-window rate limiter
│   ├── neo4j                         # Temporal knowledge graph
│   └── minio                         # S3-compat artifact storage
├── docs/                             # ADRs, benchmarks, deployment guides
├── benchmarks/                       # Performance benchmarks (pre-v0.6.0)
└── README.md                         # This file
```

---

## Portfolio Narrative

> **"For regulated domains, I build deterministic state machines with symbolic kernels. For intelligence synthesis, I build event-driven platforms with BentoML + vLLM model meshes and temporal knowledge graphs. I choose the architecture to fit the problem's non-functional requirements — not the other way around."**

---

## License

MIT — See [LICENSE](LICENSE) for details.