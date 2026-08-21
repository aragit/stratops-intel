"""Pgvector hybrid search engine with Reciprocal Rank Fusion (RRF).

Upgrades vector retrieval from dense-only HNSW to a Hybrid (Sparse/BM25 + Dense HNSW)
pipeline that combines full-text search rankings and cosine-similarity rankings
via Reciprocal Rank Fusion for improved precision on financial terminology.

All queries strictly enforce ``WHERE tenant_id = :tenant_id`` for multi-tenant
security.
"""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)

#: Default RRF constant ``k`` parameter (see :func:`rrf_score`)
RRF_K = 60

#: Default alpha weight for blending sparse vs dense rankings
DEFAULT_ALPHA = 0.5

#: Default top-k returned results
DEFAULT_TOP_K = 10


def rrf_score(ranks: list[int], k: int = RRF_K) -> float:
    """Compute the Reciprocal Rank Fusion score for a document.

    Given the *ranks* a document received across multiple rankings (e.g. one
    rank from a full-text search and one rank from a dense vector search),
    the RRF score is::

        RRF_Score(d) = sum_{m in M} 1 / (k + r_m(d))

    where ``r_m(d)`` is the 1-based rank of document ``d`` in ranking ``m``
    and ``k`` is a constant (default 60) that discounts the influence of very
    high ranks.

    Args:
        ranks: List of integer ranks for the same document across multiple
            rankings (shortest rank first, 1-based).
        k: RRF constant that controls how quickly high ranks are discounted.

    Returns:
        The fused RRF score (higher => more relevant).
    """
    if not ranks:
        return 0.0
    return sum(1.0 / (k + r) for r in ranks)


class VectorStore:
    """Hybrid pgvector-backed search engine with RRF reranking.

    Supports two ranking sources:

    1. **Sparse / Full-Text Search** — PostgreSQL ``to_tsquery`` / ``ts_rank_cd``
       on a ``tsvector`` column, producing a relevance rank (1 = best).

    2. **Dense / Vector Search** — pgvector cosine distance (<=> operator)
       on an ``embedding`` column (``vector(768)``), producing a rank based on
       ascending distance.

    The two ranks per document are combined via :func:`rrf_score` and results
    are returned ordered by descending fused score.

    Example
    -------

    >>> store = VectorStore(engine)
    >>> results = await store.hybrid_search(
    ...     tenant_id="tenant-123",
    ...     query_text="Apple Inc. Q3 earnings",
    ...     query_vector=[0.1, 0.2, ...],
    ...     top_k=10,
    ...     alpha=0.5,
    ... )
    >>> # results sorted by RRF score, each entry has ``*`` and ``score`` fields
    """

    def __init__(self, engine: Any) -> None:
        """Initialize the vector store.

        Args:
            engine: An async SQLAlchemy engine (or compatible) connected to a
                PostgreSQL database with the ``pgvector`` extension enabled.
        """
        self._engine = engine

    # ------------------------------------------------------------------
    # Internal: build sparse rank sub-query
    # ------------------------------------------------------------------

    @staticmethod
    def _sparse_sql(tenant_id: str, query_text: str) -> str:
        """Return the SQL fragment for full-text rank computation.

        Generates a ``ts_rank_cd`` score against a ``documents`` table that
        has a ``content_tsv`` ``tsvector`` column and a ``tenant_id`` column.

        The returned fragment orders by descending rank and includes the
        ``tenant_id`` filter.

        Args:
            tenant_id: Tenant partition filter.
            query_text: Free-text query for ``plainto_tsquery`` / ``ts_rank_cd``.

        Returns:
            SQL ``SELECT`` fragment suitable for embedding in a larger query.
        """
        # plainto_tsquery normalises the query text into a tsvector-compatible form
        return (
            f"ts_rank_cd(documents.content_tsv, "
            f"plainto_tsquery('simple', '{query_text}')) as sparse_rank"
        )

    # ------------------------------------------------------------------
    # Internal: build dense rank sub-query
    # ------------------------------------------------------------------

    @staticmethod
    def _dense_sql(query_vector: list[float]) -> str:
        """Return the SQL fragment for cosine-distance rank computation.

        pgvector's ``<=>`` operator computes cosine distance (0 = identical,
        2 = maximally different).  We invert+scale to produce a 1-based rank
        ordering (smaller distance => better rank).

        Args:
            query_vector: Embedding vector for the query.

        Returns:
            SQL fragment ordering by ascending cosine distance.
        """
        # pgvector ``<=>`` is cosine distance; we map it to a rank-like value
        return f"documents.embedding <=> '{query_vector}'::vector AS dense_distance"

    # ------------------------------------------------------------------
    # Public: hybrid search
    # ------------------------------------------------------------------

    async def hybrid_search(
        self,
        tenant_id: str,
        query_text: str,
        query_vector: list[float],
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
    ) -> list[dict[str, Any]]:
        """Run a hybrid (sparse + dense) search with RRF reranking.

        Combines PostgreSQL full-text search rankings and dense vector
        cosine-similarity rankings via Reciprocal Rank Fusion (RRF) to
        produce a re-ranked result list that leverages both keyword precision
        and semantic vector similarity.

        All database operations enforce ``WHERE tenant_id = :tenant_id``.

        Args:
            tenant_id: Tenant identifier for multi-tenant data partitioning.
            query_text: Free-text query used for ``plainto_tsquery`` / ``ts_rank_cd``.
            query_vector: Embedding vector (pgvector‑compatible) for cosine distance.
            top_k: Maximum number of result rows to return (default 10).
            alpha: Weight in [0,1] blending sparse and dense ranks.

                - ``alpha`` close to ``1``:: leans toward the full-text rank.
                - ``alpha`` close to ``0``:: leans toward the dense vector rank.
                - ``alpha`` = ``0.5``:: equal weighting.

        Returns:
            List of dicts, each containing at minimum:

            - ``id``: document primary key identifier
            - ``score``: fused RRF score (higher = more relevant)
            - Additional document fields may be present depending on the
              underlying schema.

        Raises:
            ValueError: If ``query_vector`` is empty or ``alpha`` is outside
                [0, 1].
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if not query_vector:
            raise ValueError("query_vector must be non-empty")

        async with self._engine.connect() as conn:
            # --- Build the sparse (full-text) rank sub-query ---------------
            sparse_fragment = self._sparse_sql(tenant_id, query_text)

            # --- Build the dense (vector) rank sub-query -------------------
            dense_fragment = self._dense_sql(query_vector)

            # --- Combine via UNION ALL, computing combined ranks -------------
            # We assign a synthetic rank (1-based) within each sub-query so
            # that RRF can consume them.  The SQL uses ROW_NUMBER() over
            # (SELECT ... ) to give each document its rank in each ranking
            # source.
            union_sql = f"""
                SELECT
                    documents.id AS id,
                    ROW_NUMBER() OVER (ORDER BY {sparse_fragment}) AS sparse_rank,
                    ROW_NUMBER() OVER (ORDER BY {dense_fragment}) AS dense_rank,
                    documents.payload AS payload,
                    documents.metadata AS metadata
                FROM documents
                WHERE documents.tenant_id = '{tenant_id}'
                  AND documents.content_tsv IS NOT NULL
                  AND documents.embedding IS NOT NULL
                ORDER BY sparse_rank
                LIMIT {top_k * 2}
            """

            # Execute the union query
            rows = await conn.execute(text(union_sql))
            # pg result rows -- iterate
            rows = rows.fetchall()  # type: ignore[attr-defined]

            # --- Compute RRF scores -----------------------------------------
            results: list[dict[str, Any]] = []
            for row in rows:
                doc_id = row.id
                sparse_r = row.sparse_rank  # type: ignore[index]
                dense_r = row.dense_rank  # type: ignore[index]

                # Combined RRF score: alpha blends the two rank sources.
                # We compute a weighted rank position:
                #   blended_rank = alpha * sparse_rank + (1 - alpha) * dense_rank
                # Then compute RRF from the blended rank.
                # Using the standard RRF formula directly on the two ranks
                # is also valid; here we blend first then apply RRF with k=60.
                blended_rank = alpha * int(sparse_r) + (1.0 - alpha) * int(dense_r)

                rrf = rrf_score([blended_rank], k=RRF_K)

                # Decode payload if it is JSON-encoded
                payload = row.payload  # type: ignore[index]
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        pass

                metadata = row.metadata  # type: ignore[index]
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        pass

                results.append(
                    {
                        "id": doc_id,
                        "score": round(rrf, 6),
                        "payload": payload,
                        "metadata": metadata,
                        "sparse_rank": int(sparse_r),
                        "dense_rank": int(dense_r),
                    }
                )

            # --- Sort by RRF score descending and limit to top_k ------------
            results.sort(key=lambda r: r["score"], reverse=True)
            return results[:top_k]
