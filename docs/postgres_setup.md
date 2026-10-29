# PostgreSQL + pgvector storage

By default LightRAG keeps everything in files under `WORKING_DIR`
(JSON key-value stores, NanoVectorDB, NetworkX). For large corpora
(thousands of pages) PostgreSQL with pgvector is sturdier: concurrent
access, full-precision `float32` vectors (NanoVectorDB stores `float16`),
and standard backup tooling.

## 1. Start a server

```bash
export POSTGRES_PASSWORD=change-me
docker compose -f docker-compose.pg.yml up -d
```

## 2. Point RAG-Anything at it

```python
import os
from lightrag.utils import EmbeddingFunc
from raganything import RAGAnything, RAGAnythingConfig

os.environ.update(
    POSTGRES_HOST="localhost", POSTGRES_PORT="5432", POSTGRES_USER="rag",
    POSTGRES_PASSWORD="change-me", POSTGRES_DATABASE="rag",
)

rag = RAGAnything(
    config=RAGAnythingConfig(working_dir="./rag_storage"),
    llm_model_func=..., vision_model_func=...,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024, max_token_size=8192, func=embed,
        model_name="bge-m3",  # suffixes the vector tables: one model per table set
    ),
    lightrag_kwargs=dict(
        workspace="my_project",
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="NetworkXStorage",  # or PGGraphStorage if Apache AGE is installed
    ),
)
```

Notes:

* **Graph.** `PGGraphStorage` needs the Apache AGE extension (not in the
  `pgvector` image). With `NetworkXStorage` the graph stays a file in
  `WORKING_DIR/<workspace>/`: keep that directory with the database.
* **Parse cache.** PostgreSQL's KV backend only has LightRAG's fixed tables,
  so RAG-Anything keeps its parse cache and multimodal status as JSON files in
  `WORKING_DIR/<workspace>/`. Keep `WORKING_DIR` on persistent storage: it is
  what lets an interrupted 12,000-page ingestion resume without re-parsing.
* **Embedding model.** Set `model_name` and never change the model or its
  dimension on an existing database; export, then import with `reembed=True`.

## 3. Back up and move the data

```python
from raganything.export import (
    create_snapshot, restore_snapshot, export_bundle, import_bundle,
)

await create_snapshot(rag, "backup.tar.gz")        # exact copy: working dir + pg_dump
restore_snapshot("backup.tar.gz", "./rag_storage")  # before creating the instance

await export_bundle(rag, "kb.zip", include_assets=True)  # portable bundle
await import_bundle(other_rag, "kb.zip")                 # file <-> PostgreSQL, server <-> server
```

* A **snapshot** needs the PostgreSQL client tools (`pg_dump`, `pg_restore`)
  and dumps the whole database named by `POSTGRES_DATABASE`; restore it into
  an empty database that already has `CREATE EXTENSION vector`.
* A **bundle** goes through LightRAG's storage interfaces, so it works between
  any of the backends above. Vectors are reused when the embedding model and
  dimension match; otherwise import refuses unless `reembed=True`.
  `mode="replace", confirm_replace=True` empties the target first.
* `python -m raganything.export kb.zip` prints a bundle's or snapshot's manifest.
