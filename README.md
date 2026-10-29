# 🚀 RAG-Anything (fork technique)

**Index Outlook mails and large technical documentation (PDFs of several thousand pages, plans, multi-column tables) with local models, PostgreSQL storage and a search interface that cites its sources.**

*Technical-documentation fork of [RAG-Anything](https://github.com/HKUDS/RAG-Anything) — Copyright (c) 2025 Data Intelligence Lab@HKU, MIT License. Not affiliated with or endorsed by the original authors.*

<p>
  <a href="https://github.com/HKUDS/RAG-Anything"><img src="https://img.shields.io/badge/based%20on-HKUDS%2FRAG--Anything-4ecdc4?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge&labelColor=1a1a2e"></a>
  <img src="https://img.shields.io/badge/python-3.10--3.13-4ecdc4?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e">
</p>

[Fork notice](#-fork-notice-and-attribution) · [What this fork adds](#-what-this-fork-adds) · [Install](#-installation-and-first-run) · [Documentation](docs/technical_docs.md) · [Original README](https://github.com/HKUDS/RAG-Anything#readme)

---

<details>
<summary>News from the original project</summary>

- [X] [2026.06]🎯📢 🎉 [LightRAG](https://github.com/HKUDS/LightRAG) enables multimodal RAG through native integration of RAG-Anything.
- [X] [2025.10]🎯📢 🚀 We have released the technical report of [RAG-Anything](http://arxiv.org/abs/2510.12323). Access it now to explore our latest research findings.
- [X] [2025.08]🎯📢 🔍 RAG-Anything now features **VLM-Enhanced Query** mode! When documents include images, the system seamlessly integrates them into VLM for advanced multimodal analysis, combining visual and textual context for deeper insights.
- [X] [2025.07]🎯📢 RAG-Anything now features a [context configuration module](docs/context_aware_processing.md), enabling intelligent integration of relevant contextual information to enhance multimodal content processing.
- [X] [2025.07]🎯📢 🚀 RAG-Anything now supports multimodal query capabilities, enabling enhanced RAG with seamless processing of text, images, tables, and equations.
- [X] [2025.07]🎯📢 🎉 RAG-Anything has reached 1k🌟 stars on GitHub! Thank you for your incredible support and valuable contributions to the project.

</details>

---

## 🧩 What this fork adds

Changes made on top of the upstream project to index Outlook mails and very large technical PDFs (plans, multi-column tables) with local models. Details and settings: [docs/technical_docs.md](docs/technical_docs.md).

| Addition | Summary |
|---|---|
| 📧 **Outlook `.msg`** | Headers, body and recursive attachments (PDF, Office, images, nested `.msg`). `pip install -e ".[email]"` |
| 📚 **Very large PDFs** | Parsed in resumable windows (default 200 pages) with checkpoints, bad-page isolation and cross-page table stitching, for documents of several thousand pages |
| 📐 **Technical drawings** | Plan pages detected, rendered as overview + overlapping tiles and described by the vision model with a drawing-specific prompt |
| 🔌 **Resilient embeddings** | Waits out model-server disconnections, validates every vector, resumes interrupted documents |
| 🗄️ **PostgreSQL + pgvector** | Optional storage backend ([docs/postgres_setup.md](docs/postgres_setup.md), `docker-compose.pg.yml`) |
| 📦 **Export / import** | Exact snapshots and a portable bundle between file storage and PostgreSQL |
| 🌐 **Web interface and CLI** | Ingestion page and search page; answers cite their sources (📄 🖼️ 📊 📐 ✉️) with links to the original at the cited page and a hover preview. `pip install -e ".[app]"`, then `python -m raganything.app` or `python -m raganything.cli` |

New code lives in `raganything/email_parser.py`, `pdf_windowing.py`, `drawing_pages.py`, `resilient_embedding.py`, `resume.py`, `export.py`, `cli.py` and `raganything/app/`. Upstream modules were extended where needed (parsers, processor, configuration, image processor and prompts).

---

## 📦 Installation and first run

### 1. Prerequisites

| Need | Details |
|---|---|
| Python | 3.10 to 3.13 (MinerU 3.x does not support 3.14) |
| Git | to clone this repository |
| A model server | any OpenAI-compatible server for a **text/vision model** and an **embedding model**, e.g. [Ollama](https://ollama.com) or vLLM |
| LibreOffice | only to ingest Word/Excel/PowerPoint files (and Office attachments of mails) |
| PostgreSQL + pgvector | optional (recommended for large corpora): Docker, or `pg_dump`/`pg_restore` for snapshots |
| GPU | optional but strongly advised for MinerU and the vision model on large PDFs |

### 2. Install the fork (from source)

```bash
git clone https://github.com/Willtahiti/RAG-Anything.git
cd RAG-Anything

python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate

# Mails (.msg), web/CLI interface and drawing rendering
pip install -e ".[email,app,image]"
```

Other extras: `.[text]` (TXT/MD), `.[markdown]`, `.[paddleocr]`, `.[audio]`, `.[video]`, or `.[all]` for everything (large download). The first MinerU run downloads its models; see [docs/offline_setup.md](docs/offline_setup.md) for air-gapped machines.

Check the parser: `mineru --version` (should print a 3.x version).

> The PyPI package `raganything` is the **original** project and does not contain this fork's features: always install from this repository.

### 3. Start the models (example with Ollama)

```bash
ollama pull qwen2.5vl:7b      # reads text and images, including plans
ollama pull bge-m3            # embeddings, 1024 dimensions
```

### 4. Configure

Create a `.env` file in the folder where you run the commands (it is loaded automatically), or export the variables:

```bash
RAG_LLM_BASE_URL=http://localhost:11434/v1
RAG_LLM_MODEL=qwen2.5vl:7b
RAG_EMBED_MODEL=bge-m3
RAG_EMBED_DIM=1024
# RAG_VISION_MODEL=...        # defaults to RAG_LLM_MODEL
# RAG_API_KEY=...             # if your server requires a key

WORKING_DIR=./rag_storage     # index and job state: keep it on persistent storage
OUTPUT_DIR=./output           # parsed content and extracted images
```

Never change `RAG_EMBED_MODEL` or `RAG_EMBED_DIM` on an existing index: export, change, then import with `--reembed` (see [docs/technical_docs.md](docs/technical_docs.md)).

**Optional: PostgreSQL + pgvector**

```bash
export POSTGRES_PASSWORD=choose-a-password
docker compose -f docker-compose.pg.yml up -d
```

Then add to your configuration: `RAG_STORAGE=postgres`, `POSTGRES_HOST=localhost`, `POSTGRES_PORT=5432`, `POSTGRES_USER=rag`, `POSTGRES_PASSWORD=<same password>`, `POSTGRES_DATABASE=rag`. Details: [docs/postgres_setup.md](docs/postgres_setup.md).

### 5. First run

Web interface (listens on `127.0.0.1:8080`):

```bash
python -m raganything.app
```

Open <http://127.0.0.1:8080/ingest> to drop files in, then <http://127.0.0.1:8080/> to ask questions. Answers carry numbered sources that open the original at the cited page.

Command line (same engine):

```bash
python -m raganything.cli ingest mail.msg manual.pdf some_folder/
python -m raganything.cli query "Which cable section feeds the main switchboard?"
python -m raganything.cli resume          # continue after an interruption
python -m raganything.cli docs            # list indexed documents
```

### 6. Useful settings for large PDFs

| Variable | Default | Meaning |
|---|---|---|
| `PDF_WINDOW_PAGES` | 200 | pages parsed per run, with a checkpoint after each |
| `PDF_WINDOW_THRESHOLD` | 400 | PDFs up to this length are parsed in one go |
| `PDF_WINDOW_TIMEOUT` | none | seconds allowed per window |
| `ENABLE_DRAWING_PROCESSING` | true | render plan pages in tiles for the vision model |
| `MAX_IMAGE_SIDE` | 0 (off) | downscale images sent to the vision model (e.g. 1792) |
| `RAG_MAX_OUTAGE_S` | 3600 | how long to wait for a model server that went away |

To expose the web interface beyond your machine, set `RAG_APP_HOST=0.0.0.0` **and** `RAG_APP_TOKEN=<secret>` (it refuses to start without a token on a non-local address).

---

## 🌟 Overview

This project turns a collection of mails and technical documents into a searchable knowledge base that answers in natural language and **shows where each statement comes from**.

It builds on the multimodal RAG framework published by the Data Intelligence Lab@HKU (parsing, image/table/equation analysis, knowledge graph, hybrid retrieval, all on top of [LightRAG](https://github.com/HKUDS/LightRAG)); the framework's design and the underlying research are theirs — see their [README](https://github.com/HKUDS/RAG-Anything#readme) and [paper](https://arxiv.org/abs/2510.12323) for the detailed description.

### How it works

1. **Ingest** — Outlook `.msg` files (with attachments), PDFs of any length, Office documents and images are parsed. Large PDFs are processed window by window with checkpoints; plan pages are rendered in tiles.
2. **Understand** — a local vision/text model describes images, plans and tables; entities and relations are extracted into a knowledge graph.
3. **Index** — text, descriptions and graph go to file storage or PostgreSQL + pgvector. Embeddings favor quality and wait out server disconnections.
4. **Search** — questions combine vector and graph retrieval; answers cite numbered sources (📄 🖼️ 📊 📐 ✉️) linked to the original page, with a hover preview.
5. **Move** — knowledge bases can be exported and imported between file storage and PostgreSQL.

### Typical use

- Technical documentation of a building or an installation (electrical, plumbing, HVAC): manuals, specifications, plans, multi-column tables.
- Project mail archives (`.msg`) searched together with the documents they refer to.

---

## 🚀 Quick Start

*Initialize Your AI Journey*

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212284158-e840e285-664b-44d7-b79b-e264b5e54825.gif" width="400">
</div>

### Installation (original project)

> **Inherited from the original documentation.** For this fork, follow [Installation and first run](#-installation-and-first-run) instead: the PyPI package `raganything` below is the original project and lacks this fork's features. The usage and configuration sections that follow still apply.


> **Python version:** RAG-Anything supports Python 3.10 and later, but its default MinerU parser needs MinerU 3.x, which supports Python 3.10 to 3.13 only. On Python 3.14, RAG-Anything is installed without MinerU: use the Docling parser there (`pip install docling`, then `RAGAnythingConfig(parser="docling")`), or create the environment with Python 3.13 to use MinerU.

#### Option 1: Install from PyPI (Recommended)

```bash
# Basic installation
pip install raganything

# With optional dependencies for extended format support:
pip install 'raganything[all]'              # All optional features (Docling not included)
pip install 'raganything[image]'            # Image format conversion (BMP, TIFF, GIF, WebP)
pip install 'raganything[text]'             # Text file processing (TXT, MD)
pip install 'raganything[image,text]'       # Multiple features
```

#### Option 2: Install from Source
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup the project with uv
git clone https://github.com/HKUDS/RAG-Anything.git
cd RAG-Anything

# Install the package and dependencies in a virtual environment
uv sync

# If you encounter network timeouts (especially for opencv packages):
# UV_HTTP_TIMEOUT=120 uv sync

# Run commands directly with uv (recommended approach)
uv run python examples/raganything_example.py --help

# Install with optional dependencies
uv sync --extra image --extra text  # Specific extras
uv sync --all-extras                 # All optional features
```

#### Optional Dependencies

- **`[image]`** - Enables processing of BMP, TIFF, GIF, WebP image formats (requires Pillow)
- **`[text]`** - Enables processing of TXT and MD files (requires ReportLab)
- **`[all]`** - Includes all Python optional dependencies

> **⚠️ Office Document Processing Requirements:**
> - Office documents (.doc, .docx, .ppt, .pptx, .xls, .xlsx) require **LibreOffice** installation
> - Download from [LibreOffice official website](https://www.libreoffice.org/download/download/)
> - **Windows**: Download installer from official website
> - **macOS**: `brew install --cask libreoffice`
> - **Ubuntu/Debian**: `sudo apt-get install libreoffice`
> - **CentOS/RHEL**: `sudo yum install libreoffice`

**Check MinerU installation:**

On Python 3.10 to 3.13, RAG-Anything installs MinerU 3.x as a dependency (`mineru[core]>=3.4.1,<4`). MinerU 3.x does not support Python 3.14, so on Python 3.14 MinerU is not installed and the MinerU parser is unavailable (see the Python version note above). MinerU 4.x is not supported yet: it replaced the command-line interface that RAG-Anything drives. With MinerU 4.x installed, `check_parser_installation()` returns `False` and a `RAGAnything` instance using the MinerU parser fails to initialize with an error that asks for a 3.x release. If `mineru --version` reports 4.x, install a 3.x release in the same environment (Python 3.10 to 3.13):

```bash
pip install -U "mineru[core]>=3.4.1,<4"
# Optional: remove two packages only MinerU 4.x uses; docvortex otherwise makes
# `pip check` report a pypdfium2 conflict. Keep mineru-magika: it provides the
# `magika` module MinerU 3.x imports. A fresh virtual environment also works.
pip uninstall -y docvortex mineru-llama-cpp
```

On minimal Linux or Docker images (for example `python:3.12-slim`), MinerU 3.x also needs system libraries for OpenCV. Without them, `mineru --version` fails with `libxcb.so.1` or `libGL.so.1: cannot open shared object file`, and the installation check reports that error. On Debian or Ubuntu:

```bash
apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0
```

```bash
# Verify installation (should report a 3.x version)
mineru --version

# Check if properly configured
python -c "from raganything import RAGAnything; rag = RAGAnything(); print('✅ MinerU installed properly' if rag.check_parser_installation() else '❌ MinerU installation issue')"
```

Models are downloaded automatically on first use. For manual download, refer to [MinerU Model Source Configuration](https://github.com/opendatalab/MinerU/blob/mineru-3.4.5-released/docs/en/usage/model_source.md) (MinerU 3.x docs; MinerU's current docs describe MinerU 4.x, which RAG-Anything does not support yet).

### Usage Examples

#### 1. End-to-End Document Processing

```python
import asyncio
import base64
from functools import partial
from raganything import RAGAnything, RAGAnythingConfig
from raganything.image_signature import image_mime_type
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def main():
    # Set up API configuration
    api_key = "your-api-key"
    base_url = "your-base-url"  # Optional

    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        parser="mineru",  # Parser selection: mineru, docling, or paddleocr
        parse_method="auto",  # Parse method: auto, ocr, or txt
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define LLM model function
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    # Define vision model function for image processing
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Traditional single image format
        elif image_data:
            # Label the image with its real media type (PNG, GIF, WebP, ...),
            # identified from its first bytes (24 base64 characters)
            mime_type = image_mime_type(base64.b64decode(image_data[:24])) or "image/jpeg"
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                },
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Pure text format
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # Define embedding function
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        ),
    )

    # Initialize RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # Process a document
    await rag.process_document_complete(
        file_path="path/to/your/document.pdf",
        output_dir="./output",
        parse_method="auto"
    )

    # Query the processed content
    # Pure text query - for basic knowledge base search
    text_result = await rag.aquery(
        "What are the main findings shown in the figures and tables?",
        mode="hybrid"
    )
    print("Text query result:", text_result)

    # Multimodal query with specific multimodal content
    multimodal_result = await rag.aquery_with_multimodal(
    "Explain this formula and its relevance to the document content",
    multimodal_content=[{
        "type": "equation",
        "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
        "equation_caption": "Document relevance probability"
    }],
    mode="hybrid"
)
    print("Multimodal query result:", multimodal_result)

if __name__ == "__main__":
    asyncio.run(main())
```

> **Text indexing**: `process_document_complete()` and `insert_content_list()` return only after LightRAG has indexed the document's text; if another insert is running LightRAG's pipeline, they wait for that run to process it. If LightRAG fails the text (for example, the LLM is unavailable) and no run is left to retry it, they raise `RuntimeError` and the document stays `failed` in `doc_status`; LightRAG retries it on its next pipeline run, such as when the file is processed again. See [multimodal ingestion concurrency](docs/batch_processing.md#multimodal-ingestion-concurrency).

#### 2. Direct Multimodal Content Processing

```python
import asyncio
import base64
from functools import partial
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from raganything.modalprocessors import ImageModalProcessor, TableModalProcessor
from raganything.image_signature import image_mime_type

def image_data_url(image_data):
    # Label the image with its real media type (PNG, GIF, WebP, ...),
    # identified from its first bytes (24 base64 characters)
    mime_type = image_mime_type(base64.b64decode(image_data[:24])) or "image/jpeg"
    return f"data:{mime_type};base64,{image_data}"


async def process_multimodal_content():
    # Set up API configuration
    api_key = "your-api-key"
    base_url = "your-base-url"  # Optional

    # Initialize LightRAG
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        ),
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        )
    )
    await rag.initialize_storages()

    # Process an image
    image_processor = ImageModalProcessor(
        lightrag=rag,
        modal_caption_func=lambda prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs: openai_complete_if_cache(
            "gpt-4o",
            "",
            system_prompt=None,
            history_messages=[],
            messages=[
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url(image_data)}}
                ]} if image_data else {"role": "user", "content": prompt}
            ],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        ) if image_data else openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    )

    image_content = {
        "img_path": "path/to/image.jpg",
        "image_caption": ["Figure 1: Experimental results"],
        "image_footnote": ["Data collected in 2024"]
    }

    description, entity_info, _ = await image_processor.process_multimodal_content(
        modal_content=image_content,
        content_type="image",
        file_path="research_paper.pdf",
        entity_name="Experimental Results Figure"
    )

    # Process a table
    table_processor = TableModalProcessor(
        lightrag=rag,
        modal_caption_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    )

    table_content = {
        "table_body": """
        | Method | Accuracy | F1-Score |
        |--------|----------|----------|
        | RAGAnything | 95.2% | 0.94 |
        | Baseline | 87.3% | 0.85 |
        """,
        "table_caption": ["Performance Comparison"],
        "table_footnote": ["Results on test dataset"]
    }

    description, entity_info, _ = await table_processor.process_multimodal_content(
        modal_content=table_content,
        content_type="table",
        file_path="research_paper.pdf",
        entity_name="Performance Results Table"
    )

if __name__ == "__main__":
    asyncio.run(process_multimodal_content())
```

#### 3. Batch Processing

```python
# Process multiple documents
await rag.process_folder_complete(
    folder_path="./documents",
    output_dir="./output",
    file_extensions=[".pdf", ".docx", ".pptx"],
    recursive=True,
    max_workers=4
)
```

#### 4. Custom Modal Processors

```python
from raganything.modalprocessors import GenericModalProcessor

class CustomModalProcessor(GenericModalProcessor):
    async def process_multimodal_content(self, modal_content, content_type, file_path, entity_name):
        # Your custom processing logic
        enhanced_description = await self.analyze_custom_content(modal_content)
        entity_info = self.create_custom_entity(enhanced_description, entity_name)
        return await self._create_entity_and_chunk(enhanced_description, entity_info, file_path)
```

#### 5. Query Options

RAG-Anything provides three types of query methods:

**Pure Text Queries** - Direct knowledge base search using LightRAG:
```python
# Different query modes for text queries
text_result_hybrid = await rag.aquery("Your question", mode="hybrid")
text_result_local = await rag.aquery("Your question", mode="local")
text_result_global = await rag.aquery("Your question", mode="global")
text_result_naive = await rag.aquery("Your question", mode="naive")

# Synchronous version
sync_text_result = rag.query("Your question", mode="hybrid")
```

**VLM Enhanced Queries** - Automatically analyze images in retrieved context using VLM:
```python
# VLM enhanced query (automatically enabled when vision_model_func is provided)
vlm_result = await rag.aquery(
    "Analyze the charts and figures in the document",
    mode="hybrid"
    # vlm_enhanced=True is automatically set when vision_model_func is available
)

# Manually control VLM enhancement
vlm_enabled = await rag.aquery(
    "What do the images show in this document?",
    mode="hybrid",
    vlm_enhanced=True  # Force enable VLM enhancement
)

vlm_disabled = await rag.aquery(
    "What do the images show in this document?",
    mode="hybrid",
    vlm_enhanced=False  # Force disable VLM enhancement
)

# When documents contain images, VLM can see and analyze them directly
# The system will automatically:
# 1. Retrieve relevant context containing image paths
# 2. Load and encode images as base64
# 3. Send both text context and images to VLM for comprehensive analysis
```

> **Streaming**: with `stream=True`, a VLM enhanced query that sends retrieved images returns the vision model's async iterator instead of a string (without images it falls back to LightRAG's streaming answer). `vision_model_func` then receives `stream=True`, so a custom function must accept `**kwargs` and pass them to a client that can stream, as the example above does; one without `**kwargs` raises `TypeError`.

**Multimodal Queries** - Enhanced queries with specific multimodal content analysis:
```python
# Query with table data
table_result = await rag.aquery_with_multimodal(
    "Compare these performance metrics with the document content",
    multimodal_content=[{
        "type": "table",
        "table_data": """Method,Accuracy,Speed
                        RAGAnything,95.2%,120ms
                        Traditional,87.3%,180ms""",
        "table_caption": "Performance comparison"
    }],
    mode="hybrid"
)

# Query with equation content
equation_result = await rag.aquery_with_multimodal(
    "Explain this formula and its relevance to the document content",
    multimodal_content=[{
        "type": "equation",
        "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
        "equation_caption": "Document relevance probability"
    }],
    mode="hybrid"
)
```

#### 6. Loading Existing LightRAG Instance

```python
import asyncio
import base64
from functools import partial
from raganything import RAGAnything, RAGAnythingConfig
from raganything.image_signature import image_mime_type
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
import os

async def load_existing_lightrag():
    # Set up API configuration
    api_key = "your-api-key"
    base_url = "your-base-url"  # Optional

    # First, create or load existing LightRAG instance
    lightrag_working_dir = "./existing_lightrag_storage"

    # Check if previous LightRAG instance exists
    if os.path.exists(lightrag_working_dir) and os.listdir(lightrag_working_dir):
        print("✅ Found existing LightRAG instance, loading...")
    else:
        print("❌ No existing LightRAG instance found, will create new one")

    # Create/load LightRAG instance with your configuration
    lightrag_instance = LightRAG(
        working_dir=lightrag_working_dir,
        llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        ),
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        )
    )

    # Initialize storage (this will load existing data if available)
    await lightrag_instance.initialize_storages()
    await initialize_pipeline_status()

    # Define vision model function for image processing
    def vision_model_func(
        prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs
    ):
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Traditional single image format
        elif image_data:
            # Label the image with its real media type (PNG, GIF, WebP, ...),
            # identified from its first bytes (24 base64 characters)
            mime_type = image_mime_type(base64.b64decode(image_data[:24])) or "image/jpeg"
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                },
                            },
                        ],
                    }
                    if image_data
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Pure text format
        else:
            return lightrag_instance.llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    # Now use existing LightRAG instance to initialize RAGAnything
    rag = RAGAnything(
        lightrag=lightrag_instance,  # Pass existing LightRAG instance
        vision_model_func=vision_model_func,
        # Note: working_dir, llm_model_func, embedding_func, etc. are inherited from lightrag_instance
    )

    # Query existing knowledge base
    result = await rag.aquery(
        "What data has been processed in this LightRAG instance?",
        mode="hybrid"
    )
    print("Query result:", result)

    # Add new multimodal document to existing LightRAG instance
    await rag.process_document_complete(
        file_path="path/to/new/multimodal_document.pdf",
        output_dir="./output"
    )

if __name__ == "__main__":
    asyncio.run(load_existing_lightrag())
```

#### 7. Direct Content List Insertion

For scenarios where you already have a pre-parsed content list (e.g., from external parsers or previous processing), you can directly insert it into RAGAnything without document parsing:

```python
import asyncio
import base64
from functools import partial
from raganything import RAGAnything, RAGAnythingConfig
from raganything.image_signature import image_mime_type
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def insert_content_list_example():
    # Set up API configuration
    api_key = "your-api-key"
    base_url = "your-base-url"  # Optional

    # Create RAGAnything configuration
    config = RAGAnythingConfig(
        working_dir="./rag_storage",
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )

    # Define model functions
    def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, messages=None, **kwargs):
        # If messages format is provided (for multimodal VLM enhanced query), use it directly
        if messages:
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Traditional single image format
        elif image_data:
            # Label the image with its real media type (PNG, GIF, WebP, ...),
            # identified from its first bytes (24 base64 characters)
            mime_type = image_mime_type(base64.b64decode(image_data[:24])) or "image/jpeg"
            return openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt} if system_prompt else None,
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                        ],
                    } if image_data else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
        # Pure text format
        else:
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model="text-embedding-3-large",
            api_key=api_key,
            base_url=base_url,
        ),
    )

    # Initialize RAGAnything
    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    # Example: Pre-parsed content list from external source
    content_list = [
        {
            "type": "text",
            "text": "This is the introduction section of our research paper.",
            "page_idx": 0  # Page number where this content appears
        },
        {
            "type": "image",
            "img_path": "/absolute/path/to/figure1.jpg",  # IMPORTANT: Use absolute path
            "image_caption": ["Figure 1: System Architecture"],
            "image_footnote": ["Source: Authors' original design"],
            "page_idx": 1  # Page number where this image appears
        },
        {
            "type": "table",
            "table_body": "| Method | Accuracy | F1-Score |\n|--------|----------|----------|\n| Ours | 95.2% | 0.94 |\n| Baseline | 87.3% | 0.85 |",
            "table_caption": ["Table 1: Performance Comparison"],
            "table_footnote": ["Results on test dataset"],
            "page_idx": 2  # Page number where this table appears
        },
        {
            "type": "equation",
            "latex": "P(d|q) = \\frac{P(q|d) \\cdot P(d)}{P(q)}",
            "text": "Document relevance probability formula",
            "page_idx": 3  # Page number where this equation appears
        },
        {
            "type": "text",
            "text": "In conclusion, our method demonstrates superior performance across all metrics.",
            "page_idx": 4  # Page number where this content appears
        }
    ]

    # Insert the content list directly
    await rag.insert_content_list(
        content_list=content_list,
        file_path="research_paper.pdf",  # Reference file name for citation
        split_by_character=None,         # Optional text splitting
        split_by_character_only=False,   # Optional text splitting mode
        doc_id=None,                     # Optional custom document ID (will be auto-generated if not provided)
        display_stats=True               # Show content statistics
    )

    # Query the inserted content
    result = await rag.aquery(
        "What are the key findings and performance metrics mentioned in the research?",
        mode="hybrid"
    )
    print("Query result:", result)

    # You can also insert multiple content lists with different document IDs
    another_content_list = [
        {
            "type": "text",
            "text": "This is content from another document.",
            "page_idx": 0  # Page number where this content appears
        },
        {
            "type": "table",
            "table_body": "| Feature | Value |\n|---------|-------|\n| Speed | Fast |\n| Accuracy | High |",
            "table_caption": ["Feature Comparison"],
            "page_idx": 1  # Page number where this table appears
        }
    ]

    await rag.insert_content_list(
        content_list=another_content_list,
        file_path="another_document.pdf",
        doc_id="custom-doc-id-123"  # Custom document ID
    )

if __name__ == "__main__":
    asyncio.run(insert_content_list_example())
```

**Content List Format:**

The `content_list` should follow the standard format with each item being a dictionary containing:

- **Text content**: `{"type": "text", "text": "content text", "page_idx": 0}`
- **Image content**: `{"type": "image", "img_path": "/absolute/path/to/image.jpg", "image_caption": ["caption"], "image_footnote": ["note"], "page_idx": 1}`
- **Table content**: `{"type": "table", "table_body": "markdown table", "table_caption": ["caption"], "table_footnote": ["note"], "page_idx": 2}`
- **Equation content**: `{"type": "equation", "latex": "LaTeX formula", "text": "description", "page_idx": 3}`
- **Generic content**: `{"type": "custom_type", "content": "any content", "page_idx": 4}`
- **Audio content**: `{"type": "audio", "audio_path": "/absolute/path/to/talk.wav", "audio_caption": ["caption"], "page_idx": 5}` — transcribed locally with faster-whisper (timestamped segments); long recordings are windowed into ordered chunks
- **Video content**: `{"type": "video", "video_path": "/absolute/path/to/demo.mp4", "video_caption": ["caption"], "page_idx": 6}` — SceneDetect scene boundaries + keyframe VLM description + audio-track transcription, merged by timestamp

**Important Notes:**
- **`img_path`**: Must be an absolute path to the image file (e.g., `/home/user/images/chart.jpg` or `C:\Users\user\images\chart.jpg`)
- **`page_idx`**: Represents the page number where the content appears in the original document (0-based indexing)
- **Text chunk pages**: text blocks are joined before LightRAG splits them; once LightRAG has processed the document, each of its text chunks is annotated with the `page_idx` of the blocks it came from (plus `page_idx_end` when it spans pages). Positions come from replaying LightRAG's default chunker, never from searching for chunk text: `split_by_character` pieces are placed with any tokenizer, token windows only with a tokenizer that decodes losslessly position by position (LightRAG's default tiktoken does). A chunk is left without `page_idx` rather than guessed when that is not possible: a custom `chunking_func` or other tokenizer, a document already known to LightRAG (a re-insert or failed retry), one queued behind another insert and chunked by that insert's pipeline run with different `split_by_character` settings, or, through `process_document_complete_lightrag_api` (which does not wait for LightRAG), one still queued, or a chunk whose text recurs on other pages. LightRAG stores one record per distinct chunk text, so a chunk shared by several documents carries the pages of the document its `full_doc_id` names (on backends that merge updates, such as MongoDB, a chunk re-stored by a document that was not annotated can keep the previous document's pages). The fields are extra keys on the `text_chunks` record: kept by the default JSON KV storage and other schemaless backends, but fixed-schema backends such as PostgreSQL may drop them
- **Content ordering**: Items are processed in the order they appear in the list

This method is particularly useful when:
- You have content from external parsers (non-MinerU/Docling)
- You want to process programmatically generated content
- You need to insert content from multiple sources into a single knowledge base
- You have cached parsing results that you want to reuse

#### 8. Re-processing Multimodal Content After a Storage Backend Change

By default, `process_document_complete` and `insert_content_list` skip multimodal
(image/table/equation) processing for a document that is already marked as fully
processed, to avoid redundant LLM calls. If you switch to a **new** graph/vector
storage backend (e.g. migrating from the default file-based storage to Neo4j),
the new backend will not yet contain the multimodal entities/relations that were
written to the old one — see [#154](https://github.com/HKUDS/RAG-Anything/issues/154).

Pass `force_multimodal_reprocess=True` to explicitly re-run multimodal processing
and re-populate the current storage backend for an already-processed document:

```python
# Re-run multimodal processing so the current storage backend has the
# image/table/equation entities and relations (default is False, i.e. no change
# to existing behavior for documents that have not switched backends)
await rag.process_document_complete(
    file_path="path/to/your/document.pdf",
    doc_id="doc-already-processed-id",
    force_multimodal_reprocess=True,
)

# Same flag is available for direct content list insertion
await rag.insert_content_list(
    content_list=content_list,
    doc_id="doc-already-processed-id",
    force_multimodal_reprocess=True,
)
```

This flag only affects multimodal content; it is opt-in and defaults to `False`,
so existing behavior is unchanged unless you explicitly set it.

---

## 🛠️ Examples

*Practical Implementation Demos*

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/212257455-13e3e01e-d6a6-45dc-bb92-3ab87b12dfc1.gif" width="300">
</div>

The `examples/` directory contains comprehensive usage examples:

- **`raganything_example.py`**: End-to-end document processing with MinerU
- **`modalprocessors_example.py`**: Direct multimodal content processing
- **`office_document_test.py`**: Office document parsing test with MinerU (no API key required)
- **`image_format_test.py`**: Image format parsing test with MinerU (no API key required)
- **`text_format_test.py`**: Text format parsing test with MinerU (no API key required)

**Run examples:**

```bash
# End-to-end processing with parser selection
python examples/raganything_example.py path/to/document.pdf --api-key YOUR_API_KEY --parser mineru

# Direct modal processing
python examples/modalprocessors_example.py --api-key YOUR_API_KEY

# Office document parsing test (MinerU only)
python examples/office_document_test.py --file path/to/document.docx

# Image format parsing test (MinerU only)
python examples/image_format_test.py --file path/to/image.bmp

# Text format parsing test (MinerU only)
python examples/text_format_test.py --file path/to/document.md

# Check LibreOffice installation
python examples/office_document_test.py --check-libreoffice --file dummy

# Check PIL/Pillow installation
python examples/image_format_test.py --check-pillow --file dummy

# Check ReportLab installation
python examples/text_format_test.py --check-reportlab --file dummy
```

---

## 🔧 Configuration

*System Optimization Parameters*

### Environment Variables

Create a `.env` file (refer to `.env.example`):

```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=your_base_url  # Optional
OUTPUT_DIR=./output             # Default output directory for parsed documents
PARSER=mineru                   # Parser selection: mineru, docling, or paddleocr
PARSE_METHOD=auto              # Parse method: auto, ocr, or txt
```

**Note:** For backward compatibility, legacy environment variable names are still supported:
- `MINERU_PARSE_METHOD` is deprecated, please use `PARSE_METHOD`

> **Note**: API keys are only required for full RAG processing with LLM integration. The parsing test files (`office_document_test.py` and `image_format_test.py`) only test parser functionality and do not require API keys.

### Troubleshooting and multimodal checklist

- See **[docs/multimodal_rag_failure_modes.md](docs/multimodal_rag_failure_modes.md)** for a short checklist of common pipeline issues (OCR, tables, retrieval bias, debugging tips). Related: [#207](https://github.com/HKUDS/RAG-Anything/issues/207), [#213](https://github.com/HKUDS/RAG-Anything/issues/213).

### Public media URLs (CDN / object storage)

When ingestion runs on a server but your UI or another service needs **HTTPS** (or S3-style) links to figures, set:

```bash
# Base URL for assets (no trailing slash required)
RAGANYTHING_PUBLIC_ASSET_BASE_URL=https://my-bucket.s3.us-east-1.amazonaws.com/prefix
# Filesystem root that should be stripped from absolute paths under that tree
RAGANYTHING_PUBLIC_ASSET_STRIP_PREFIX=/var/rag/output
```

After parsing, each non-empty `img_path`, `table_img_path`, or `equation_img_path` may gain a sibling field `*_public_url` while the original path stays on disk for local processing. See [#272](https://github.com/HKUDS/RAG-Anything/issues/272).

> **Scope today**: this mapping runs in the MinerU parser path only. Other parsers (e.g. Docling) keep working but will not produce `*_public_url` fields until the helper is wired into their content_list post-processing as well. If only one of the two env vars is set, RAG-Anything logs a warning and skips URL attachment.

### Parser Configuration

The parser is chosen when `RAGAnything` is created: set `RAGAnythingConfig(parser=...)` to `"mineru"` (default), `"docling"`, or `"paddleocr"`, or use the `PARSER` environment variable (command-line tools: `--parser`). It cannot be switched per `process_document_complete()` call.

RAGAnything now supports multiple parsers, each with specific advantages:

#### MinerU Parser
- Supports PDF, images, Office documents, and more formats
- Powerful OCR and table extraction capabilities
- GPU acceleration support

#### Docling Parser
- Optimized for Office documents and HTML files
- The only built-in parser that accepts HTML (.html, .htm, .xhtml); MinerU and PaddleOCR reject it (see [Document Formats](#document-formats))
- Better document structure preservation
- Native support for multiple Office formats

Install Docling (not included in `raganything[all]`):

```bash
pip install docling
```

#### PaddleOCR Parser
- OCR-focused parser for images and PDFs
- Produces text blocks compatible with existing `content_list` processing
- Supports optional Office/TXT/MD parsing by converting to PDF first

Install PaddleOCR parser extras:

```bash
pip install -e ".[paddleocr]"
# or
uv sync --extra paddleocr
```

> **Note**: PaddleOCR also requires `paddlepaddle` (CPU/GPU package varies by platform). Install it with the official guide: https://www.paddlepaddle.org.cn/install/quick

### MinerU Configuration

```bash
# MinerU 3.x uses command-line parameters instead of config files (MinerU 4.x is not supported)
# Check available options:
mineru --help

# Common configurations:
mineru -p input.pdf -o output_dir -m auto    # Automatic parsing mode
mineru -p input.pdf -o output_dir -m ocr     # OCR-focused parsing
mineru -p input.pdf -o output_dir -b pipeline --device cuda  # GPU acceleration
```

You can also configure parsing through RAGAnything parameters:

```python
# The parser comes from RAGAnythingConfig (or PARSER), not from process_document_complete()
config = RAGAnythingConfig(parser="mineru")  # "mineru", "docling", or "paddleocr"
rag = RAGAnything(config=config, llm_model_func=llm_model_func, embedding_func=embedding_func)

# Basic parsing configuration
await rag.process_document_complete(
    file_path="document.pdf",
    output_dir="./output/",
    parse_method="auto"           # or "ocr", "txt"
)

# Advanced parsing configuration with special parameters
await rag.process_document_complete(
    file_path="document.pdf",
    output_dir="./output/",
    parse_method="auto",          # Parsing method: "auto", "ocr", "txt"

    # MinerU special parameters - all supported kwargs:
    lang="ch",                   # Document language for OCR optimization (e.g., "ch", "en", "ja")
    device="cuda:0",             # Inference device: "cpu", "cuda", "cuda:0", "npu", "mps"
    start_page=0,                # Starting page number (0-based, for PDF)
    end_page=10,                 # Ending page number (0-based, for PDF)
    formula=True,                # Enable formula parsing
    table=True,                  # Enable table parsing
    backend="pipeline",          # Parsing backend: pipeline|hybrid-auto-engine|hybrid-http-client|vlm-auto-engine|vlm-http-client.
    source="huggingface",        # Model source: "huggingface", "modelscope", "local"
    # vlm_url="http://127.0.0.1:3000" # Service address when using backend=vlm-http-client
    include_layout_blocks=False,  # Keep MinerU v2 headers, footers, and page numbers only when needed

    # Standard RAGAnything parameters
    display_stats=True,          # Display content statistics
    split_by_character=None,     # Optional character to split text by
    doc_id=None                  # Optional document ID
)
```

> **Note**: MinerU 2.0 no longer uses the `magic-pdf.json` configuration file. All settings are now passed as command-line parameters or function arguments. RAG-Anything supports multiple document parsers, including MinerU, Docling, and PaddleOCR.

#### MinerU 3.x Content List v2

RAG-Anything natively reads MinerU 3.x `*_content_list_v2.json` and `content_list_v2.json` outputs when they are available, ahead of the legacy flat content list. Document-scoped output directories are preferred over shared roots; within one bundle, v2 is preferred over legacy and exact stem filenames are preferred over generic filenames. The v2 page-grouped representation is converted to the existing flat `content_list` contract while preserving page indexes, bounding boxes, anchors, heading levels, visual metadata, captions, and footnotes. Algorithm blocks use the compatible `type="code", sub_type="algorithm"` representation.

Charts with an `img_path` use the image processor for ingestion and multimodal queries, including their captions, footnotes and extracted text. Charts without an image keep generic text processing. The original parsed content and document IDs stay unchanged. To update previously indexed charts, re-ingest with `force_multimodal_reprocess=True`.

Page headers, footers, page numbers, aside text, and page footnotes are excluded by default so repeated layout artifacts do not enter semantic retrieval or graph construction. Set `include_layout_blocks=True` only when those blocks are meaningful for your use case:

```python
await rag.process_document_complete(
    file_path="document.pdf",
    include_layout_blocks=True,
)
```

MinerU marks v2 as an evolving output schema. RAG-Anything skips unknown future block types with a warning, but falls back to the legacy content list when a v2 file is malformed, structurally invalid, or yields no supported semantic blocks and a legacy file is present.

### Processing Requirements

Different content types require specific optional dependencies:

- **Office Documents** (.doc, .docx, .ppt, .pptx, .xls, .xlsx): Install [LibreOffice](https://www.libreoffice.org/download/download/)
- **Extended Image Formats** (.bmp, .tiff, .gif, .webp): Install with `pip install raganything[image]`
- **Audio** (.mp3, .wav, .flac, .m4a, .ogg): Install with `pip install raganything[audio]` and set `enable_audio_processing=True` (env `ENABLE_AUDIO_PROCESSING`); local transcription via faster-whisper, model picked by `WHISPER_MODEL` (default `base`; for CJK audio use `WHISPER_MODEL=medium` or larger and set `WHISPER_LANGUAGE`, e.g. `zh` — smaller models garble non-Latin proper nouns)
- **Video** (.mp4, .mov, .webm, .avi, .mkv): Install with `pip install raganything[video]` (also needs ffmpeg on PATH) and set `enable_video_processing=True` (env `ENABLE_VIDEO_PROCESSING`); dual-channel: SceneDetect + keyframe VLM description + audio-track transcription. Media files can be passed directly to `process_document_complete()` / folder batches, or inserted as content-list items
- **Text Files** (.txt, .md): Install with `pip install raganything[text]`. Parsed directly (no PDF/OCR round trip); markdown image references (`![alt](path)`) that point to local image files (an image extension and image file contents) become image blocks and flow through the same multimodal pipeline as images extracted from PDFs — URLs, missing files and other files stay as plain text. Referenced images are sent to the configured vision model wherever they are on disk, so only ingest Markdown whose image references you trust
- **PaddleOCR Parser** (`parser="paddleocr"`): Install with `pip install raganything[paddleocr]`, then install `paddlepaddle` for your platform

> **📋 Quick Install**: Use `pip install raganything[all]` to enable all format support (Python dependencies only - LibreOffice still needs separate installation, and HTML needs Docling: `pip install docling`)

---

## 🧪 Supported Content Types

### Document Formats

- **PDFs** - Research papers, reports, presentations
- **Office Documents** - DOC, DOCX, PPT, PPTX, XLS, XLSX
- **Images** - JPG, PNG, BMP, TIFF, GIF, WebP
- **Text Files** - TXT, MD
- **HTML** - HTML, HTM, XHTML (Docling parser only: install it with `pip install docling` and select it with `RAGAnythingConfig(parser="docling")` or `PARSER=docling`). MinerU and PaddleOCR raise an error for these files instead of converting them to PDF with LibreOffice, because that conversion fetches remote URLs and embeds local files referenced by the page. HTML is recognised by file extension: HTML content saved under an Office extension (for example `.doc` or `.xls`) is still converted by LibreOffice

### Multimodal Elements

- **Images** - Photographs, diagrams, charts, screenshots
- **Tables** - Data tables, comparison charts, statistical summaries
- **Equations** - Mathematical formulas in LaTeX format
- **Generic Content** - Custom content types via extensible processors

*For installation of format-specific dependencies, see the [Configuration](#-configuration) section.*

---

## 📖 Citation

This fork has no paper. The framework it builds on is described in the original paper; if you use it in research, please cite that paper:

```bibtex
@misc{guo2025raganythingallinoneragframework,
      title={RAG-Anything: All-in-One RAG Framework},
      author={Zirui Guo and Xubin Ren and Lingrui Xu and Jiahao Zhang and Chao Huang},
      year={2025},
      eprint={2510.12323},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.12323},
}
```

If you also want to reference this fork, cite it as a software repository:

```bibtex
@software{raganything_fork_technique,
      title={RAG-Anything (fork technique): Outlook mail and large technical PDF indexing},
      author={{Willtahiti}},
      url={https://github.com/Willtahiti/RAG-Anything},
      note={Modified fork of HKUDS/RAG-Anything (MIT License, Copyright (c) 2025 Data Intelligence Lab@HKU)},
}
```

---

## 📜 Fork notice and attribution

- **Original work:** [RAG-Anything](https://github.com/HKUDS/RAG-Anything), Copyright (c) 2025 Data Intelligence Lab@HKU, MIT License. The upstream license and copyright notice are kept unchanged in [LICENSE](LICENSE) and must stay with every copy or substantial portion of this software; see also [NOTICE](NOTICE).
- **Modifications:** the additions listed above were made in this fork and are provided under the same MIT License. The full change history is in the git log; the upstream history is preserved.
- **Not endorsed:** the original authors did not review or endorse these modifications and are not responsible for them. Report issues with this fork here, not upstream, unless you can reproduce them on the unmodified upstream code.
- **Citing:** if you use RAG-Anything in research, please cite the original paper (see [Citation](#-citation)).

- Names, logos and project links of the original project remain those of the upstream project; this README no longer reuses its logo, badges or community links.

---

## 🔗 Original project and ecosystem

*Upstream credits*

- [HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything) — the original framework (Data Intelligence Lab@HKU); its contributors are listed [on GitHub](https://github.com/HKUDS/RAG-Anything/graphs/contributors). Thank you to everyone who built it.
- [LightRAG](https://github.com/HKUDS/LightRAG) — the retrieval engine underneath.
- Parsers used by the pipeline: MinerU, Docling, PaddleOCR.

---

## 🤝 Contributing and support

Issues and pull requests for **this fork** go to [Willtahiti/RAG-Anything](https://github.com/Willtahiti/RAG-Anything). For problems that also occur on the unmodified original code, please report them to the [original project](https://github.com/HKUDS/RAG-Anything/issues).

<div align="center">

**RAG-Anything (fork technique)** — built on the work of the Data Intelligence Lab@HKU · MIT License

</div>

