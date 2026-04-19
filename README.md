# magic-embedder

Modular text and image embedding library for RAG applications.

## Features

| Strategy | Class | Description |
|----------|-------|-------------|
| OpenAI Text | `OpenAITextEmbedder` | text-embedding-3-small/large, ada-002 |
| SentenceTransformer | `SentenceTransformerEmbedder` | all-MiniLM-L6-v2 and other models |
| CLIP Image | `CLIPImageEmbedder` | openai/clip-vit-base-patch32 via transformers |

## Installation

```bash
pip install sentence-transformers openai torch torchvision Pillow transformers
```

## Quick Start

```python
from magic_embedder.strategies import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
result = embedder.embed(["Hello world", "This is a test"])
print(f"Dimension: {result.dimension}, Count: {len(result.embeddings)}")
```

## Module Structure

```
magic_embedder/
    __init__.py
    run.py
    core/           # BaseEmbedder, EmbeddingResult
    strategies/
        text/       # OpenAI, SentenceTransformer
        image/      # CLIP (transformers)
    utils/
```

## CLI

```bash
python -m magic_embedder.run --input input.json --strategy sentence_transformer --model all-MiniLM-L6-v2
```
