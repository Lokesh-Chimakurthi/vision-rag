# Vision-RAG

A Vision RAG (Retrieval Augmented Generation) system that combines Jina CLIP V2 embeddings with ColPali - ColQwen reranking using Qdrant vector database for efficient image retrieval and Gemini for response generation.

## Features

- PDF to image conversion with size optimization
- Dual embedding system:
  - JINA CLIP v2 for dense image embeddings
  - ColQwen for mean-pooling embeddings and reranking
- Qdrant vector database integration with vision hybrid search
- Gemini 1.5 Flash for response generation
- Batch processing support
- Built-in error handling and logging

## Prerequisites

- Python 3.10
- Qdrant API key
- Google Gemini API key

## Installation

1. Set up environment variables:
```bash
export QDRANT_API=your_qdrant_api_key
export GEMINI_API_KEY=your_gemini_api_key
```

2. Install dependencies:
You can use either uv or pip to install the dependencies

Using uv:
```bash
uv venv
uv sync
```

Using pip:
```bash
pip install -r requirements.txt
```

## Configuration

Key configuration parameters in `config` class:

```python
collection_name: str = "vision_rag"
image_model: str = "jinaai/jina-clip-v2"
gemini_model: str = "gemini-1.5-flash-002"
colpali_model: str = "vidore/colqwen2-v1.0"
cpu_threads: int = 2
batch_size: int = 2
```

Note: Code is written to run on CPU. If you have GPU or if you use mac, change the code accordingly.
## Usage

1. Place your PDF files in the current working directory or any subdirectory.
2. Add the PDF path in the main script and run it:
```bash
python main.py
```
3. The script will process the PDFs, convert them to images, and index them in Qdrant.
4. The script will then perform a Qdrant search, retrieve points, and send the user prompt and retrieved images to Gemini Flash.
5. Based on the retrieved information, Gemini's response is printed.

### Features Details

#### PDF Processing
- Converts PDF pages to images
- Maintains aspect ratio
- Optimizes image size

#### Vector Search
- Uses dual embedding system
- JINA CLIP V2 for primary image embeddings
- ColQwen for reranking
- Supports batch processing

#### Chat Interface
- System instruction customization
- Chat history management
- Error handling
- Clear chat functionality

## Error Handling

The system includes custom exceptions:
- `ProcessingError`: For document processing issues
- `ChatError`: For chat-related issues

## Logging

Comprehensive logging system with:
- INFO level for general operations
- WARNING level for non-critical issues
- ERROR level for critical failures

## Performance Considerations

- Uses parallel tokenization
- Configurable CPU threads for gpu poor.
- Batch processing for embeddings
- Efficient memory management
