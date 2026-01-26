# Memory

A memory system for AI agents. Store, search, and manage memories with support for text, images, and documents.

## Features

- **Semantic Search**: Find memories by meaning, not just keywords
- **Auto-Chunking**: Intelligent chunking for emails, chats, and documents
- **Multi-Modal**: Support for text, images (with auto-captioning), and PDFs
- **Web UI**: Streamlit-based interface for managing memories
- **REST API**: FastAPI backend for programmatic access

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd mem

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Create `.env` file

```env
OPENAI_API_KEY=sk-your-key-here
API_KEY=your-api-key-for-rest-api
```

### Run the Web UI

```bash
streamlit run web/app.py
```

Open http://localhost:8501

### Run the API Server

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for API documentation.

## Usage

### Python Client

```python
from app.client import Memory

# Initialize
m = Memory()

# Add text memories
m.add("User prefers dark mode", user_id="alice")
m.add("Meeting scheduled for Friday at 3pm", user_id="alice", metadata={"type": "calendar"})

# Search memories
results = m.search("user preferences", user_id="alice")
for r in results:
    print(f"{r['score']:.2f}: {r['content']}")

# Add an image (auto-generates caption)
m.add_image(image_path="screenshot.png", user_id="alice")

# Search by image similarity
similar = m.search_image(image_path="query.png", user_id="alice")

# Add a PDF document
m.add_pdf(file_path="report.pdf", user_id="alice")

# Get all memories
memories = m.get_all(user_id="alice")

# Update a memory
m.update(memory_id="abc123", content="Updated content")

# Delete a memory
m.delete(memory_id="abc123")
```

### Auto-Detection Chunking

The system automatically detects content type and applies appropriate chunking:

```python
# Email - strips quotes, signatures, applies sentence-based chunking
m.add(email_content, content_type="email")  # or auto-detected

# Chat - groups messages into conversation windows
m.add(chat_log, content_type="chat")

# Document - paragraph-aware chunking with overlap
m.add(document_text, content_type="document")
```

## Project Structure

```
mem/
├── app/
│   ├── client.py           # Memory client (main interface)
│   ├── config.py           # Settings management
│   ├── main.py             # FastAPI application
│   ├── core/
│   │   ├── chunker.py      # Auto-detecting text chunker
│   │   ├── embeddings.py   # OpenAI embedding service
│   │   ├── pdf_parser.py   # PDF text extraction
│   │   └── vision.py       # Image captioning (GPT-4V)
│   ├── api/
│   │   └── routes/         # API endpoints
│   └── models/
│       └── schemas.py      # Pydantic models
├── web/
│   ├── app.py              # Streamlit entry point
│   └── components/         # UI components
├── tests/                  # Test suite
├── requirements.txt
└── README.md
```

## Development

### Setup

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=app --cov-report=html
```

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_client.py

# With verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Code Style

- Type hints required for all functions
- Keep components small and focused
- Write tests for new features

### Adding a New Content Type

1. Create parser in `app/core/` (e.g., `excel_parser.py`)
2. Add chunking strategy in `app/core/chunker.py` if needed
3. Add method to `app/client.py` (e.g., `add_excel()`)
4. Update web UI in `web/components/`
5. Write tests in `tests/`

## API Reference

### Memory Client Methods

| Method | Description |
|--------|-------------|
| `add(content, user_id, agent_id, metadata)` | Add text memory with auto-chunking |
| `add_image(image_path, image_bytes, user_id)` | Add image with auto-captioning |
| `add_pdf(file_path, file_bytes, user_id)` | Add PDF with text extraction |
| `search(query, user_id, limit)` | Semantic search |
| `search_image(image_path, image_bytes, user_id)` | Find similar images |
| `get(memory_id)` | Get single memory |
| `get_all(user_id, limit)` | Get all memories |
| `update(memory_id, content, metadata)` | Update memory |
| `delete(memory_id)` | Delete memory |
| `delete_all(user_id)` | Delete all memories for user |
| `count(user_id)` | Count memories |

### Content Types

| Type | Chunk Size | Features |
|------|-----------|----------|
| `email` | 1500 chars | Strips quotes, signatures |
| `chat` | 800 chars | Conversation windows, max 10 messages |
| `document` | 2000 chars | Paragraph-aware, 200 char overlap |
| `image` | N/A | GPT-4V captioning |
| `pdf` | 2000 chars | Page-aware extraction |

## Tech Stack

- **Vector DB**: ChromaDB
- **Embeddings**: OpenAI text-embedding-3-small
- **Vision**: GPT-4o-mini
- **API**: FastAPI
- **Web UI**: Streamlit
- **PDF**: pypdf

## License

MIT
