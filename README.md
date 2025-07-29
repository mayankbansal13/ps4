# 🤖 RAG Question & Answering System

A complete **Retrieval-Augmented Generation (RAG)** system for intelligent document Q&A with multi-format support, hybrid search, and a beautiful modern web interface.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![AI](https://img.shields.io/badge/AI-RAG%20System-purple)

## 🎯 Features

### 📄 Multi-Format Document Support
- **PDF** (including scanned PDFs with OCR)
- **DOCX** (Word documents with table extraction)
- **PPTX** (PowerPoint presentations)
- **TXT** (Plain text files)
- **CSV** (Comma-separated values)

### 🔍 Advanced Search & Retrieval
- **Hybrid Search**: Combines semantic and keyword search
- **Semantic Search**: Vector similarity using sentence transformers
- **Keyword Search**: Traditional text matching with scoring
- **Query Expansion**: Automatic query enhancement
- **Confidence Scoring**: Reliability metrics for answers

### 🧠 AI-Powered Reasoning
- **OpenAI Integration**: GPT-3.5/4 support with API key
- **HuggingFace Fallback**: Local models when OpenAI unavailable
- **Source Citation**: Every answer links to source documents
- **Hallucination Minimization**: Grounded responses only

### 💻 Modern Web Interface
- **Dark Theme**: Sci-fi aesthetic with neon highlights
- **Glassmorphism Design**: Beautiful modern UI components
- **Drag & Drop Upload**: Intuitive file management
- **Real-time Chat**: Interactive Q&A interface
- **Progress Tracking**: Visual feedback for operations
- **Mobile Responsive**: Works on all devices

## 🚀 Quick Start

### Prerequisites

```bash
# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler

# Python 3.8+
python --version
```

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-qa-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API (Optional)**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   # Or create a .env file with: OPENAI_API_KEY=your-api-key-here
   ```

### 🎮 Usage

#### Web Interface (Recommended)
```bash
python main.py --api
```
Then open: http://localhost:8000

#### Command Line Interface
```bash
# Ingest documents
python main.py --ingest ./documents
python main.py --ingest document.pdf

# Ask questions
python main.py --ask "What is the main topic discussed?"
python main.py --ask "Summarize key findings" --search-type hybrid

# System information
python main.py --info
```

## 📖 API Documentation

### Endpoints

- **`POST /api/ingest/file`** - Upload and process a document
- **`POST /api/ask`** - Ask a question and get AI answer
- **`POST /api/search`** - Search through documents
- **`GET /api/status`** - Get system status and statistics
- **`GET /api/history`** - Retrieve Q&A session history
- **`DELETE /api/documents`** - Clear all documents
- **`GET /health`** - Health check endpoint

### Example API Usage

```python
import requests

# Upload a document
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/ingest/file',
        files={'file': f}
    )

# Ask a question
response = requests.post(
    'http://localhost:8000/api/ask',
    json={
        'question': 'What are the key findings?',
        'search_type': 'hybrid'
    }
)
answer = response.json()
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI Web   │    │   Document      │    │   Vector Store  │
│   Interface     │────│   Processor     │────│   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │              ┌─────────────────┐               │
         │              │   Retrieval     │               │
         └──────────────│   System        │───────────────┘
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │   LLM           │
                        │   Integration   │
                        └─────────────────┘
```

### Core Components

1. **`config.py`** - Configuration and hyperparameters
2. **`document_processor.py`** - Multi-format document processing with OCR
3. **`vector_store.py`** - ChromaDB vector database management
4. **`retrieval_system.py`** - Hybrid search and retrieval
5. **`llm_integration.py`** - OpenAI/HuggingFace model integration
6. **`rag_system.py`** - Main orchestration layer
7. **`api.py`** - FastAPI web server and modern UI
8. **`main.py`** - CLI interface

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Document Processing
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
TOP_K_RETRIEVAL = 5
SIMILARITY_THRESHOLD = 0.7
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-3.5-turbo"
HUGGINGFACE_MODEL = "microsoft/DialoGPT-medium"
```

## 🔧 Advanced Features

### OCR for Scanned Documents
Automatically detects and processes scanned PDFs using Tesseract OCR:
```python
# OCR is automatically triggered for low-text PDFs
result = rag_system.ingest_document("scanned_document.pdf")
```

### Confidence Scoring
Every answer includes a confidence score based on:
- Retrieval similarity scores
- Number of relevant sources
- Answer length and structure
- Source citation presence

### Session History
Maintains conversation context:
```python
# Get session history
history = rag_system.get_session_history()

# Clear history
rag_system.clear_session_history()
```

## 🎨 UI Screenshots

### Main Interface
The modern dark theme with glassmorphism effects provides an intuitive experience:
- **Document Upload**: Drag & drop zone with progress tracking
- **Chat Interface**: Real-time Q&A with source citations
- **Search Panel**: Advanced document search capabilities
- **System Stats**: Live system monitoring

### Features Showcase
- ✨ **Smooth Animations**: Fade-in effects and hover transitions
- 🎨 **Neon Highlights**: Purple/blue gradient accents
- 📱 **Responsive Design**: Works on desktop and mobile
- 🔍 **Interactive Elements**: Hover effects and visual feedback

## 🐛 Troubleshooting

### Common Issues

1. **OCR Not Working**
   ```bash
   # Install Tesseract
   sudo apt-get install tesseract-ocr
   # Or check PATH configuration
   ```

2. **PDF Processing Errors**
   ```bash
   # Install poppler-utils
   sudo apt-get install poppler-utils
   ```

3. **Model Loading Issues**
   ```bash
   # Clear cache and retry
   rm -rf ~/.cache/huggingface/
   ```

4. **Port Already in Use**
   ```bash
   python main.py --api --port 8001
   ```

### Performance Optimization

- **GPU Support**: Automatically detects and uses CUDA if available
- **Chunking Strategy**: Adjust `CHUNK_SIZE` for your document types
- **Model Selection**: Use OpenAI API for faster responses
- **Batch Processing**: Ingest multiple documents simultaneously

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Create a GitHub issue for bugs or feature requests
- **API Docs**: Visit `/api/docs` when running the server

## 🔮 Future Enhancements

- [ ] Support for more file formats (EPUB, RTF, etc.)
- [ ] Multi-language document support
- [ ] Advanced query understanding with NER
- [ ] Document summarization capabilities
- [ ] User authentication and multi-tenancy
- [ ] Integration with cloud storage (AWS S3, Google Drive)
- [ ] Advanced analytics and usage metrics

---

**Built with ❤️ for intelligent document analysis and Q&A**