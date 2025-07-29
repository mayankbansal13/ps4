import os
import logging
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_system import RAGSystem
from config import Config

# Setup logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Question Answering System",
    description="A complete Retrieval-Augmented Generation system for document Q&A",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem()

# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    search_type: str = "hybrid"

class SearchRequest(BaseModel):
    query: str
    search_type: str = "hybrid"
    top_k: Optional[int] = None

class IngestResponse(BaseModel):
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float
    timestamp: str

# Ensure required directories exist
Config.ensure_directories()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = rag_system.get_system_status()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_info": status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Document ingestion endpoints
@app.post("/api/ingest/file", response_model=IngestResponse)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and ingest a single document"""
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in Config.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported types: {Config.SUPPORTED_EXTENSIONS}"
            )
        
        # Validate file size
        if file.size > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Save uploaded file
        file_path = os.path.join(Config.UPLOAD_DIR, file.filename)
        
        # Handle duplicate filenames
        counter = 1
        original_path = file_path
        while os.path.exists(file_path):
            name, ext = os.path.splitext(original_path)
            file_path = f"{name}_{counter}{ext}"
            counter += 1
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Ingest the document
        result = rag_system.ingest_document(file_path)
        
        if result["success"]:
            return IngestResponse(
                success=True,
                message=f"Successfully ingested {result['chunks_extracted']} chunks from {file.filename}",
                details=result
            )
        else:
            # Clean up failed upload
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return IngestResponse(
                success=False,
                message=f"Failed to ingest document: {result.get('error', 'Unknown error')}",
                details=result
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/ingest/folder")
async def ingest_folder(folder_path: str):
    """Ingest all supported documents from a folder"""
    try:
        if not os.path.exists(folder_path):
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_path}")
        
        result = rag_system.ingest_folder(folder_path)
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Successfully ingested {result['total_chunks']} chunks from {result['total_files']} files",
                "details": result
            }
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": f"Failed to ingest folder: {result.get('error', 'Unknown error')}",
                    "details": result
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Folder ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Question answering endpoint
@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an AI-generated answer"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        result = rag_system.ask_question(
            question=request.question,
            search_type=request.search_type
        )
        
        return QuestionResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            processing_time=result["processing_time"],
            timestamp=result["timestamp"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Search endpoint
@app.post("/api/search")
async def search_documents(request: SearchRequest):
    """Search for relevant document chunks"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        result = rag_system.search_documents(
            query=request.query,
            search_type=request.search_type,
            top_k=request.top_k
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# System management endpoints
@app.get("/api/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        return rag_system.get_system_status()
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/history")
async def get_session_history():
    """Get current session's Q&A history"""
    try:
        return {
            "history": rag_system.get_session_history(),
            "total_entries": len(rag_system.get_session_history())
        }
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/api/history")
async def clear_session_history():
    """Clear current session's Q&A history"""
    try:
        rag_system.clear_session_history()
        return {"message": "Session history cleared successfully"}
    except Exception as e:
        logger.error(f"History clearing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/api/documents")
async def clear_vector_store():
    """Clear all documents from the vector store"""
    try:
        result = rag_system.clear_vector_store()
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    except Exception as e:
        logger.error(f"Vector store clearing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/test")
async def test_system():
    """Run system diagnostics"""
    try:
        return rag_system.test_system()
    except Exception as e:
        logger.error(f"System test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Serve the main UI
@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
async def serve_ui():
    """Serve the main web interface"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Q&A System</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Custom animations and glassmorphism */
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 20px rgba(147, 51, 234, 0.3); }
            50% { box-shadow: 0 0 30px rgba(147, 51, 234, 0.6); }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .glass {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .glow-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            animation: glow 2s ease-in-out infinite;
        }
        
        .fade-in-up {
            animation: fadeInUp 0.6s ease-out;
        }
        
        .loading {
            animation: pulse 1.5s ease-in-out infinite;
        }
        
        .gradient-text {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .chat-bubble-user {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .chat-bubble-ai {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .file-drop-zone {
            transition: all 0.3s ease;
            border: 2px dashed rgba(147, 51, 234, 0.3);
        }
        
        .file-drop-zone.dragover {
            border-color: #667eea;
            background: rgba(147, 51, 234, 0.1);
            transform: scale(1.02);
        }
        
        .progress-bar {
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        }
    </style>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <!-- Header -->
    <header class="glass border-b border-purple-500/20 sticky top-0 z-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div class="flex items-center justify-between">
                <h1 class="text-2xl font-bold gradient-text">🤖 RAG Q&A System</h1>
                <div class="flex items-center space-x-4">
                    <button id="statusBtn" class="glass px-4 py-2 rounded-lg hover:bg-white/10 transition-all">
                        <span id="statusText">System Status</span>
                    </button>
                    <button id="clearBtn" class="glass px-4 py-2 rounded-lg hover:bg-red-500/20 transition-all text-red-400">
                        Clear All
                    </button>
                </div>
            </div>
        </div>
    </header>

    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Document Upload Section -->
            <div class="lg:col-span-1">
                <div class="glass rounded-xl p-6 fade-in-up">
                    <h2 class="text-xl font-semibold mb-4 flex items-center">
                        📄 Document Upload
                    </h2>
                    
                    <!-- File Drop Zone -->
                    <div id="dropZone" class="file-drop-zone rounded-lg p-8 text-center mb-4 cursor-pointer">
                        <div class="flex flex-col items-center">
                            <svg class="w-12 h-12 text-purple-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                            </svg>
                            <p class="text-lg font-medium mb-2">Drop files here or click to browse</p>
                            <p class="text-sm text-gray-400">PDF, DOCX, PPTX, TXT, CSV (max 100MB)</p>
                        </div>
                        <input type="file" id="fileInput" class="hidden" multiple accept=".pdf,.docx,.pptx,.txt,.csv">
                    </div>
                    
                    <!-- Upload Progress -->
                    <div id="uploadProgress" class="hidden mb-4">
                        <div class="bg-gray-700 rounded-full h-2 mb-2">
                            <div id="progressBar" class="progress-bar h-2 rounded-full" style="width: 0%"></div>
                        </div>
                        <p id="uploadStatus" class="text-sm text-gray-400">Uploading...</p>
                    </div>
                    
                    <!-- Uploaded Files List -->
                    <div id="uploadedFiles" class="space-y-2">
                        <!-- Files will be added here -->
                    </div>
                </div>
                
                <!-- System Stats -->
                <div class="glass rounded-xl p-6 mt-6 fade-in-up">
                    <h3 class="text-lg font-semibold mb-3">📊 System Stats</h3>
                    <div id="systemStats" class="space-y-3">
                        <div class="flex justify-between">
                            <span class="text-gray-400">Documents:</span>
                            <span id="docCount" class="text-purple-400">0</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Chunks:</span>
                            <span id="chunkCount" class="text-purple-400">0</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-400">Model:</span>
                            <span id="modelName" class="text-purple-400">Loading...</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Chat Interface -->
            <div class="lg:col-span-2">
                <div class="glass rounded-xl h-[600px] flex flex-col fade-in-up">
                    <!-- Chat Header -->
                    <div class="border-b border-purple-500/20 p-4">
                        <h2 class="text-xl font-semibold">💬 Ask Questions</h2>
                        <p class="text-sm text-gray-400">Ask questions about your uploaded documents</p>
                    </div>
                    
                    <!-- Chat Messages -->
                    <div id="chatMessages" class="flex-1 overflow-y-auto p-4 space-y-4">
                        <!-- Welcome message -->
                        <div class="chat-bubble-ai rounded-lg p-4">
                            <div class="flex items-start space-x-3">
                                <div class="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center">
                                    🤖
                                </div>
                                <div>
                                    <p class="text-sm text-gray-300">Welcome! Upload some documents and ask me questions about them. I can help you find information using advanced AI search and reasoning.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Chat Input -->
                    <div class="border-t border-purple-500/20 p-4">
                        <div class="flex space-x-3">
                            <input 
                                type="text" 
                                id="questionInput" 
                                placeholder="Ask a question about your documents..."
                                class="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
                            >
                            <select id="searchType" class="bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 focus:outline-none focus:border-purple-500">
                                <option value="hybrid">Hybrid</option>
                                <option value="semantic">Semantic</option>
                                <option value="keyword">Keyword</option>
                            </select>
                            <button 
                                id="askBtn" 
                                class="glow-button px-6 py-2 rounded-lg font-medium transition-all hover:scale-105"
                            >
                                Ask
                            </button>
                        </div>
                    </div>
                </div>
                
                <!-- Search Interface -->
                <div class="glass rounded-xl p-6 mt-6 fade-in-up">
                    <h3 class="text-lg font-semibold mb-3">🔍 Document Search</h3>
                    <div class="flex space-x-3 mb-4">
                        <input 
                            type="text" 
                            id="searchInput" 
                            placeholder="Search through documents..."
                            class="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 focus:outline-none focus:border-purple-500"
                        >
                        <button 
                            id="searchBtn" 
                            class="glow-button px-6 py-2 rounded-lg font-medium"
                        >
                            Search
                        </button>
                    </div>
                    <div id="searchResults" class="space-y-3 max-h-64 overflow-y-auto">
                        <!-- Search results will appear here -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Loading Overlay -->
    <div id="loadingOverlay" class="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 hidden">
        <div class="glass rounded-xl p-8 text-center">
            <div class="loading">
                <svg class="w-16 h-16 text-purple-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                </svg>
            </div>
            <p class="text-lg font-medium">Processing...</p>
        </div>
    </div>

    <script>
        // Global state
        let isUploading = false;
        let sessionHistory = [];

        // DOM elements
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const uploadProgress = document.getElementById('uploadProgress');
        const progressBar = document.getElementById('progressBar');
        const uploadStatus = document.getElementById('uploadStatus');
        const uploadedFiles = document.getElementById('uploadedFiles');
        const questionInput = document.getElementById('questionInput');
        const searchType = document.getElementById('searchType');
        const askBtn = document.getElementById('askBtn');
        const chatMessages = document.getElementById('chatMessages');
        const searchInput = document.getElementById('searchInput');
        const searchBtn = document.getElementById('searchBtn');
        const searchResults = document.getElementById('searchResults');
        const loadingOverlay = document.getElementById('loadingOverlay');
        const statusBtn = document.getElementById('statusBtn');
        const clearBtn = document.getElementById('clearBtn');

        // Initialize the app
        document.addEventListener('DOMContentLoaded', function() {
            setupEventListeners();
            updateSystemStats();
            loadHistory();
        });

        function setupEventListeners() {
            // File upload
            dropZone.addEventListener('click', () => fileInput.click());
            dropZone.addEventListener('dragover', handleDragOver);
            dropZone.addEventListener('dragleave', handleDragLeave);
            dropZone.addEventListener('drop', handleDrop);
            fileInput.addEventListener('change', handleFileSelect);

            // Chat
            askBtn.addEventListener('click', askQuestion);
            questionInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') askQuestion();
            });

            // Search
            searchBtn.addEventListener('click', searchDocuments);
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') searchDocuments();
            });

            // System controls
            statusBtn.addEventListener('click', showSystemStatus);
            clearBtn.addEventListener('click', clearAll);
        }

        function handleDragOver(e) {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }

        function handleDragLeave(e) {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }

        function handleDrop(e) {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = Array.from(e.dataTransfer.files);
            uploadFiles(files);
        }

        function handleFileSelect(e) {
            const files = Array.from(e.target.files);
            uploadFiles(files);
        }

        async function uploadFiles(files) {
            if (isUploading || files.length === 0) return;

            isUploading = true;
            uploadProgress.classList.remove('hidden');
            
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                const progress = ((i + 1) / files.length) * 100;
                
                progressBar.style.width = progress + '%';
                uploadStatus.textContent = `Uploading ${file.name}...`;
                
                try {
                    const result = await uploadFile(file);
                    addUploadedFile(file.name, result.success, result.details);
                } catch (error) {
                    addUploadedFile(file.name, false, { error: error.message });
                }
            }
            
            uploadProgress.classList.add('hidden');
            isUploading = false;
            updateSystemStats();
        }

        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/api/ingest/file', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }
            
            return await response.json();
        }

        function addUploadedFile(filename, success, details) {
            const fileElement = document.createElement('div');
            fileElement.className = `glass rounded-lg p-3 ${success ? 'border-green-500/30' : 'border-red-500/30'} fade-in-up`;
            
            const icon = success ? '✅' : '❌';
            const statusText = success ? `${details.chunks_extracted} chunks` : 'Failed';
            const statusColor = success ? 'text-green-400' : 'text-red-400';
            
            fileElement.innerHTML = `
                <div class="flex items-center justify-between">
                    <div class="flex items-center space-x-3">
                        <span class="text-lg">${icon}</span>
                        <div>
                            <p class="font-medium truncate">${filename}</p>
                            <p class="text-sm ${statusColor}">${statusText}</p>
                        </div>
                    </div>
                </div>
            `;
            
            uploadedFiles.appendChild(fileElement);
        }

        async function askQuestion() {
            const question = questionInput.value.trim();
            if (!question) return;

            addChatMessage(question, 'user');
            questionInput.value = '';
            showLoading();

            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: question,
                        search_type: searchType.value
                    })
                });

                const result = await response.json();
                
                if (response.ok) {
                    addChatMessage(result.answer, 'ai', result);
                    sessionHistory.push(result);
                } else {
                    addChatMessage(`Error: ${result.detail}`, 'ai', null, true);
                }
            } catch (error) {
                addChatMessage(`Error: ${error.message}`, 'ai', null, true);
            } finally {
                hideLoading();
            }
        }

        function addChatMessage(message, type, metadata = null, isError = false) {
            const messageElement = document.createElement('div');
            messageElement.className = `${type === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai'} rounded-lg p-4 fade-in-up`;
            
            if (isError) {
                messageElement.classList.add('border-red-500/30');
            }
            
            const avatar = type === 'user' ? '👤' : '🤖';
            const timestamp = new Date().toLocaleTimeString();
            
            let sourcesHtml = '';
            if (metadata && metadata.sources && metadata.sources.length > 0) {
                sourcesHtml = `
                    <div class="mt-3 pt-3 border-t border-white/10">
                        <p class="text-xs text-gray-400 mb-2">Sources:</p>
                        <div class="space-y-2">
                            ${metadata.sources.map(source => `
                                <div class="text-xs bg-white/5 rounded p-2">
                                    <div class="flex justify-between items-center mb-1">
                                        <span class="font-medium">${source.filename}</span>
                                        <span class="text-purple-400">Score: ${Object.values(source.scores)[0] || 'N/A'}</span>
                                    </div>
                                    <p class="text-gray-400 truncate">${source.content_preview}</p>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            }
            
            let confidenceHtml = '';
            if (metadata && metadata.confidence !== undefined) {
                const confidenceColor = metadata.confidence > 0.7 ? 'text-green-400' : 
                                       metadata.confidence > 0.5 ? 'text-yellow-400' : 'text-red-400';
                confidenceHtml = `
                    <div class="mt-2">
                        <span class="text-xs ${confidenceColor}">
                            Confidence: ${(metadata.confidence * 100).toFixed(1)}%
                        </span>
                    </div>
                `;
            }
            
            messageElement.innerHTML = `
                <div class="flex items-start space-x-3">
                    <div class="w-8 h-8 ${type === 'user' ? 'bg-blue-500' : 'bg-purple-500'} rounded-full flex items-center justify-center flex-shrink-0">
                        ${avatar}
                    </div>
                    <div class="flex-1 min-w-0">
                        <div class="flex items-center justify-between mb-1">
                            <span class="text-sm font-medium">${type === 'user' ? 'You' : 'AI Assistant'}</span>
                            <span class="text-xs text-gray-400">${timestamp}</span>
                        </div>
                        <p class="text-sm whitespace-pre-wrap">${message}</p>
                        ${confidenceHtml}
                        ${sourcesHtml}
                    </div>
                </div>
            `;
            
            chatMessages.appendChild(messageElement);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        async function searchDocuments() {
            const query = searchInput.value.trim();
            if (!query) return;

            showLoading();
            searchResults.innerHTML = '';

            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        search_type: 'hybrid',
                        top_k: 5
                    })
                });

                const result = await response.json();
                
                if (response.ok && result.results.length > 0) {
                    result.results.forEach(item => {
                        const resultElement = document.createElement('div');
                        resultElement.className = 'glass rounded-lg p-3 fade-in-up';
                        
                        const score = Object.values(item.scores)[0] || 0;
                        
                        resultElement.innerHTML = `
                            <div class="flex justify-between items-start mb-2">
                                <span class="font-medium text-purple-400">${item.filename}</span>
                                <span class="text-xs text-gray-400">Score: ${score.toFixed(3)}</span>
                            </div>
                            <p class="text-sm text-gray-300">${item.content.substring(0, 200)}...</p>
                        `;
                        
                        searchResults.appendChild(resultElement);
                    });
                } else {
                    searchResults.innerHTML = '<p class="text-gray-400 text-center py-4">No results found</p>';
                }
            } catch (error) {
                searchResults.innerHTML = `<p class="text-red-400 text-center py-4">Error: ${error.message}</p>`;
            } finally {
                hideLoading();
            }
        }

        async function updateSystemStats() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                if (response.ok) {
                    document.getElementById('docCount').textContent = status.vector_store.unique_sources || 0;
                    document.getElementById('chunkCount').textContent = status.vector_store.total_chunks || 0;
                    document.getElementById('modelName').textContent = 
                        status.language_models.openai_available ? 'OpenAI' : 'HuggingFace';
                }
            } catch (error) {
                console.error('Failed to update system stats:', error);
            }
        }

        async function showSystemStatus() {
            showLoading();
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                const statusText = JSON.stringify(status, null, 2);
                addChatMessage(`System Status:\n\n${statusText}`, 'ai');
            } catch (error) {
                addChatMessage(`Error getting system status: ${error.message}`, 'ai', null, true);
            } finally {
                hideLoading();
            }
        }

        async function clearAll() {
            if (!confirm('This will clear all documents and chat history. Are you sure?')) return;
            
            showLoading();
            try {
                await fetch('/api/documents', { method: 'DELETE' });
                await fetch('/api/history', { method: 'DELETE' });
                
                chatMessages.innerHTML = `
                    <div class="chat-bubble-ai rounded-lg p-4">
                        <div class="flex items-start space-x-3">
                            <div class="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center">
                                🤖
                            </div>
                            <div>
                                <p class="text-sm text-gray-300">All documents and chat history have been cleared. You can upload new documents to get started.</p>
                            </div>
                        </div>
                    </div>
                `;
                
                uploadedFiles.innerHTML = '';
                searchResults.innerHTML = '';
                sessionHistory = [];
                updateSystemStats();
            } catch (error) {
                addChatMessage(`Error clearing data: ${error.message}`, 'ai', null, true);
            } finally {
                hideLoading();
            }
        }

        async function loadHistory() {
            try {
                const response = await fetch('/api/history');
                const data = await response.json();
                sessionHistory = data.history || [];
                
                // Add recent history to chat
                sessionHistory.slice(-5).forEach(item => {
                    addChatMessage(item.question, 'user');
                    addChatMessage(item.answer, 'ai', item);
                });
            } catch (error) {
                console.error('Failed to load history:', error);
            }
        }

        function showLoading() {
            loadingOverlay.classList.remove('hidden');
        }

        function hideLoading() {
            loadingOverlay.classList.add('hidden');
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

# Run the server
if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting RAG Q&A System on {Config.HOST}:{Config.PORT}")
    uvicorn.run(
        "api:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,
        log_level=Config.LOG_LEVEL.lower()
    )