#!/usr/bin/env python3
"""
Main CLI interface for the RAG Question & Answering System
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import uvicorn

from rag_system import RAGSystem
from config import Config

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(Config.LOG_DIR, 'rag_system.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )

def ingest_documents(path: str):
    """Ingest documents from a file or folder"""
    print("🚀 Initializing RAG System...")
    rag_system = RAGSystem()
    
    path = Path(path)
    
    if not path.exists():
        print(f"❌ Error: Path '{path}' does not exist.")
        return False
    
    print(f"📄 Processing: {path}")
    
    if path.is_file():
        # Single file ingestion
        result = rag_system.ingest_document(str(path))
        
        if result["success"]:
            print(f"✅ Successfully ingested {result['chunks_extracted']} chunks from {result['filename']}")
            print(f"📊 File type: {result['file_type']}")
            print(f"📝 Total characters: {result['total_characters']:,}")
        else:
            print(f"❌ Failed to ingest document: {result['error']}")
            return False
            
    elif path.is_dir():
        # Folder ingestion
        result = rag_system.ingest_folder(str(path))
        
        if result["success"]:
            print(f"✅ Successfully ingested {result['total_chunks']} chunks from {result['total_files']} files")
            print(f"📝 Total characters: {result['total_characters']:,}")
            print("\n📋 Files processed:")
            for filename, details in result['files_processed'].items():
                print(f"  • {filename}: {details['chunks']} chunks ({details['file_type'].upper()})")
        else:
            print(f"❌ Failed to ingest folder: {result['error']}")
            return False
    else:
        print(f"❌ Error: '{path}' is neither a file nor a directory.")
        return False
    
    # Show system status
    status = rag_system.get_system_status()
    print(f"\n📊 System Status:")
    print(f"  • Total documents: {status['vector_store']['unique_sources']}")
    print(f"  • Total chunks: {status['vector_store']['total_chunks']}")
    print(f"  • Model: {'OpenAI' if status['language_models']['openai_available'] else 'HuggingFace'}")
    
    return True

def ask_question(question: str, search_type: str = "hybrid"):
    """Ask a question to the RAG system"""
    print("🚀 Initializing RAG System...")
    rag_system = RAGSystem()
    
    # Check if there are any documents
    status = rag_system.get_system_status()
    if status['vector_store']['total_chunks'] == 0:
        print("❌ No documents found in the system. Please ingest some documents first using --ingest.")
        return False
    
    print(f"❓ Question: {question}")
    print(f"🔍 Search type: {search_type}")
    print("⏳ Processing...")
    
    result = rag_system.ask_question(question, search_type)
    
    print(f"\n🤖 Answer:")
    print("=" * 80)
    print(result["answer"])
    print("=" * 80)
    
    print(f"\n📊 Confidence: {result['confidence']:.1%}")
    print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
    print(f"🔧 Model used: {result.get('model_name', 'Unknown')}")
    
    if result.get("warning"):
        print(f"\n⚠️  Warning: {result['warning']}")
    
    if result["sources"]:
        print(f"\n📚 Sources ({len(result['sources'])} found):")
        for i, source in enumerate(result["sources"], 1):
            score_info = ""
            if source["scores"]:
                scores = [f"{k}: {v:.3f}" for k, v in source["scores"].items()]
                score_info = f" (scores: {', '.join(scores)})"
            
            print(f"  {i}. {source['filename']}{score_info}")
            print(f"     Preview: {source['content_preview'][:100]}...")
    else:
        print("\n📚 No sources found")
    
    return True

def start_server():
    """Start the FastAPI web server"""
    print("🚀 Starting RAG Q&A System Web Server...")
    print(f"🌐 Server will be available at: http://{Config.HOST}:{Config.PORT}")
    print(f"📖 API documentation at: http://{Config.HOST}:{Config.PORT}/api/docs")
    print(f"💻 Web interface at: http://{Config.HOST}:{Config.PORT}/ui")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "api:app",
            host=Config.HOST,
            port=Config.PORT,
            reload=False,
            log_level=Config.LOG_LEVEL.lower()
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")

def show_system_info():
    """Show system information and status"""
    print("🚀 Initializing RAG System...")
    rag_system = RAGSystem()
    
    print("🔧 RAG Q&A System Information")
    print("=" * 50)
    
    # System test
    test_result = rag_system.test_system()
    print(f"System Status: {'✅ Healthy' if test_result['system_status'] == 'healthy' else '⚠️ Degraded'}")
    
    # Get detailed status
    status = rag_system.get_system_status()
    
    print(f"\n📊 Vector Store:")
    print(f"  • Documents: {status['vector_store']['unique_sources']}")
    print(f"  • Chunks: {status['vector_store']['total_chunks']}")
    print(f"  • File types: {', '.join(status['vector_store']['file_types'].keys()) if status['vector_store']['file_types'] else 'None'}")
    
    print(f"\n🧠 Language Models:")
    print(f"  • OpenAI available: {'✅ Yes' if status['language_models']['openai_available'] else '❌ No'}")
    print(f"  • HuggingFace loaded: {'✅ Yes' if status['language_models']['models_loaded']['huggingface'] else '❌ No'}")
    print(f"  • Device: {status['language_models']['device']}")
    
    print(f"\n⚙️ Configuration:")
    print(f"  • Chunk size: {status['configuration']['chunk_size']}")
    print(f"  • Chunk overlap: {status['configuration']['chunk_overlap']}")
    print(f"  • Top-k retrieval: {status['configuration']['top_k_retrieval']}")
    print(f"  • Similarity threshold: {status['configuration']['similarity_threshold']}")
    print(f"  • Supported formats: {', '.join(status['configuration']['supported_extensions'])}")
    
    if status['vector_store']['sample_sources']:
        print(f"\n📄 Sample Documents:")
        for source in status['vector_store']['sample_sources'][:5]:
            print(f"  • {source}")

def main():
    """Main CLI entry point"""
    # Ensure directories exist
    Config.ensure_directories()
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="RAG Question & Answering System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --ingest ./documents                    # Ingest all files from documents folder
  %(prog)s --ingest document.pdf                   # Ingest a single PDF file
  %(prog)s --ask "What is the main topic?"         # Ask a question
  %(prog)s --ask "Summarize key points" --hybrid   # Ask with specific search type
  %(prog)s --api                                   # Start web server
  %(prog)s --info                                  # Show system information

Supported file formats: PDF, DOCX, PPTX, TXT, CSV
        """
    )
    
    # Main actions (mutually exclusive)
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument(
        "--ingest",
        type=str,
        metavar="PATH",
        help="Ingest documents from a file or folder"
    )
    action_group.add_argument(
        "--ask",
        type=str,
        metavar="QUESTION",
        help="Ask a question to the RAG system"
    )
    action_group.add_argument(
        "--api",
        action="store_true",
        help="Start the web API server"
    )
    action_group.add_argument(
        "--info",
        action="store_true",
        help="Show system information and status"
    )
    
    # Optional arguments
    parser.add_argument(
        "--search-type",
        choices=["semantic", "keyword", "hybrid"],
        default="hybrid",
        help="Search type for questions (default: hybrid)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=Config.HOST,
        help=f"Host for web server (default: {Config.HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=Config.PORT,
        help=f"Port for web server (default: {Config.PORT})"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=Config.LOG_LEVEL,
        help=f"Logging level (default: {Config.LOG_LEVEL})"
    )
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.host != Config.HOST:
        Config.HOST = args.host
    if args.port != Config.PORT:
        Config.PORT = args.port
    if args.log_level != Config.LOG_LEVEL:
        Config.LOG_LEVEL = args.log_level
    
    # Execute the requested action
    try:
        if args.ingest:
            success = ingest_documents(args.ingest)
            sys.exit(0 if success else 1)
            
        elif args.ask:
            success = ask_question(args.ask, args.search_type)
            sys.exit(0 if success else 1)
            
        elif args.api:
            start_server()
            
        elif args.info:
            show_system_info()
            
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        logging.exception("Unhandled exception in main")
        sys.exit(1)

if __name__ == "__main__":
    main()