#!/usr/bin/env python3
"""
Example usage of the RAG Question & Answering System
"""

import os
import tempfile
from pathlib import Path
from rag_system import RAGSystem

def create_sample_documents():
    """Create sample documents for testing"""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Sample text document
    doc1 = temp_dir / "sample_tech.txt"
    doc1.write_text("""
    Artificial Intelligence and Machine Learning
    
    Artificial Intelligence (AI) is a broad field of computer science focused on creating smart machines 
    capable of performing tasks that typically require human intelligence. These tasks include learning, 
    reasoning, problem-solving, perception, and language understanding.
    
    Machine Learning (ML) is a subset of AI that enables computers to learn and improve from experience 
    without being explicitly programmed. ML algorithms build mathematical models based on training data 
    to make predictions or decisions.
    
    Deep Learning is a subset of machine learning that uses artificial neural networks with multiple 
    layers (hence "deep") to model and understand complex patterns in data. It has been particularly 
    successful in areas like image recognition, natural language processing, and speech recognition.
    
    Key applications of AI include:
    - Computer vision and image recognition
    - Natural language processing and chatbots  
    - Autonomous vehicles and robotics
    - Medical diagnosis and drug discovery
    - Financial trading and fraud detection
    """)
    
    # Sample document about the company
    doc2 = temp_dir / "company_info.txt"
    doc2.write_text("""
    TechCorp Company Information
    
    Founded in 2020, TechCorp is a leading technology company specializing in AI-powered solutions.
    
    Our Mission: To democratize artificial intelligence and make advanced AI accessible to businesses 
    of all sizes.
    
    Our Products:
    1. AI Assistant Platform - Conversational AI for customer service
    2. Document Intelligence Suite - Automated document processing and analysis
    3. Predictive Analytics Engine - Data-driven insights and forecasting
    
    Team: We have 150+ employees across engineering, research, sales, and support teams.
    
    Locations: Headquarters in San Francisco, with offices in New York, London, and Singapore.
    
    Recent Achievements:
    - Processed over 10 million documents in 2023
    - Achieved 99.5% uptime across all services
    - Secured $50M Series B funding
    - Named "AI Startup of the Year" by Tech Magazine
    """)
    
    return temp_dir, [doc1, doc2]

def demo_rag_system():
    """Demonstrate the RAG system capabilities"""
    print("🚀 RAG System Demo")
    print("=" * 50)
    
    # Create sample documents
    print("📄 Creating sample documents...")
    temp_dir, documents = create_sample_documents()
    
    # Initialize RAG system
    print("🔧 Initializing RAG system...")
    rag = RAGSystem()
    
    # Ingest documents
    print("📚 Ingesting documents...")
    for doc in documents:
        result = rag.ingest_document(str(doc))
        if result["success"]:
            print(f"✅ Ingested: {doc.name} ({result['chunks_extracted']} chunks)")
        else:
            print(f"❌ Failed to ingest: {doc.name}")
    
    # Show system status
    print("\n📊 System Status:")
    status = rag.get_system_status()
    print(f"Documents: {status['vector_store']['unique_sources']}")
    print(f"Chunks: {status['vector_store']['total_chunks']}")
    print(f"Model: {'OpenAI' if status['language_models']['openai_available'] else 'HuggingFace'}")
    
    # Example questions
    questions = [
        "What is artificial intelligence?",
        "What products does TechCorp offer?",
        "When was TechCorp founded?",
        "What are the main applications of AI?",
        "How many employees does TechCorp have?"
    ]
    
    print("\n❓ Asking questions:")
    print("-" * 50)
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        
        result = rag.ask_question(question)
        
        print(f"   Answer: {result['answer'][:200]}...")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Sources: {len(result['sources'])} documents")
        
        if result['sources']:
            for j, source in enumerate(result['sources'][:2], 1):  # Show top 2 sources
                score = next(iter(source['scores'].values())) if source['scores'] else 'N/A'
                print(f"     {j}. {source['filename']} (score: {score})")
    
    # Test different search types
    print(f"\n🔍 Testing different search types:")
    print("-" * 50)
    
    test_question = "What are AI applications?"
    
    for search_type in ["semantic", "keyword", "hybrid"]:
        result = rag.ask_question(test_question, search_type=search_type)
        print(f"\n{search_type.title()} Search:")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Sources: {len(result['sources'])}")
        print(f"  Answer length: {len(result['answer'])} chars")
    
    # Test search functionality
    print(f"\n🔎 Testing document search:")
    print("-" * 50)
    
    search_results = rag.search_documents("machine learning", top_k=3)
    print(f"Found {search_results['total_results']} results for 'machine learning'")
    
    for i, result in enumerate(search_results['results'][:2], 1):
        score = next(iter(result['scores'].values())) if result['scores'] else 'N/A'
        print(f"  {i}. {result['filename']} (score: {score})")
        print(f"     Preview: {result['content'][:100]}...")
    
    # Session history
    print(f"\n📝 Session History:")
    print("-" * 50)
    
    history = rag.get_session_history()
    print(f"Total Q&A pairs: {len(history)}")
    
    if history:
        latest = history[-1]
        print(f"Latest question: {latest['question'][:50]}...")
        print(f"Confidence: {latest['confidence']:.1%}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up...")
    import shutil
    shutil.rmtree(temp_dir)
    print("Temporary files removed")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"🌐 To try the web interface, run: python main.py --api")

if __name__ == "__main__":
    try:
        demo_rag_system()
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()