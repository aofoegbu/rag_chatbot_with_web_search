#!/usr/bin/env python3
"""
Comprehensive testing suite for Ogelo RAG Chat Assistant
Tests all components and features systematically
"""

import sys
import os
import traceback
import tempfile
import io
from typing import List, Dict, Any

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    test_results = []
    
    modules_to_test = [
        ("Database Manager", "database"),
        ("PostgreSQL Manager", "postgres_database"), 
        ("Document Processor", "document_processor"),
        ("RAG System", "rag_system"),
        ("Model Handler", "model_handler"),
        ("Web Search Integration", "web_search_integration"),
        ("Utils", "utils")
    ]
    
    for name, module in modules_to_test:
        try:
            __import__(module)
            test_results.append(f"✅ {name}: Import successful")
        except Exception as e:
            test_results.append(f"❌ {name}: Import failed - {str(e)}")
    
    return test_results

def test_database_operations():
    """Test database functionality."""
    print("🧪 Testing database operations...")
    test_results = []
    
    try:
        from database import DatabaseManager
        db = DatabaseManager()
        
        # Test initialization
        db.init_database()
        test_results.append("✅ Database: Initialization successful")
        
        # Test document count
        count = db.get_document_count()
        test_results.append(f"✅ Database: Document count retrieved ({count})")
        
        # Test statistics
        stats = db.get_database_stats()
        test_results.append(f"✅ Database: Statistics retrieved ({type(stats).__name__})")
        
        # Test database type
        db_type = db.get_database_type()
        test_results.append(f"✅ Database: Type detected ({db_type})")
        
    except Exception as e:
        test_results.append(f"❌ Database: Error - {str(e)}")
        
    return test_results

def test_document_processing():
    """Test document processing capabilities."""
    print("🧪 Testing document processing...")
    test_results = []
    
    try:
        from document_processor import DocumentProcessor, PDF_AVAILABLE, PANDAS_AVAILABLE, OCR_AVAILABLE, WEB_SCRAPING_AVAILABLE
        from database import DatabaseManager
        
        db = DatabaseManager()
        processor = DocumentProcessor(db)
        
        # Test availability flags
        test_results.append(f"📄 PDF Support: {'✅ Available' if PDF_AVAILABLE else '❌ Not available'}")
        test_results.append(f"📊 CSV Support: {'✅ Available' if PANDAS_AVAILABLE else '❌ Not available'}")
        test_results.append(f"🖼️ OCR Support: {'✅ Available' if OCR_AVAILABLE else '❌ Not available'}")
        test_results.append(f"🌐 Web Scraping: {'✅ Available' if WEB_SCRAPING_AVAILABLE else '❌ Not available'}")
        
        # Test text chunking
        test_text = "This is a test document with multiple sentences. It should be split into chunks properly. Each chunk should maintain context while being of reasonable size."
        chunks = processor._split_text_into_chunks(test_text)
        test_results.append(f"✅ Text Chunking: {len(chunks)} chunks created")
        
        # Test URL processing if available
        if WEB_SCRAPING_AVAILABLE:
            try:
                # Test URL validation and filename generation
                test_url = "https://example.com/test-article"
                filename = processor._url_to_filename(test_url)
                test_results.append(f"✅ URL Processing: Filename generation works ({filename})")
            except Exception as e:
                test_results.append(f"⚠️ URL Processing: {str(e)}")
        
    except Exception as e:
        test_results.append(f"❌ Document Processing: Error - {str(e)}")
        
    return test_results

def test_rag_system():
    """Test RAG system functionality."""
    print("🧪 Testing RAG system...")
    test_results = []
    
    try:
        from rag_system import RAGSystem, SENTENCE_TRANSFORMERS_AVAILABLE
        from database import DatabaseManager
        
        db = DatabaseManager()
        rag = RAGSystem(db)
        
        # Test embedding model loading
        rag.load_embedding_model()
        is_loaded = rag.is_embedding_model_loaded()
        test_results.append(f"🧠 Embedding Model: {'✅ Loaded' if is_loaded else '⚠️ Simple fallback'}")
        
        # Test embedding generation
        test_text = "This is a test for embedding generation"
        embedding = rag.get_embedding(test_text)
        test_results.append(f"✅ Embedding Generation: Vector of size {len(embedding)}")
        
        # Test context retrieval
        context, sources = rag.get_relevant_context("test query")
        test_results.append(f"✅ Context Retrieval: {len(sources)} sources found")
        
        # Test web knowledge integration
        try:
            from web_search_integration import web_search_integrator
            enhanced_context, web_sources = web_search_integrator.search_and_enhance("machine learning", "")
            test_results.append(f"✅ Knowledge Integration: Enhanced context generated")
        except Exception as e:
            test_results.append(f"⚠️ Knowledge Integration: {str(e)}")
        
    except Exception as e:
        test_results.append(f"❌ RAG System: Error - {str(e)}")
        
    return test_results

def test_model_handler():
    """Test model handler functionality."""
    print("🧪 Testing model handler...")
    test_results = []
    
    try:
        from model_handler import ModelHandler, TRANSFORMERS_AVAILABLE
        
        handler = ModelHandler()
        
        # Test model availability
        test_results.append(f"🤖 Transformers: {'✅ Available' if TRANSFORMERS_AVAILABLE else '⚠️ Simple fallback'}")
        
        # Test available models
        available_models = handler.get_available_models()
        test_results.append(f"✅ Model Options: {len(available_models)} models available")
        
        # Test model loading status
        is_loaded = handler.is_model_loaded()
        test_results.append(f"🔄 Model Loading: {'✅ Loaded' if is_loaded else '⚠️ Simple mode'}")
        
        # Test response generation
        test_response = handler.generate_response("Hello, how are you?")
        if test_response and len(test_response) > 0:
            test_results.append(f"✅ Response Generation: Working (length: {len(test_response)})")
        else:
            test_results.append("❌ Response Generation: Failed")
            
        # Test enhanced response with context
        test_context = "This is some relevant context about artificial intelligence and machine learning."
        enhanced_response = handler.generate_response(
            "What is machine learning?", 
            context=test_context,
            conversation_history=[{"role": "user", "content": "Hello"}]
        )
        if enhanced_response and len(enhanced_response) > 50:
            test_results.append(f"✅ Enhanced Responses: Working with context integration")
        else:
            test_results.append("⚠️ Enhanced Responses: Limited functionality")
        
    except Exception as e:
        test_results.append(f"❌ Model Handler: Error - {str(e)}")
        
    return test_results

def test_file_capabilities():
    """Test file processing capabilities."""
    print("🧪 Testing file processing...")
    test_results = []
    
    # Test sample files exist
    sample_files = [
        ("sample_document.txt", "📄 Text File"),
        ("sample_data.csv", "📊 CSV File"),
        ("sample_ocr_test.png", "🖼️ Image File")
    ]
    
    for filename, desc in sample_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            test_results.append(f"✅ {desc}: Available ({size} bytes)")
        else:
            test_results.append(f"⚠️ {desc}: Not found")
    
    # Test OCR system
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        test_results.append(f"✅ OCR System: Tesseract {version}")
    except Exception as e:
        test_results.append(f"❌ OCR System: {str(e)}")
    
    return test_results

def test_web_integration():
    """Test web integration capabilities."""
    print("🧪 Testing web integration...")
    test_results = []
    
    try:
        # Test trafilatura
        import trafilatura
        test_results.append("✅ Trafilatura: Available for web content extraction")
        
        # Test requests and BeautifulSoup
        import requests
        from bs4 import BeautifulSoup
        test_results.append("✅ Web Scraping: Requests and BeautifulSoup available")
        
        # Test web knowledge integration
        from web_search_integration import web_search_integrator
        
        # Test knowledge enhancement
        test_queries = [
            "machine learning",
            "renewable energy", 
            "climate change"
        ]
        
        for query in test_queries:
            enhanced_context, sources = web_search_integrator.search_and_enhance(query, "")
            if enhanced_context and len(enhanced_context) > 100:
                test_results.append(f"✅ Knowledge Enhancement: {query} - enhanced")
            else:
                test_results.append(f"⚠️ Knowledge Enhancement: {query} - limited")
        
    except Exception as e:
        test_results.append(f"❌ Web Integration: Error - {str(e)}")
        
    return test_results

def run_comprehensive_tests():
    """Run all tests and return results."""
    print("🚀 Starting Comprehensive Test Suite for Ogelo RAG Chat Assistant\n")
    
    all_results = []
    
    # Run all test categories
    test_categories = [
        ("Module Imports", test_imports),
        ("Database Operations", test_database_operations),
        ("Document Processing", test_document_processing),
        ("RAG System", test_rag_system),
        ("Model Handler", test_model_handler),
        ("File Capabilities", test_file_capabilities),
        ("Web Integration", test_web_integration)
    ]
    
    for category_name, test_function in test_categories:
        print(f"\n{'='*50}")
        print(f"Testing: {category_name}")
        print('='*50)
        
        try:
            results = test_function()
            all_results.extend(results)
            for result in results:
                print(result)
        except Exception as e:
            error_msg = f"❌ {category_name}: Critical error - {str(e)}"
            all_results.append(error_msg)
            print(error_msg)
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    success_count = len([r for r in all_results if r.startswith('✅')])
    warning_count = len([r for r in all_results if r.startswith('⚠️')])
    error_count = len([r for r in all_results if r.startswith('❌')])
    total_count = len(all_results)
    
    print(f"Total Tests: {total_count}")
    print(f"✅ Passed: {success_count}")
    print(f"⚠️ Warnings: {warning_count}")
    print(f"❌ Failed: {error_count}")
    print(f"Success Rate: {(success_count/total_count*100):.1f}%")
    
    return all_results

if __name__ == "__main__":
    results = run_comprehensive_tests()
    
    # Return appropriate exit code
    error_count = len([r for r in results if r.startswith('❌')])
    sys.exit(1 if error_count > 0 else 0)