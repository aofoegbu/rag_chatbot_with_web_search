#!/usr/bin/env python3
"""
Test script to verify the enhanced knowledge system works correctly
Tests various knowledge domains and response quality
"""

import sys
import os
sys.path.append('.')

from web_search_integration import WebSearchIntegrator
from model_handler import ModelHandler
from rag_system import RAGSystem
from database import DatabaseManager

def test_knowledge_domains():
    """Test various knowledge domains to ensure comprehensive responses."""
    print("🧪 Testing Knowledge Domains...")
    
    integrator = WebSearchIntegrator()
    test_queries = [
        # AI/ML domain
        "What is machine learning and how does it work?",
        "Explain neural networks and deep learning",
        
        # Programming domain
        "What are the best programming languages for data science?",
        "How does software development work?",
        
        # Climate/Environment domain
        "What causes climate change?",
        "How do renewable energy systems work?",
        
        # Business domain
        "What are the key principles of business management?",
        "How does digital marketing work?",
        
        # Health domain
        "What are the main areas of medicine?",
        "How does public health work?",
        
        # Education domain
        "What are different learning theories?",
        "How does online education work?",
        
        # General knowledge fallback
        "Tell me about quantum physics",
        "Explain the history of art"
    ]
    
    results = []
    for query in test_queries:
        print(f"\n📝 Testing: {query}")
        try:
            enhanced_context = integrator._enhance_with_knowledge(query)
            if enhanced_context and len(enhanced_context) > 100:
                print(f"✅ Generated comprehensive response ({len(enhanced_context)} chars)")
                print(f"   Preview: {enhanced_context[:150]}...")
                results.append({"query": query, "status": "SUCCESS", "length": len(enhanced_context)})
            else:
                print(f"❌ Response too short or empty: {len(enhanced_context) if enhanced_context else 0} chars")
                results.append({"query": query, "status": "FAILED", "length": len(enhanced_context) if enhanced_context else 0})
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({"query": query, "status": "ERROR", "error": str(e)})
    
    return results

def test_model_handler_responses():
    """Test model handler with various inputs."""
    print("\n🤖 Testing Model Handler...")
    
    handler = ModelHandler()
    test_inputs = [
        "What is artificial intelligence?",
        "How do solar panels work?",
        "Explain database design principles",
        "What are the benefits of exercise?",
        "How does photosynthesis work?"
    ]
    
    results = []
    for input_text in test_inputs:
        print(f"\n🔍 Testing model response: {input_text}")
        try:
            response = handler.generate_response(input_text)
            if response and len(response) > 50:
                print(f"✅ Generated response ({len(response)} chars)")
                print(f"   Preview: {response[:100]}...")
                results.append({"input": input_text, "status": "SUCCESS", "length": len(response)})
            else:
                print(f"❌ Response too short: {len(response) if response else 0} chars")
                results.append({"input": input_text, "status": "FAILED", "length": len(response) if response else 0})
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({"input": input_text, "status": "ERROR", "error": str(e)})
    
    return results

def test_citations_and_sources():
    """Test that responses include proper citations."""
    print("\n📚 Testing Citations and Sources...")
    
    integrator = WebSearchIntegrator()
    test_queries = [
        "machine learning applications",
        "renewable energy benefits", 
        "programming best practices"
    ]
    
    results = []
    for query in test_queries:
        print(f"\n🔍 Testing citations for: {query}")
        try:
            enhanced_context = integrator._enhance_with_knowledge(query)
            has_sources = "Knowledge Sources" in enhanced_context or "Internal knowledge" in enhanced_context
            has_references = "**" in enhanced_context and any(term in enhanced_context.lower() for term in ["knowledge", "source", "reference"])
            
            if has_sources and has_references:
                print("✅ Contains proper citations and sources")
                results.append({"query": query, "status": "SUCCESS", "has_citations": True})
            else:
                print("❌ Missing citations or sources")
                results.append({"query": query, "status": "FAILED", "has_citations": False})
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({"query": query, "status": "ERROR", "error": str(e)})
    
    return results

def test_database_integration():
    """Test database functionality."""
    print("\n💾 Testing Database Integration...")
    
    try:
        db = DatabaseManager()
        
        # Test basic operations
        stats = db.get_database_stats()
        print(f"✅ Database stats retrieved: {stats['database_type']}")
        
        # Test document count
        doc_count = db.get_document_count()
        print(f"✅ Document count: {doc_count}")
        
        return {"status": "SUCCESS", "stats": stats}
        
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        return {"status": "ERROR", "error": str(e)}

def generate_test_report(knowledge_results, model_results, citation_results, db_result):
    """Generate comprehensive test report."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE TEST REPORT")
    print("="*60)
    
    # Knowledge domain results
    knowledge_success = sum(1 for r in knowledge_results if r["status"] == "SUCCESS")
    print(f"\n🧠 Knowledge Domains: {knowledge_success}/{len(knowledge_results)} passed")
    
    # Model handler results
    model_success = sum(1 for r in model_results if r["status"] == "SUCCESS")
    print(f"🤖 Model Responses: {model_success}/{len(model_results)} passed")
    
    # Citation results
    citation_success = sum(1 for r in citation_results if r["status"] == "SUCCESS")
    print(f"📚 Citations/Sources: {citation_success}/{len(citation_results)} passed")
    
    # Database results
    db_status = "✅ PASSED" if db_result["status"] == "SUCCESS" else "❌ FAILED"
    print(f"💾 Database Integration: {db_status}")
    
    # Overall success rate
    total_tests = len(knowledge_results) + len(model_results) + len(citation_results) + 1
    total_passed = knowledge_success + model_success + citation_success + (1 if db_result["status"] == "SUCCESS" else 0)
    success_rate = (total_passed / total_tests) * 100
    
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({total_passed}/{total_tests})")
    
    if success_rate >= 80:
        print("🟢 EXCELLENT: System performing very well!")
    elif success_rate >= 60:
        print("🟡 GOOD: System mostly functional with minor issues")
    else:
        print("🔴 NEEDS IMPROVEMENT: Several components need attention")
    
    # Detailed failure analysis
    failed_knowledge = [r for r in knowledge_results if r["status"] != "SUCCESS"]
    failed_model = [r for r in model_results if r["status"] != "SUCCESS"]
    failed_citations = [r for r in citation_results if r["status"] != "SUCCESS"]
    
    if failed_knowledge:
        print(f"\n❌ Failed Knowledge Tests ({len(failed_knowledge)}):")
        for failure in failed_knowledge[:3]:  # Show first 3
            print(f"   - {failure['query']}")
    
    if failed_model:
        print(f"\n❌ Failed Model Tests ({len(failed_model)}):")
        for failure in failed_model[:3]:
            print(f"   - {failure['input']}")
    
    if failed_citations:
        print(f"\n❌ Failed Citation Tests ({len(failed_citations)}):")
        for failure in failed_citations:
            print(f"   - {failure['query']}")
    
    print("\n" + "="*60)
    return success_rate

def main():
    """Run comprehensive knowledge system tests."""
    print("🚀 Starting Comprehensive Knowledge System Tests...")
    print("="*60)
    
    # Run all test categories
    knowledge_results = test_knowledge_domains()
    model_results = test_model_handler_responses()
    citation_results = test_citations_and_sources()
    db_result = test_database_integration()
    
    # Generate final report
    success_rate = generate_test_report(knowledge_results, model_results, citation_results, db_result)
    
    return success_rate >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)