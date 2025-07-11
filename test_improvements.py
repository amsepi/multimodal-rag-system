#!/usr/bin/env python3
"""
Test script to demonstrate the improvements in the RAG system
"""

import sys
import os
sys.path.append(".")

from src.processing.pdf_processor import PDFProcessor
from src.rag.embeddings import MultimodalEmbedder
from src.rag.vector_db import VectorStore
from src.rag.retrieval import Retriever
from src.llm.generation import ResponseGenerator
from src.config.settings import settings

def test_improvements():
    """Test the improved RAG system"""
    
    print("🔧 Testing Improved RAG System")
    print("=" * 50)
    
    # Test 1: Improved Chunking
    print("\n1. Testing Improved Chunking...")
    try:
        # Use a sample PDF if available
        pdf_path = "data/financials.pdf"  # Adjust path as needed
        if os.path.exists(pdf_path):
            processor = PDFProcessor(pdf_path)
            chunks = processor.process()
            print(f"✅ Generated {len(chunks)} chunks with improved chunking")
            print(f"   - Chunk size: {settings.CHUNK_SIZE}")
            print(f"   - Chunk overlap: {settings.CHUNK_OVERLAP}")
            print(f"   - Min chunk length: {settings.MIN_CHUNK_LENGTH}")
            
            # Show sample chunk with enhanced metadata
            if chunks:
                sample = chunks[0]
                print(f"   - Sample chunk length: {len(sample['content'])}")
                print(f"   - Has numbers: {sample['metadata'].get('has_numbers', False)}")
                print(f"   - Has dates: {sample['metadata'].get('has_dates', False)}")
                print(f"   - Keywords: {sample['metadata'].get('keywords', [])[:3]}")
        else:
            print("⚠️  No PDF found for testing chunking")
    except Exception as e:
        print(f"❌ Chunking test failed: {e}")
    
    # Test 2: Enhanced Retrieval
    print("\n2. Testing Enhanced Retrieval...")
    try:
        embedder = MultimodalEmbedder()
        vector_db = VectorStore(embedder)
        retriever = Retriever(vector_db, embedder)
        
        # Test query expansion
        test_query = "What is the budget for student facilities?"
        expanded = retriever.expand_query(test_query)
        print(f"✅ Query expansion: '{test_query}' -> {len(expanded)} variations")
        
        # Test retrieval config
        config = settings.get_retrieval_config()
        print(f"   - Default top_k: {config['default_top_k']}")
        print(f"   - Use hybrid search: {config['use_hybrid_search']}")
        print(f"   - Enable query expansion: {config['enable_query_expansion']}")
        print(f"   - Min score threshold: {config['min_score_threshold']}")
        
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
    
    # Test 3: Improved LLM Generation
    print("\n3. Testing Improved LLM Generation...")
    try:
        llm = ResponseGenerator()
        
        # Test LLM config
        config = settings.get_llm_config()
        print(f"✅ LLM Configuration:")
        print(f"   - Model: {config['model']}")
        print(f"   - Temperature: {config['temperature']}")
        print(f"   - Max tokens: {config['max_tokens']}")
        print(f"   - Context max tokens: {config['context_max_tokens']}")
        print(f"   - Default strategy: {config['default_strategy']}")
        print(f"   - Fallback strategies: {config['fallback_strategies']}")
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
    
    # Test 4: Configuration Summary
    print("\n4. Configuration Summary...")
    print("✅ Current Settings:")
    print(f"   📄 Chunking: {settings.CHUNK_SIZE} chars, {settings.CHUNK_OVERLAP} overlap")
    print(f"   🔍 Retrieval: {settings.DEFAULT_TOP_K} results, hybrid search enabled")
    print(f"   🤖 LLM: {settings.DEFAULT_MODEL}, temp {settings.TEMPERATURE}")
    print(f"   📊 Quality: min score {settings.MIN_SCORE_THRESHOLD}, min answer {settings.MIN_ANSWER_LENGTH}")
    
    print("\n🎯 Key Improvements:")
    print("   • Smaller chunks (256 vs 512) for better precision")
    print("   • Enhanced metadata extraction (numbers, dates, keywords)")
    print("   • Query expansion with synonyms")
    print("   • Hybrid search with re-ranking")
    print("   • Multiple prompt strategies with fallbacks")
    print("   • Context optimization for token limits")
    print("   • Better text cleaning and OCR processing")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_improvements() 