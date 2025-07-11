from typing import List, Dict, Union
import numpy as np
from PIL import Image
import re
from src.rag.embeddings import MultimodalEmbedder
from src.config.settings import settings

class Retriever:
    def __init__(self, vector_store, embedder: MultimodalEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        # Simple query expansion - can be enhanced with more sophisticated methods
        expansions = [query]
        
        # Add common synonyms
        synonyms = {
            "budget": ["funding", "allocation", "finance", "cost"],
            "revenue": ["income", "earnings", "sales", "profit"],
            "student": ["learner", "pupil", "enrollee"],
            "faculty": ["staff", "professor", "teacher", "academic"],
            "program": ["course", "curriculum", "degree", "study"],
            "research": ["study", "investigation", "analysis", "examination"],
            "facility": ["building", "infrastructure", "campus", "premises"],
            "requirement": ["prerequisite", "condition", "criteria", "standard"]
        }
        
        query_lower = query.lower()
        for term, syns in synonyms.items():
            if term in query_lower:
                for syn in syns:
                    expanded = query_lower.replace(term, syn)
                    if expanded != query_lower:
                        expansions.append(expanded)
        
        return expansions

    def filter_by_metadata(self, chunks: List[Dict], query: str) -> List[Dict]:
        """Filter chunks based on metadata relevance"""
        query_lower = query.lower()
        filtered_chunks = []
        
        for chunk in chunks:
            score_boost = 0
            
            # Check if query contains numbers and chunk has numbers
            if re.findall(r'\d+', query) and chunk['metadata'].get('has_numbers', False):
                score_boost += 0.2
            
            # Check if query contains dates and chunk has dates
            if re.findall(r'\d{4}', query) and chunk['metadata'].get('has_dates', False):
                score_boost += 0.2
            
            # Check keywords
            keywords = chunk['metadata'].get('keywords', [])
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score_boost += 0.1
            
            # Check section headers
            section = chunk['metadata'].get('section', '').lower()
            if section and any(word in section for word in query_lower.split()):
                score_boost += 0.3
            
            # Apply score boost
            if score_boost > 0:
                chunk['score'] += score_boost
                filtered_chunks.append(chunk)
            else:
                filtered_chunks.append(chunk)
        
        return filtered_chunks

    def rerank_results(self, chunks: List[Dict], query: str) -> List[Dict]:
        """Re-rank results using content-based scoring"""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        for chunk in chunks:
            content_terms = set(re.findall(r'\b\w+\b', chunk['content'].lower()))
            
            # Calculate term overlap
            overlap = len(query_terms.intersection(content_terms))
            overlap_score = overlap / len(query_terms) if query_terms else 0
            
            # Calculate content length score (prefer medium-length chunks)
            length = len(chunk['content'])
            length_score = 1.0 if 100 <= length <= 500 else 0.5
            
            # Combine scores
            chunk['score'] = chunk['score'] * 0.7 + overlap_score * 0.2 + length_score * 0.1
        
        return sorted(chunks, key=lambda x: x['score'], reverse=True)

    def retrieve(self, query: Union[str, Image.Image], top_k: int=None, use_hybrid: bool=None) -> List[Dict]:
        """Enhanced retrieval with hybrid search and re-ranking"""
        try:
            # Use settings if not provided
            config = settings.get_retrieval_config()
            top_k = top_k or config["default_top_k"]
            use_hybrid = use_hybrid if use_hybrid is not None else config["use_hybrid_search"]
            
            if isinstance(query, str):
                if not query.strip():
                    raise ValueError("Empty text query")
                
                # Query expansion
                expanded_queries = self.expand_query(query) if use_hybrid and config["enable_query_expansion"] else [query]
                
                all_results = []
                
                # Get results for each expanded query
                for exp_query in expanded_queries:
                    query_embed = self.embedder.embed_text(exp_query)
                    
                    if not query_embed or len(query_embed) == 0:
                        continue
                    
                    # Get more results than needed for re-ranking
                    results = self.vector_store.text_collection.query(
                        query_embeddings=[query_embed],
                        n_results=top_k * 2
                    )
                    
                    if results['ids'] and len(results['ids'][0]) > 0:
                        for i in range(len(results['ids'][0])):
                            chunk = {
                                "content": results['documents'][0][i],
                                "score": results['distances'][0][i],
                                "metadata": results['metadatas'][0][i]
                            }
                            all_results.append(chunk)
                
                # Remove duplicates based on content
                seen_contents = set()
                unique_results = []
                for chunk in all_results:
                    content_hash = hash(chunk['content'][:100])  # Use first 100 chars as hash
                    if content_hash not in seen_contents:
                        seen_contents.add(content_hash)
                        unique_results.append(chunk)
                
                # Filter by metadata relevance
                filtered_results = self.filter_by_metadata(unique_results, query)
                
                # Re-rank results
                reranked_results = self.rerank_results(filtered_results, query)
                
                # Return top-k results
                return reranked_results[:top_k]
                
            else:
                # Image query - simpler processing
                query_embed = self.embedder.embed_image(query)
                
                if not query_embed or len(query_embed) == 0:
                    raise ValueError("Failed to generate query embedding")
                
                results = self.vector_store.image_collection.query(
                    query_embeddings=[query_embed],
                    n_results=top_k
                )
                
                retrieved_chunks = []
                if results['ids'] and len(results['ids'][0]) > 0:
                    for i in range(len(results['ids'][0])):
                        chunk = {
                            "content": results['documents'][0][i],
                            "score": results['distances'][0][i],
                            "metadata": results['metadatas'][0][i]
                        }
                        retrieved_chunks.append(chunk)
                
                return sorted(retrieved_chunks, key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            return []

    def retrieve_with_fallback(self, query: Union[str, Image.Image], top_k: int=None) -> List[Dict]:
        """Retrieve with fallback to broader search if initial results are poor"""
        config = settings.get_retrieval_config()
        top_k = top_k or config["default_top_k"]
        
        results = self.retrieve(query, top_k)
        
        # If results are poor (low scores), try with more results
        if results and all(r['score'] < config["min_score_threshold"] for r in results):
            print("Low confidence results, trying broader search...")
            results = self.retrieve(query, top_k * 2)
        
        return results