from typing import List, Dict, Union
import numpy as np
from PIL import Image
from src.rag.embeddings import MultimodalEmbedder

class Retriever:
    def __init__(self, vector_store, embedder: MultimodalEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: Union[str, Image.Image], top_k: int=5) -> List[Dict]:
        """Handle text/image queries and return ranked results"""
        try:
            # Validate query and get embedding
            if isinstance(query, str):
                if not query.strip():
                    raise ValueError("Empty text query")
                query_embed = self.embedder.embed_text(query)
                collection = self.vector_store.text_collection
            else:
                query_embed = self.embedder.embed_image(query)
                collection = self.vector_store.image_collection

            # Validate embedding
            if not query_embed or len(query_embed) == 0:
                raise ValueError("Failed to generate query embedding")

            # Execute query
            results = collection.query(
                query_embeddings=[query_embed],
                n_results=top_k
            )

            # Process results safely
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
            
            return []  # Return empty list if no results

        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            return []