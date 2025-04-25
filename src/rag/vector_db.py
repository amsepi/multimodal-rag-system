import chromadb
from chromadb.config import Settings
from typing import List, Dict
import uuid
from src.rag.embeddings import MultimodalEmbedder  

class VectorStore:
    def __init__(self, embedder: MultimodalEmbedder):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=".chromadb/")
        
        # Separate collections for text vs images
        self.text_collection = self.client.get_or_create_collection(
            name="text_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        self.image_collection = self.client.get_or_create_collection(
            name="image_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks: List[Dict]):
        text_ids, text_docs, text_embeds = [], [], []
        img_ids, img_docs, img_embeds = [], [], []
        
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            
            if chunk["metadata"]["type"] == "text":
                embedding = self.embedder.embed_text(chunk["content"])
                # Add these checks
                if not embedding or len(embedding) != 384:
                    print(f"Skipping invalid text embedding: {chunk['content'][:50]}...")
                    continue
            else:
                embedding = self.embedder.embed_image(
                    chunk["metadata"]["image_path"]
                )
                # Add these checks
                if not embedding or len(embedding) != 512:
                    print(f"Skipping invalid image embedding: {chunk['metadata']['image_path']}")
                    continue
                img_ids.append(doc_id)
                img_docs.append(chunk["content"])
                img_embeds.append(embedding)

        # Add to separate collections
        if text_embeds:
            self.text_collection.add(
                ids=text_ids,
                documents=text_docs,
                metadatas=[c["metadata"] for c in chunks if c["metadata"]["type"] == "text"],
                embeddings=text_embeds
            )
        if img_embeds:
            self.image_collection.add(
                ids=img_ids,
                documents=img_docs,
                metadatas=[c["metadata"] for c in chunks if c["metadata"]["type"] == "image"],
                embeddings=img_embeds
            )
    
    