import chromadb
from chromadb.config import Settings
from typing import List, Dict
import uuid
from src.rag.embeddings import MultimodalEmbedder  

class VectorStore:
    
    def __init__(self, embedder: MultimodalEmbedder):
        self.embedder = embedder
        
        # Updated Chroma client configuration
        self.client = chromadb.PersistentClient(
            path=".chromadb/",
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name="multimodal_rag",
            metadata={"hnsw:space": "cosine"}
        )
    def add_documents(self, chunks: List[Dict]):
        """Store text/image embeddings with metadata"""
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            documents.append(chunk["content"])
            metadatas.append(chunk["metadata"])
            
            if chunk["metadata"]["type"] == "text":
                embedding = self.embedder.embed_text(chunk["content"])
            else:
                embedding = self.embedder.embed_image(
                    chunk["metadata"]["image_path"]
                )
                
            if embedding:
                embeddings.append(embedding)
            else:
                print(f"Skipped invalid embedding for {doc_id}")

        # This single add call will automatically persist
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
