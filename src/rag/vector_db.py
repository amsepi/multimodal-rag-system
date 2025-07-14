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
        text_ids, text_docs, text_embeds, text_metas = [], [], [], []
        img_ids, img_docs, img_embeds, img_metas = [], [], [], []
        
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            
            if chunk["metadata"]["type"] == "text":
                try:
                    # Text embedding generation
                    embedding = self.embedder.embed_text(chunk["content"])
                    
                    # Validate text embedding
                    if not embedding or len(embedding) != 1024:
                        print(f"Skipping invalid text embedding: {chunk['content'][:50]}...")
                        continue
                    
                    # Store text data
                    text_ids.append(doc_id)
                    text_docs.append(chunk["content"])
                    text_embeds.append(embedding)
                    text_metas.append(chunk["metadata"])
                    
                except Exception as e:
                    print(f"Error processing text chunk: {str(e)}")
                    continue

            else:
                try:
                    # Image embedding generation
                    embedding = self.embedder.embed_image(
                        chunk["metadata"]["image_path"]
                    )
                    
                    # Validate image embedding
                    if not embedding or len(embedding) != 4096:
                        print(f"Skipping invalid image embedding: {chunk['metadata']['image_path']}")
                        continue
                    
                    # Store image data
                    img_ids.append(doc_id)
                    img_docs.append(chunk["content"])
                    img_embeds.append(embedding)
                    img_metas.append(chunk["metadata"])
                    
                except Exception as e:
                    print(f"Error processing image chunk: {str(e)}")
                    continue

        # Add to Chroma collections
        if text_embeds:
            self.text_collection.add(
                ids=text_ids,
                documents=text_docs,
                metadatas=text_metas,
                embeddings=text_embeds
            )
            print(f"Added {len(text_embeds)} text chunks to DB")
            
        if img_embeds:
            self.image_collection.add(
                ids=img_ids,
                documents=img_docs,
                metadatas=img_metas,
                embeddings=img_embeds
            )
            print(f"Added {len(img_embeds)} image chunks to DB")
        
        