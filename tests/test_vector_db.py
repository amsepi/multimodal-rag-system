import sys
sys.path.append("..")
from src.rag.vector_db import VectorStore
from src.rag.embeddings import MultimodalEmbedder
from src.processing.pdf_processor import PDFProcessor

def test_vector_db():
    # Initialize dependencies
    embedder = MultimodalEmbedder()
    processor = PDFProcessor("data/AnnualReport.pdf")
    vector_db = VectorStore(embedder)
    
    # Get chunks directly instead of JSON path
    chunks = processor.process()
    
    # Add to vector DB
    vector_db.add_documents(chunks)
    
    # Verify counts
    count = vector_db.collection.count()
    print(f"Stored {count} documents")
    assert count == len(chunks)

if __name__ == "__main__":
    test_vector_db()