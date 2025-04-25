import sys
import os
sys.path.append("..")
from PIL import Image
from src.rag.vector_db import VectorStore
from src.rag.embeddings import MultimodalEmbedder
from src.processing.pdf_processor import PDFProcessor
from src.rag.retrieval import Retriever
from src.llm.generation import ResponseGenerator
from src.config.settings import settings

def test_full_flow():
    # Initialize all components
    embedder = MultimodalEmbedder()
    vector_db = VectorStore(embedder)
    llm = ResponseGenerator(os.getenv("OPENAI_API_KEY"))
    
    # List of university PDF documents
    university_pdfs = [
        "data/financials.pdf",
        "data/FYPHandbook2023.pdf",
        "data/AnnualReport.pdf"
    ]
    
    # Process and store all PDFs if not already in DB
    if (vector_db.text_collection.count() + vector_db.image_collection.count()) == 0:
        print("Processing university documents...")
        for pdf_path in university_pdfs:
            print(f"Processing {os.path.basename(pdf_path)}...")
            processor = PDFProcessor(pdf_path)
            chunks = processor.process()
            vector_db.add_documents(chunks)
        print(f"Total chunks stored: Text={vector_db.text_collection.count()}, Images={vector_db.image_collection.count()}")
    # Initialize retriever after DB setup
    retriever = Retriever(vector_db, embedder)
    
    # Test 1: Financial Query
    financial_context = retriever.retrieve("What was the total budget allocated for student facilities in 2023-2024?")
    financial_response = llm.generate_response(
        "What was the total budget allocated for student facilities in 2023-2024?",
        financial_context
    )
    print("\nFinancial Query Response:")
    print(financial_response)
    
    # Test 2: Academic Program Query
    fyp_context = retriever.retrieve("What are the requirements for Final Year Projects?")
    fyp_response = llm.generate_response(
        "Explain the FYP requirements and evaluation process",
        fyp_context
    )
    print("\nAcademic Program Response:")
    print(fyp_response)
    
    # Test 3: Image-based Query (assuming AnnualReport has images)
    try:
        # Use an actual image path from your processed images
        campus_image_path = "data/images/page_5_img_1.png"  
        campus_image = Image.open(campus_image_path)
        
        campus_context = retriever.retrieve(campus_image)
        campus_response = llm.generate_response(
            "Explain this campus development image",
            campus_context
        )
        print("\nImage Query Response:")
        print(campus_response)
    except FileNotFoundError:
        print("\nImage test skipped - no campus image found")

if __name__ == "__main__":
    test_full_flow()