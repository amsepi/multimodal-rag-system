import sys
sys.path.append("..")
from src.processing.pdf_processor import PDFProcessor

def test_text_extraction(pdf_path="data/financials.pdf"):
    processor = PDFProcessor(pdf_path)
    text_chunks = processor.extract_text()
    
    print(f"\nExtracted {len(text_chunks)} text chunks from {pdf_path}")
    print("Sample text chunks:")
    for i, chunk in enumerate(text_chunks[:3]):
        print(f"Chunk {i+1}: {chunk['content'][:200]}...")

if __name__ == "__main__":
    test_text_extraction()