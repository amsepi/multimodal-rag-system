import sys
import json
sys.path.append("..")
from src.processing.pdf_processor import PDFProcessor

def test_processor():
    processor = PDFProcessor("data/AnnualReport.pdf")
    output_file = processor.process()  # Now returns JSON path
    
    print(f"Output saved to: {output_file}")
    with open(output_file) as f:
        data = json.load(f)
    
    print(f"\nTotal chunks: {len(data)}")
    print(f"Sample chunk:\n{json.dumps(data[0], indent=2)}")

if __name__ == "__main__":
    test_processor()