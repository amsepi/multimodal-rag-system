import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import json
from pathlib import Path
import io
import os
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text(self) -> List[Dict]:
        """Extract text with page-wise metadata"""
        text_chunks = []
        for page_num, page in enumerate(self.doc):
            text = page.get_text()
            chunks = self.text_splitter.split_text(text)
            
            for chunk in chunks:
                text_chunks.append({
                    "content": chunk,
                    "metadata": {
                        "source": os.path.basename(self.pdf_path),
                        "page": page_num + 1,
                        "type": "text"
                    }
                })
        return text_chunks

    def extract_images(self, output_dir: str = "data/images") -> List[Dict]:
        """Extract images with OCR and metadata"""
        os.makedirs(output_dir, exist_ok=True)
        image_chunks = []
        
        for page_num, page in enumerate(self.doc):
            img_list = page.get_images(full=True)
            
            for img_index, img in enumerate(img_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save image
                img_path = os.path.join(output_dir, 
                                      f"page_{page_num+1}_img_{img_index+1}.{base_image['ext']}")
                with open(img_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # OCR processing
                image = Image.open(io.BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(image)
                
                if ocr_text.strip():
                    image_chunks.append({
                        "content": ocr_text,
                        "metadata": {
                            "source": os.path.basename(self.pdf_path),
                            "page": page_num + 1,
                            "type": "image",
                            "image_path": img_path
                        }
                    })
        return image_chunks
    
    def save_to_json(self, chunks: List[Dict], output_dir: str = "data/text"):
        """Save processed chunks to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / f"{Path(self.pdf_path).stem}_chunks.json"
        
        with open(output_path, 'w') as f:
            json.dump(chunks, f, indent=2)
            
        return str(output_path)

    def process(self) -> List[Dict]:  # Change return type from str to List[Dict]
        """Process PDF and return chunks"""
        text_data = self.extract_text()
        image_data = self.extract_images()
        all_chunks = text_data + image_data
        self.save_to_json(all_chunks)  # Save but don't return path
        return all_chunks  # Return actual chunks list