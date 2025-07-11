import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import json
from pathlib import Path
import io
import os
import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document
from src.config.settings import settings

class PDFProcessor:
    def __init__(self, pdf_path: str, chunk_size: int = None, chunk_overlap: int = None):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        
        # Use settings if not provided
        config = settings.get_chunking_config()
        self.chunk_size = chunk_size or config["chunk_size"]
        self.chunk_overlap = chunk_overlap or config["chunk_overlap"]
        
        # Improved text splitter with better separators
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        return text.strip()

    def extract_text(self) -> List[Dict]:
        text_chunks = []
        total_pages = len(self.doc)
        
        for page_num, page in enumerate(self.doc):
            text = page.get_text()
            if not text.strip():
                print(f"No text found on page {page_num+1}")
                continue
            
            # Clean the text
            text = self.clean_text(text)
            if not text:
                continue
                
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create enhanced chunks with better metadata
            for chunk_idx, chunk in enumerate(chunks):
                # Skip very short chunks (likely noise)
                if len(chunk.strip()) < settings.MIN_CHUNK_LENGTH:
                    continue
                    
                # Extract key information for better retrieval
                chunk_info = self.extract_chunk_info(chunk, page_num + 1)
                
                text_chunks.append({
                    "content": chunk,
                    "metadata": {
                        "source": os.path.basename(self.pdf_path),
                        "page": page_num + 1,
                        "type": "text",
                        "chunk_id": chunk_idx,
                        "total_pages": total_pages,
                        "chunk_length": len(chunk),
                        "keywords": chunk_info.get("keywords", []),
                        "section": chunk_info.get("section", ""),
                        "has_numbers": chunk_info.get("has_numbers", False),
                        "has_dates": chunk_info.get("has_dates", False)
                    }
                })
        return text_chunks

    def extract_chunk_info(self, chunk: str, page: int) -> Dict:
        """Extract useful information from chunk for better retrieval"""
        info = {
            "keywords": [],
            "section": "",
            "has_numbers": False,
            "has_dates": False
        }
        
        # Extract numbers and dates
        numbers = re.findall(r'\d+(?:\.\d+)?', chunk)
        dates = re.findall(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\d{4}', chunk)
        
        info["has_numbers"] = len(numbers) > 0
        info["has_dates"] = len(dates) > 0
        
        # Extract potential section headers (lines in caps or with numbers)
        lines = chunk.split('\n')
        for line in lines[:3]:  # Check first few lines
            if re.match(r'^[A-Z\s\d]+$', line.strip()) and len(line.strip()) > 3:
                info["section"] = line.strip()
                break
        
        # Extract key terms (words that appear multiple times or are capitalized)
        words = re.findall(r'\b[A-Z][a-z]+\b|\b\w{6,}\b', chunk)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most frequent words as keywords
        info["keywords"] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        info["keywords"] = [word for word, freq in info["keywords"] if freq > 1]
        
        return info

    def extract_images(self, output_dir: str = "data/images") -> List[Dict]:
        """Extract images with enhanced OCR and metadata"""
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
                
                # Enhanced OCR processing with better configuration
                image = Image.open(io.BytesIO(image_bytes))
                
                # Try different OCR configurations for better results
                ocr_configs = [
                    '--oem 3 --psm 6',  # Default
                    '--oem 3 --psm 3',  # Fully automatic
                    '--oem 1 --psm 6'   # Legacy engine
                ]
                
                ocr_text = ""
                for config in ocr_configs:
                    try:
                        ocr_text = pytesseract.image_to_string(image, config=config)
                        if ocr_text.strip():
                            break
                    except:
                        continue
                
                if ocr_text.strip():
                    # Clean OCR text
                    ocr_text = self.clean_text(ocr_text)
                    
                    if len(ocr_text) > 20:  # Only include substantial OCR results
                        image_chunks.append({
                            "content": ocr_text,
                            "metadata": {
                                "source": os.path.basename(self.pdf_path),
                                "page": page_num + 1,
                                "type": "image",
                                "image_path": img_path,
                                "image_index": img_index + 1,
                                "ocr_length": len(ocr_text),
                                "has_numbers": bool(re.findall(r'\d+', ocr_text)),
                                "has_dates": bool(re.findall(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', ocr_text))
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

    def process(self) -> List[Dict]:
        """Process PDF and return enhanced chunks"""
        text_data = self.extract_text()
        image_data = self.extract_images()
        all_chunks = text_data + image_data
        
        print(f"Processed {len(text_data)} text chunks and {len(image_data)} image chunks")
        
        self.save_to_json(all_chunks)
        return all_chunks