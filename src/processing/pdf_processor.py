import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import json
from pathlib import Path
import io
import os
import re
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.config.settings import settings
import nltk
from nltk.tokenize import sent_tokenize

try:
    import camelot
    HAS_CAMELOT = True
except ImportError:
    HAS_CAMELOT = False

class PDFProcessor:
    def __init__(self, pdf_path: str, chunk_size: int = None, chunk_overlap: int = None):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        config = settings.get_chunking_config()
        self.chunk_size = chunk_size or config["chunk_size"]
        self.chunk_overlap = chunk_overlap or config["chunk_overlap"]

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'^[0-9]+\s*$', '', text, flags=re.MULTILINE)
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
            text = self.clean_text(text)
            if not text:
                continue
            # --- Sentence-aware chunking ---
            sentences = sent_tokenize(text)
            chunks = []
            current = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 <= self.chunk_size:
                    current += (" " if current else "") + sent
                else:
                    if current:
                        chunks.append(current.strip())
                    current = sent
            if current:
                chunks.append(current.strip())
            # Overlap: add previous chunk's last N chars to next chunk
            overlapped_chunks = []
            overlap = self.chunk_overlap
            for i, chunk in enumerate(chunks):
                if i > 0 and overlap > 0:
                    prev = overlapped_chunks[-1]
                    chunk = prev[-overlap:] + " " + chunk
                overlapped_chunks.append(chunk.strip())
            # ---
            for chunk_idx, chunk in enumerate(overlapped_chunks):
                if len(chunk.strip()) < settings.MIN_CHUNK_LENGTH:
                    continue
                chunk_info = self.extract_chunk_info(chunk, page_num + 1)
                has_table = self.detect_table(chunk)
                section = self.detect_section(chunk)
                keywords = self.extract_keywords(chunk)
                text_chunks.append({
                    "content": chunk,
                    "metadata": {
                        "source": os.path.basename(self.pdf_path),
                        "page": page_num + 1,
                        "type": "text",
                        "chunk_id": chunk_idx,
                        "total_pages": total_pages,
                        "chunk_length": len(chunk),
                        "keywords": ", ".join(keywords),
                        "section": section,
                        "has_numbers": chunk_info.get("has_numbers", False),
                        "has_dates": chunk_info.get("has_dates", False),
                        "has_table": has_table
                    }
                })
        # --- Table extraction with camelot ---
        if HAS_CAMELOT:
            try:
                tables = camelot.read_pdf(self.pdf_path, pages="all")
                for t in tables:
                    table_str = t.df.to_string(index=False, header=True)
                    text_chunks.append({
                        "content": table_str,
                        "metadata": {
                            "source": os.path.basename(self.pdf_path),
                            "page": t.page,
                            "type": "table",
                            "chunk_id": f"table_{t.page}",
                            "total_pages": total_pages,
                            "chunk_length": len(table_str),
                            "keywords": "table",
                            "section": "TABLE",
                            "has_numbers": True,
                            "has_dates": False,
                            "has_table": True
                        }
                    })
            except Exception as e:
                print(f"Camelot table extraction failed: {e}")
        return text_chunks

    def extract_chunk_info(self, chunk: str, page: int) -> Dict:
        info = {
            "has_numbers": bool(re.search(r'\d', chunk)),
            "has_dates": bool(re.search(r'\d{4}', chunk)),
        }
        return info

    def detect_table(self, chunk: str) -> bool:
        lines = chunk.split('\n')
        table_lines = 0
        for line in lines:
            if (line.count('   ') >= 1 or '\t' in line or line.count('|') >= 2 or sum(c.isdigit() for c in line) >= 3):
                table_lines += 1
        return table_lines >= 2

    def detect_section(self, chunk: str) -> str:
        # Look for ALL CAPS or numbered headings at the start
        match = re.match(r'([A-Z][A-Z\s\d\-\.]+):', chunk)
        if match:
            return match.group(1).strip()
        # Fallback: first 10 words if looks like a heading
        words = chunk.split()
        if len(words) < 15:
            return ' '.join(words[:10])
        return ""

    def extract_keywords(self, chunk: str) -> List[str]:
        # Simple frequency-based keyword extraction
        words = re.findall(r'\b\w{5,}\b', chunk.lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, c in sorted_words[:5]]

    def extract_images(self, output_dir: str = "data/images") -> List[Dict]:
        os.makedirs(output_dir, exist_ok=True)
        image_chunks = []
        for page_num, page in enumerate(self.doc):
            img_list = page.get_images(full=True)
            for img_index, img in enumerate(img_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_path = os.path.join(output_dir, f"page_{page_num+1}_img_{img_index+1}.{base_image['ext']}")
                with open(img_path, "wb") as img_file:
                    img_file.write(image_bytes)
                image = Image.open(io.BytesIO(image_bytes))
                ocr_configs = [
                    '--oem 3 --psm 6',
                    '--oem 3 --psm 3',
                    '--oem 1 --psm 6'
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
                    ocr_text = self.clean_text(ocr_text)
                    if len(ocr_text) > 20:
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
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / f"{Path(self.pdf_path).stem}_chunks.json"
        with open(output_path, 'w') as f:
            json.dump(chunks, f, indent=2)
        return str(output_path)

    def process(self) -> List[Dict]:
        text_data = self.extract_text()
        image_data = self.extract_images()
        all_chunks = text_data + image_data
        print(f"Processed {len(text_data)} text chunks and {len(image_data)} image chunks")
        print("First 5 chunks and their metadata:")
        for c in all_chunks[:5]:
            print(json.dumps(c, indent=2))
        print("--- Table Chunks (showing up to 5):")
        table_chunks = [c for c in all_chunks if c['metadata'].get('has_table')]
        for c in table_chunks[:5]:
            print(json.dumps(c, indent=2))
        self.save_to_json(all_chunks)
        return all_chunks