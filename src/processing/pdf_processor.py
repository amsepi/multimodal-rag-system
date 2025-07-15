import os
import requests
import time
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import MarkdownHeaderTextSplitter

LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
LLAMA_API_URL = "https://api.llamaindex.ai/api/parsing/parse"

class PDFProcessor:
    def __init__(self, file_path: str, chunk_size: int = 256, chunk_overlap: int = 100):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.markdown_path = str(Path(file_path).with_suffix(".md"))

    def parse_with_llamaparse(self) -> str:
        """Send the file to LlamaParse API and get Markdown output."""
        headers = {
            "x-api-key": LLAMA_API_KEY,
        }
        files = {"file": open(self.file_path, "rb")}
        data = {"output_format": "markdown"}
        response = requests.post(LLAMA_API_URL, headers=headers, files=files, data=data)
        if response.status_code != 200:
            raise RuntimeError(f"LlamaParse API error: {response.status_code} {response.text}")
        markdown = response.text
        with open(self.markdown_path, "w") as f:
            f.write(markdown)
        return self.markdown_path

    def chunk_markdown(self, markdown_path: str) -> List[Dict]:
        """Chunk the Markdown file using a Markdown-aware splitter."""
        with open(markdown_path, "r") as f:
            markdown = f.read()
        splitter = MarkdownHeaderTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_text(markdown)
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dicts.append({
                "content": chunk,
                "metadata": {
                    "source": os.path.basename(self.file_path),
                    "chunk_id": i,
                    "chunk_length": len(chunk),
                    "type": "markdown"
                }
            })
        return chunk_dicts

    def process(self) -> List[Dict]:
        print(f"Parsing {self.file_path} with LlamaParse...")
        md_path = self.parse_with_llamaparse()
        print(f"Saved Markdown to {md_path}")
        chunks = self.chunk_markdown(md_path)
        print(f"Chunked into {len(chunks)} markdown chunks.")
        # Optionally save to JSON
        out_json = str(Path(self.file_path).with_suffix("_chunks.json"))
        with open(out_json, "w") as f:
            import json
            json.dump(chunks, f, indent=2)
        print(f"Saved chunked data to {out_json}")
        return chunks