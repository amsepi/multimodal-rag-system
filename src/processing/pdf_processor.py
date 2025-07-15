import os
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import MarkdownHeaderTextSplitter
from llama_cloud_services import LlamaParse

class PDFProcessor:
    def __init__(self, file_path: str, chunk_size: int = 256, chunk_overlap: int = 100):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def process(self) -> List[Dict]:
        md_path = str(Path(self.file_path).parent / (Path(self.file_path).stem + ".md"))
        # If the markdown file already exists, skip parsing
        if os.path.exists(md_path):
            print(f"Markdown file {md_path} already exists. Skipping LlamaParse API call.")
            with open(md_path, "r") as f:
                all_markdown = f.read()
            # We don't have page offsets if skipping, so fallback to page=None
            page_offsets = None
        else:
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
                verbose=True,
                language="en"
            )
            print(f"Parsing {self.file_path} with LlamaParse (official client)...")
            result = parser.parse(self.file_path)
            # Get markdown nodes (one per page)
            markdown_nodes = result.get_markdown_nodes(split_by_page=True)
            print(f"Received {len(markdown_nodes)} markdown nodes from LlamaParse.")
            # Combine all markdown for chunking, and record page offsets
            all_markdown = ""
            page_offsets = []  # list of (start, end) tuples for each page
            curr = 0
            for node in markdown_nodes:
                page_text = node.text
                start = curr
                all_markdown += page_text + "\n\n"
                curr = len(all_markdown)
                end = curr
                page_offsets.append((start, end))
            # Save the combined markdown for reference
            with open(md_path, "w") as f:
                f.write(all_markdown)
            print(f"Saved combined Markdown to {md_path}")

        # Only header-based splitting
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]
        header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        sections = header_splitter.split_text(all_markdown)

        chunk_dicts = []
        for i, section in enumerate(sections):
            text = section.page_content if hasattr(section, 'page_content') else str(section)
            # Determine page number
            if page_offsets is not None:
                # Find the first page whose offset range contains the start of this chunk
                chunk_start = all_markdown.find(text)
                page_num = None
                for idx, (start, end) in enumerate(page_offsets):
                    if start <= chunk_start < end:
                        page_num = idx + 1  # 1-based page number
                        break
            else:
                page_num = None
            chunk_dicts.append({
                "content": text,
                "metadata": {
                    "source": os.path.basename(self.file_path),
                    "chunk_id": i,
                    "chunk_length": len(text),
                    "type": "text",
                    "page": page_num
                }
            })
        print(f"Chunked into {len(chunk_dicts)} markdown header-based chunks.")

        # Optionally save to JSON
        out_json = str(Path(self.file_path).parent / (Path(self.file_path).stem + "_chunks.json"))
        with open(out_json, "w") as f:
            import json
            json.dump(chunk_dicts, f, indent=2)
        print(f"Saved chunked data to {out_json}")
        return chunk_dicts