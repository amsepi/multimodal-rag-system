import os
import json
from pathlib import Path
from typing import List, Dict
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

def chunk_markdown_sentences(md_path: str, chunk_size: int = 256, chunk_overlap: int = 50) -> str:
    """
    Chunk a Markdown file into sentence-based chunks. Returns the path to the chunks JSON file.
    """
    with open(md_path, 'r') as f:
        text = f.read()
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0
    chunk_id = 0
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        if current_len + len(sent) <= chunk_size or not current_chunk:
            current_chunk.append(sent)
            current_len += len(sent)
            i += 1
        else:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'metadata': {
                    'source': os.path.basename(md_path),
                    'chunk_id': chunk_id,
                    'chunk_length': len(chunk_text),
                    'type': 'text',
                    'page': None
                }
            })
            chunk_id += 1
            # Overlap
            overlap_sents = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else []
            current_chunk = overlap_sents.copy()
            current_len = sum(len(s) for s in current_chunk)
    # Add last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'content': chunk_text,
            'metadata': {
                'source': os.path.basename(md_path),
                'chunk_id': chunk_id,
                'chunk_length': len(chunk_text),
                'type': 'text',
                'page': None
            }
        })
    out_json = str(Path(md_path).parent / (Path(md_path).stem + '_chunks.json'))
    with open(out_json, 'w') as f:
        json.dump(chunks, f, indent=2)
    print(f"Chunked into {len(chunks)} sentence-based chunks. Saved to {out_json}")
    return out_json 