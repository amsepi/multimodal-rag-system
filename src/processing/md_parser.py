import os
from pathlib import Path
from llama_cloud_services import LlamaParse

def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Parse a PDF file to Markdown using LlamaParse. If the Markdown file already exists, skip parsing.
    Returns the path to the Markdown file.
    """
    md_path = str(Path(pdf_path).parent / (Path(pdf_path).stem + ".md"))
    if os.path.exists(md_path):
        print(f"Markdown file {md_path} already exists. Skipping LlamaParse API call.")
        return md_path
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        verbose=True,
        language="en"
    )
    print(f"Parsing {pdf_path} with LlamaParse (official client)...")
    result = parser.parse(pdf_path)
    markdown_nodes = result.get_markdown_nodes(split_by_page=True)
    print(f"Received {len(markdown_nodes)} markdown nodes from LlamaParse.")
    all_markdown = "\n\n".join([node.text for node in markdown_nodes])
    with open(md_path, "w") as f:
        f.write(all_markdown)
    print(f"Saved combined Markdown to {md_path}")
    return md_path 