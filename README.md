# Multimodal RAG System

A powerful Retrieval-Augmented Generation (RAG) system that supports multimodal document processing and intelligent question answering. This system combines text and image processing capabilities with advanced language models to provide comprehensive answers to user queries.

## Features

- **Multimodal Document Processing**: Handles both text and image content from documents
- **Advanced RAG Pipeline**: Implements a sophisticated retrieval-augmented generation system
- **Streamlit Web Interface**: User-friendly web application for document upload and querying
- **Document Embedding**: Utilizes sentence transformers for efficient document indexing
- **Vector Database**: Uses ChromaDB for storing and retrieving document embeddings
- **Language Model Integration**: Leverages OpenAI's models for generating responses
- **Visualization Tools**: Includes UMAP for document visualization and Plotly for interactive plots

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multimodal-rag-system.git
cd multimodal-rag-system
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
```

## Project Structure

```
multimodal-rag-system/
├── src/
│   ├── app.py              # Streamlit application
│   ├── config/             # Configuration files
│   ├── llm/                # Language model integration
│   ├── processing/         # Document processing modules
│   └── rag/                # RAG pipeline implementation
├── tests/                  # Test files
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run src/app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload documents (PDF, images, or text files)

4. Ask questions about the uploaded documents

## Dependencies

- streamlit: Web application framework
- openai: OpenAI API integration
- pymupdf: PDF processing
- pytesseract: OCR for image processing
- sentence-transformers: Document embedding
- chromadb: Vector database
- python-dotenv: Environment variable management
- pillow: Image processing
- langchain: LLM framework
- plotly: Interactive visualization
- umap-learn: Dimensionality reduction
- rouge: Text evaluation metrics
- nltk: Natural language processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
