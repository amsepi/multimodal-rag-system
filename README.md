# Multimodal RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system that supports both text and image inputs, built with Streamlit, OpenAI, and LangChain.

## Features

- **Multimodal Support**: Process both text and image inputs
- **PDF Processing**: Extract and process text from PDF documents
- **Vector Storage**: Store and retrieve embeddings using ChromaDB
- **Evaluation Module**: Comprehensive evaluation capabilities including:
  - BLEU score calculation
  - ROUGE metrics
  - Visualization of embeddings
  - Test question evaluation
- **Streamlit Interface**: User-friendly web interface for interaction

## Project Structure

```
multimodal-rag-system/
├── src/
│   ├── app.py                 # Main Streamlit application
│   ├── config/               # Configuration files
│   ├── processing/           # Data processing modules
│   └── evaluation/          # Evaluation module
│       ├── metrics.py       # Evaluation metrics (BLEU, ROUGE)
│       ├── visualisation.py # Embedding visualization
│       └── test_questions.json # Test questions for evaluation
├── data/                    # Directory for PDF files
├── .streamlit/             # Streamlit configuration
├── tests/                  # Test files
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/amsepi/multimodal-rag-system.git
cd multimodal-rag-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a `.streamlit/secrets.toml` file with your OpenAI API key:
```toml
openai_api_key = "your-api-key-here"
```

## Usage

1. Place your PDF files in the `data` directory
2. Run the Streamlit app:
```bash
streamlit run src/app.py
```

3. Access the application through your web browser at `http://localhost:8501`

## Evaluation

The system includes a comprehensive evaluation module that provides:

- **Text Quality Metrics**: BLEU and ROUGE scores for generated responses
- **Embedding Visualization**: Visual representation of document embeddings
- **Test Questions**: Pre-defined test questions for system evaluation

To run evaluations:
1. Navigate to the "Evaluation" tab in the Streamlit interface
2. Select the evaluation metrics to compute
3. View the results and visualizations

## Dependencies

- streamlit
- openai
- langchain
- chromadb
- nltk
- rouge-score
- torchvision
- Other dependencies listed in requirements.txt

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
