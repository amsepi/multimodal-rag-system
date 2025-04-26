import streamlit as st
from PIL import Image
import os
import time
import sys
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
st.set_page_config(
    page_title="UniRAG - University Document Assistant",
    page_icon="🎓",
    layout="centered"
)
#from src.config.settings import settings
from src.rag.embeddings import MultimodalEmbedder
from src.rag.vector_db import VectorStore
from src.rag.retrieval import Retriever
from src.llm.generation import ResponseGenerator
from src.processing.pdf_processor import PDFProcessor
from src.evaluation.visualisation import plot_embeddings, plot_search_results
from src.evaluation.metrics import Evaluator



# Create .streamlit folder and config.toml in project root
# Contents of .streamlit/config.toml:

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def init_system():
    """Initialize components without UI elements"""
    embedder = MultimodalEmbedder()
    vector_db = VectorStore(embedder)
    retriever = Retriever(vector_db, embedder)
    llm = ResponseGenerator()
    return embedder, vector_db, retriever, llm

def handle_pdf_processing(vector_db):
    """Separate function for PDF processing with UI"""
    if vector_db.text_collection.count() == 0 and vector_db.image_collection.count() == 0:
        with st.spinner("Processing initial PDFs..."):
            pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
            progress_bar = st.progress(0)
            
            for i, pdf_file in enumerate(pdf_files):
                try:
                    processor = PDFProcessor(os.path.join("data", pdf_file))
                    chunks = processor.process()
                    vector_db.add_documents(chunks)
                    progress_bar.progress((i+1)/len(pdf_files))
                except Exception as e:
                    st.error(f"Failed to process {pdf_file}: {str(e)}")
            progress_bar.empty()

def save_uploaded_file(uploaded_file):
    """Save uploaded file to data/ directory"""
    file_path = os.path.join("data", uploaded_file.name)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return file_path

def main():
    
    # Initialize system components
    embedder, vector_db, retriever, llm = init_system()
    
    handle_pdf_processing(vector_db)
    # Sidebar - File Management
    with st.sidebar:
        st.metric("Text Chunks", vector_db.text_collection.count())
        st.metric("Image Chunks", vector_db.image_collection.count())
    with st.sidebar:
        st.header("📂 Document Management")
        uploaded_files = st.file_uploader(
            "Upload university PDF documents",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            processing_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    file_path = save_uploaded_file(uploaded_file)
                    processor = PDFProcessor(file_path)
                    chunks = processor.process()
                    vector_db.add_documents(chunks)
                    processing_bar.progress((i+1)/len(uploaded_files))
                    st.success(f"Processed: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {str(e)}")
            processing_bar.empty()
            
        st.markdown("---")
        st.markdown("### System Status")
        st.metric("Text Chunks", vector_db.text_collection.count())
        st.metric("Image Chunks", vector_db.image_collection.count())
    
    # Main Interface
    st.title("🎓 UniRAG - University Document Assistant")
    st.markdown("Ask questions about financial reports, academic programs, and campus development plans")
    
    # Query Input
    query_tab, image_tab = st.tabs(["💬 Text Query", "🖼️ Image Query"])
    
    with query_tab:
        query_text = st.text_input("Enter your question:", placeholder="e.g., What's the 2024 research budget?")
        
    with image_tab:
        uploaded_image = st.file_uploader("Upload an image query", type=["png", "jpg", "jpeg"])
    
    # Search Button
    if st.button("🔍 Search", use_container_width=True):
        if not query_text.strip() and not uploaded_image:
            st.warning("Please enter a question or upload an image")
            st.stop()
            
        with st.spinner("Analyzing university documents..."):
            start_time = time.time()
            
            try:
                # Retrieve context
                if query_text.strip():
                    context = retriever.retrieve(query_text)
                    query_type = "text"
                else:
                    image = Image.open(uploaded_image)
                    context = retriever.retrieve(image)
                    query_type = "image"
                
                # Generate response
                if context:
                    response = llm.generate_response(
                        query_text if query_type == "text" else "Explain this image",
                        context
                    )
                    
                    # Display results
                    # Replace the existing response display code with:
                    st.subheader("📝 Answer")
                    st.markdown(f"""
                    <div class="response-box">
                    {response}
                    </div>
                    """, unsafe_allow_html=True)

                    # Show sources with improved styling
                    st.subheader("🔍 Source References")
                    for i, chunk in enumerate(context[:3]):
                        with st.expander(f"📑 Source {i+1} - {chunk['metadata']['source']} (Page {chunk['metadata']['page']})", expanded=True if i==0 else False):
                            st.caption(f"**Relevance Score:** {chunk['score']:.2f}")
                            if chunk["metadata"]["type"] == "image":
                                st.image(chunk["metadata"]["image_path"])
                            st.markdown(f"""
                            <div class="source-content">
                            {chunk["content"][:500]}{'...' if len(chunk["content"]) > 500 else ''}
                            </div>
                            """, unsafe_allow_html=True)
                                        
                    # Show sources
                    st.subheader("🔍 Sources")
                    for i, chunk in enumerate(context[:3]):  # Show top 3 sources
                        with st.expander(f"Source {i+1} from {chunk['metadata']['source']} (Page {chunk['metadata']['page']})"):
                            if chunk["metadata"]["type"] == "image":
                                st.image(chunk["metadata"]["image_path"])
                            st.write(chunk["content"][:500] + "...")
                            
                    # Performance metrics
                    st.caption(f"⏱️ Response generated in {time.time()-start_time:.1f}s | Sources found: {len(context)}")
                else:
                    st.error("No relevant information found in university documents")
                    
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
            
eval_tab, vis_tab = st.tabs(["Evaluation Metrics", "Visualization"])

with vis_tab:
    st.header("Embedding Space Visualization")
    
    # Initialize system components
    _, vector_db, _, _ = init_system()
    
    try:
        # Initialize empty arrays with proper dimensions
        text_embeddings = np.empty((0, 384))  # Sentence-BERT dimension
        image_embeddings = np.empty((0, 512))  # CLIP dimension
        text_labels = []
        image_labels = []

        # Load text embeddings
        if vector_db.text_collection.count() > 0:
            text_data = vector_db.text_collection.get(include=["embeddings"])
            if text_data["embeddings"]:
                text_embeddings = np.array(text_data["embeddings"])
                text_labels = [f"Text: {m['source']} (p.{m['page']})" 
                             for m in vector_db.text_collection.get()["metadatas"]]

        # Load image embeddings
        if vector_db.image_collection.count() > 0:
            image_data = vector_db.image_collection.get(include=["embeddings"])
            if image_data["embeddings"]:
                image_embeddings = np.array(image_data["embeddings"])
                image_labels = [f"Image: {m['source']} (p.{m['page']})" 
                              for m in vector_db.image_collection.get()["metadatas"]]

        # Create visualization
        if text_embeddings.size > 0 or image_embeddings.size > 0:
            fig = plot_embeddings(text_embeddings, image_embeddings, 
                                 text_labels, image_labels)
            st.plotly_chart(fig)
        else:
            st.warning("Process documents to enable visualization")

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")

# Modified Evaluation Tab
with eval_tab:
    st.header("System Evaluation Metrics")
    
    if st.button("Run Comprehensive Evaluation"):
        # Reinitialize system components
        _, _, retriever, llm = init_system()
        evaluator = Evaluator(retriever, llm)
        
        try:
            with st.expander("📈 Retrieval Performance", expanded=True):
                test_queries = [
                    ("What was 2023 revenue?", "financials.pdf"),
                    ("Explain FYP requirements", "FYPHandbook2023.pdf")
                ]
                hit_rate = evaluator.calculate_hit_rate(
                    [q[0] for q in test_queries],
                    [q[1] for q in test_queries]
                )
                st.metric("Document Hit Rate (Top-5)", f"{hit_rate*100:.1f}%")

            with st.expander("📝 Generation Quality", expanded=False):
                ref_answer = "The 2023 revenue was $5.2 million according to page 12."
                test_response = llm.generate_response(
                    "What was the 2023 revenue?",
                    retriever.retrieve("What was the 2023 revenue?")
                )
                scores = evaluator.evaluate_response(test_response, ref_answer)
                st.write(f"""
                - **BLEU-4 Score**: {scores['bleu-4']:.2f}
                - **ROUGE-L F1**: {scores['rouge-l']:.2f}
                """)

            with st.expander("⏱️ Performance Benchmarks", expanded=False):
                latency = evaluator.measure_latency("What was 2023 revenue?")
                st.write(f"""
                - **Average Response Time**: {latency:.2f}s
                - **Embedding Speed**: {len(text_embeddings)/latency:.0f} docs/s
                """)

        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")

                
                

if __name__ == "__main__":
    main()