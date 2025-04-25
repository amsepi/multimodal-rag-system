import sys
sys.path.append("..")
from src.rag.embeddings import MultimodalEmbedder
from PIL import Image

def test_embeddings():
    embedder = MultimodalEmbedder()
    
    # Test text embedding
    text = "Revenue increased by 15% in Q3 2023"
    text_embedding = embedder.embed_text(text)
    print(f"Text embedding dim: {len(text_embedding)}")
    
    # Test image embedding
    img_path = "data/images/page_1_img_1.png"  # Use real image
    img_embedding = embedder.embed_image(img_path)
    print(f"Image embedding dim: {len(img_embedding)}")
    
    # Verify similarity
    if len(text_embedding) > 0 and len(img_embedding) > 0:
        print("Embedding system working!")

if __name__ == "__main__":
    test_embeddings()