import os
import openai
from typing import List
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class MultimodalEmbedder:
    def __init__(self):
        # OpenAI API key from environment or Streamlit secrets
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.text_model = "text-embedding-3-large"
        # Image embedding model (unchanged)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        import torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.clip_model = self.clip_model.to(self.device)

    def embed_text(self, text: str) -> list:
        """Embed text using OpenAI's text-embedding-3-large API"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment.")
        client = openai.OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            input=text,
            model=self.text_model
        )
        return response.data[0].embedding

    def embed_image(self, image_path: str) -> list:
        try:
            image = Image.open(image_path)
            inputs = self.clip_processor(
                images=image, 
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
            return features.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return []
        

