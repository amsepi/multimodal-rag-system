import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import Union

class MultimodalEmbedder:
    def __init__(self):
        # Text embedding model (upgraded)
        self.text_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Image embedding model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        
        
        # Device optimization for M1/M2
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.clip_model = self.clip_model.to(self.device)
       

    def embed_text(self, text: str) -> list:
        """Embed text chunks using Sentence-BERT"""
        return self.text_model.encode(text).tolist()

    def embed_image(self, image_path: str) -> list:
        """Embed images using CLIP"""
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