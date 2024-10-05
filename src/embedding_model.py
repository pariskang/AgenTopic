# src/embedding_model.py

from src.utils import create_custom_embedding_model
import src.config as config

def initialize_embedding_model(model_name):
    """
    Initializes the embedding model.
    """
    embedding_model = create_custom_embedding_model(model_name)
    return embedding_model
        
