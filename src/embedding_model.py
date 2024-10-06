# src/embedding_model.py

from src.utils import create_custom_embedding_model
import src.config as config

def initialize_embedding_model(model_name):
    """
    Initialize the embedding model.

    Args:
        model_name (str): Name of the embedding model.

    Returns:
        CustomEmbeddingModel: Instance of the custom embedding model.
    """
    embedding_model = create_custom_embedding_model(model_name)
    return embedding_model
